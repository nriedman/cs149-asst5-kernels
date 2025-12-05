from task import input_t, output_t
import torch
import triton
import triton.language as tl

@triton.jit
def find_starts(
    sorted_bin_ids_ptr,   # pointer to sorted_bin_ids (1D)
    bin_starts_ptr,       # pointer to bin_starts (1D)
    N,                    # length of sorted_bin_ids (scalar)
    BLOCK_SIZE: tl.constexpr
):
    # program/block id
    pid = tl.program_id(0)
    # lane indices within the program (0..BLOCK_SIZE-1)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # vector of global indices
    mask = offs < N

    # load current and previous ids (contiguous)
    ids = tl.load(sorted_bin_ids_ptr + offs, mask=mask, other=0).to(tl.uint8)
    prev = tl.load(sorted_bin_ids_ptr + (offs - 1), mask=mask & (offs > 0), other=-1).to(tl.uint8)

    is_start = mask & ((offs == 0) | (ids != prev))
    # guard pointer arithmetic so masked lanes don't produce invalid addresses
    target_ptr = bin_starts_ptr + tl.where(is_start, ids, 0)
    tl.store(target_ptr, offs.to(tl.uint8), mask=is_start)


@triton.jit
def bin_sizes(
    bin_starts_ptr,   # 1D uint8 of size num_bins
    histogram_ptr,    # 1D uint8 of size num_bins
    num_items,        # int32
    num_bins,         # int32
    BLOCK_SIZE: tl.constexpr,
    MAX_BINS: tl.constexpr,  # must be >= num_bins
):
    pid = tl.program_id(0)
    tid = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = tid < num_bins
    starts = tl.load(bin_starts_ptr + tid, mask=mask, other=-1)

    # empty bins
    tl.store(histogram_ptr + tid, 0, mask=mask & (starts == -1))

    active = mask & (starts != -1)

    # forward search for next valid start
    next_idx = tid + 1
    next_valid = tl.full((BLOCK_SIZE,), -1, dtype=tl.uint8)

    for _ in range(MAX_BINS):
        searching = active & (next_valid == -1) & (next_idx < num_bins)
        cand = tl.load(bin_starts_ptr + next_idx, mask=searching, other=-1)
        found = searching & (cand != -1)
        next_valid = tl.where(found, next_idx.to(tl.uint8), next_valid)
        next_idx = next_idx + tl.where(searching, 1, 0)

    has_next = active & (next_valid != -1)
    next_starts = tl.load(bin_starts_ptr + next_valid, mask=has_next, other=-1)
    sizes = tl.where(has_next, next_starts - starts, num_items - starts)
    tl.store(histogram_ptr + tid, sizes, mask=active)


def custom_kernel(data):
    array, num_bins = data
    length, num_channels = array.shape

    device = array.device if array.is_cuda else "cuda"
    array = array.to(device)

    output_dtype = torch.uint8
    histogram = torch.zeros((num_channels, num_bins), dtype=output_dtype, device=array.device)

    # Make per-channel slices contiguous by transposing
    arr_T = array.transpose(0, 1).contiguous()  # shape: (C, L)
    sorted_ids, _ = torch.sort(arr_T, dim=1)    # still contiguous per row

    histogram = torch.zeros((num_channels, num_bins), dtype=torch.uint8, device=device)

    BLOCK_ITEMS = 1024
    BLOCK_BINS = 256
    num_item_blocks = triton.cdiv(length, BLOCK_ITEMS)
    num_bin_blocks = triton.cdiv(num_bins, BLOCK_BINS)
    MAX_BINS = max(256, int(num_bins))  # constexpr >= num_bins

    for c in range(num_channels):
        bin_starts = torch.full((num_bins,), -1, dtype=torch.uint8, device=device)

        find_starts[(num_item_blocks,)](
            sorted_ids[c, :],
            bin_starts,
            length,
            BLOCK_SIZE=BLOCK_ITEMS,
        )

        bin_sizes[(num_bin_blocks,)](
            bin_starts,
            histogram[c, :],
            length,
            num_bins,
            BLOCK_SIZE=BLOCK_BINS,
            MAX_BINS=MAX_BINS,
        )

    return histogram
