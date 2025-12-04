from task import input_t, output_t
import torch
import triton
import triton.language as tl


@triton.jit
def find_starts(
    # Pointers to inputs/outputs
    sorted_bin_ids,
    bin_starts,

    # Dims
    N,

    # Block config
    BLOCK_SIZE: tl.constexpr
):
    block_id = tl.program_id(0)
    thread_id = tl.program_id(1)
    thread_idx = block_id * BLOCK_SIZE + thread_id

    if thread_idx >= N:
        return

    if thread_idx == 0 or sorted_bin_ids[thread_idx] != sorted_bin_ids[thread_idx - 1]:
        bin_starts[sorted_bin_ids[thread_idx]] = thread_idx


@triton.jit
def bin_sizes(
    # Pointers to inputs/outputs
    bin_starts,
    histogram_bins,

    # Dims
    num_items,
    num_bins,

    # Block config
    BLOCK_SIZE: tl.constexpr
):
    block_id = tl.program_id(0)
    thread_id = tl.program_id(1)
    thread_idx = block_id * BLOCK_SIZE + thread_id

    if thread_idx >= num_bins:
        return
    
    if bin_starts[thread_idx] == -1:
        histogram_bins[thread_idx] = 0
    else:
        next_index = thread_idx + 1
        while next_index < num_bins and bin_starts[next_index] == -1:
            next_index += 1
        
        if next_index < num_bins:
            histogram_bins[thread_idx] = bin_starts[next_index] - bin_starts[thread_idx]
        else:
            histogram_bins[thread_idx] = num_items - bin_starts[thread_idx]


def custom_kernel(data: input_t) -> output_t:
    """
    Args:
        data:
            Tuple of (array, num_bins) where:
                array:    Tensor of shape [length, num_channels], dtype=uint8, containing
                          integer values in the range [0, num_bins - 1]
                num_bins: Number of histogram bins (defines allowed value range)

    Returns:
        histogram:
            Tensor of shape [num_channels, num_bins], where histogram[c][b]
            contains the count of how many times value b appears in channel c.
    """
    array, num_bins = data
    
    # get dimensions
    length, num_channels = array.shape

    if not array.is_cuda:
        array = array.cuda()
    
    # allocate output histogram
    output_dtype = torch.int32
    histogram = torch.zeros(num_channels, num_bins, dtype=output_dtype, device=array.device)

    # temp buffers
    bin_starts = torch.full((length,), -1, dtype=output_dtype, device=array.device)

    # thread blocks
    BLOCK_SIZE = 1024  # Each block processes 1024 elements
    num_item_blocks = triton.cdiv(length, BLOCK_SIZE)
    num_bin_blocks = triton.cdiv(num_bins, BLOCK_SIZE)
    
    # sort the array long the (length,) dimension
    sorted_bin_ids, _ = torch.sort(array, dim=0)
    

    for c in range(num_channels):
        # 
        find_starts[(num_item_blocks, BLOCK_SIZE)](sorted_bin_ids, )


    for b in range(length):
        for c in range(num_channels):
            histogram[c][array[b][c]] += 1
    
    return histogram
