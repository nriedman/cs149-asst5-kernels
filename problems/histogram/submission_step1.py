from task import input_t, output_t
import torch
import triton
import triton.language as tl


@triton.jit
def histogram_kernel(
    array_ptr,          # Pointer to input array [length, num_channels]
    histogram_ptr,      # Pointer to output histogram [num_channels, num_bins]
    length,             # Number of elements per channel
    num_channels,       # Number of channels
    num_bins,           # Number of histogram bins
    BLOCK_SIZE: tl.constexpr,  # Block size for processing elements
):
    """
    Triton kernel that computes histogram for one channel per program.
    Each program (thread block) handles one channel independently.
    
    Strategy: One thread block per channel. Each block iterates through all
    elements in its channel and uses atomic operations to update shared bins.
    """
    # Get the channel ID for this program
    channel_id = tl.program_id(0)
    
    # Base pointer for this channel's histogram output
    histogram_base = histogram_ptr + channel_id * num_bins
    
    # Process all elements in this channel in chunks
    for idx in range(0, length, BLOCK_SIZE):
        # Calculate offsets for this block of elements
        offsets = idx + tl.arange(0, BLOCK_SIZE)
        mask = offsets < length
        
        # Load values from this channel
        # Array layout: [length, num_channels], so element at (row, col) is at row * num_channels + col
        data_ptrs = array_ptr + offsets * num_channels + channel_id
        values = tl.load(data_ptrs, mask=mask, other=0).to(tl.int32)
        
        # Atomically increment histogram bins for each value
        # We need to iterate because Triton doesn't support scatter-add on vectors directly
        for i in range(BLOCK_SIZE):
            offset = idx + i
            if offset < length:
                # Load the actual value (not using vectorized load to ensure correctness)
                val_ptr = array_ptr + offset * num_channels + channel_id
                val = tl.load(val_ptr).to(tl.int32)
                # Atomically increment the corresponding bin
                tl.atomic_add(histogram_base + val, 1)


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
    
    # Get dimensions
    length, num_channels = array.shape
    
    # Allocate output histogram
    output_dtype = torch.int32
    histogram = torch.zeros(num_channels, num_bins, dtype=output_dtype, device=array.device)
    
    # Launch kernel with one program per channel
    BLOCK_SIZE = 1024
    grid = (num_channels,)
    
    histogram_kernel[grid](
        array,
        histogram,
        length,
        num_channels,
        num_channels,
        num_bins,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return histogram
