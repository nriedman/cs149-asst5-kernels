import torch
import triton
import triton.language as tl
from task import input_t, output_t


@triton.jit
def histogram_privatized_kernel(
    # Pointers to inputs/outputs
    data_ptr,              # [length, num_channels]
    temp_histogram_ptr,    # [num_blocks, num_channels, num_bins] - private histograms
    # Dimensions
    length,
    num_channels,
    num_bins,
    # Block configuration
    BLOCK_SIZE: tl.constexpr,
):
    """
    Iteration 1: Privatization with Shared Memory - Phase 1
    
    Each program block processes a chunk of one channel and accumulates
    into its own private histogram in temp storage.
    
    Key optimization: Multiple blocks work on the same channel, each with
    their own private histogram, eliminating atomic contention during accumulation.
    """
    # Get program IDs
    block_id = tl.program_id(0)      # Which block (chunk of data)
    channel_id = tl.program_id(1)    # Which channel
    
    # Calculate which portion of data this block processes
    start_idx = block_id * BLOCK_SIZE
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < length
    
    # Load data for this channel
    # Data layout: [length, num_channels]
    data_indices = offsets * num_channels + channel_id
    values = tl.load(data_ptr + data_indices, mask=mask, other=0).to(tl.int32)
    
    # Private histogram for this block and channel
    # Shape: [num_blocks, num_channels, num_bins]
    hist_base = temp_histogram_ptr + (block_id * num_channels + channel_id) * num_bins
    
    # Accumulate values into private histogram
    # Using vectorized approach when possible
    for i in range(BLOCK_SIZE):
        if start_idx + i < length:
            val = tl.load(data_ptr + (start_idx + i) * num_channels + channel_id).to(tl.int32)
            # No atomics needed here! Each block has its own private histogram
            # This is the key optimization - no contention during accumulation
            tl.atomic_add(hist_base + val, 1)


@triton.jit
def histogram_reduce_kernel(
    # Pointers
    temp_histogram_ptr,    # [num_blocks, num_channels, num_bins]
    histogram_ptr,         # [num_channels, num_bins]
    # Dimensions
    num_blocks,
    num_channels,
    num_bins,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Iteration 1: Privatization with Shared Memory - Phase 2
    
    Reduce all private histograms into the final global histogram.
    Each program handles one (channel, bin) combination.
    """
    channel_id = tl.program_id(0)
    bin_start = tl.program_id(1) * BLOCK_SIZE
    
    # Process multiple bins per program
    bin_offsets = bin_start + tl.arange(0, BLOCK_SIZE)
    bin_mask = bin_offsets < num_bins
    
    # Sum across all blocks for this channel and these bins
    for block_id in range(num_blocks):
        # Load from private histogram: temp_histogram[block_id, channel_id, bin_offsets]
        temp_base = temp_histogram_ptr + (block_id * num_channels + channel_id) * num_bins
        temp_indices = temp_base + bin_offsets
        counts = tl.load(temp_indices, mask=bin_mask, other=0)
        
        # Add to final histogram
        hist_base = histogram_ptr + channel_id * num_bins
        hist_indices = hist_base + bin_offsets
        tl.atomic_add(hist_indices, counts, mask=bin_mask)


def custom_kernel(data: input_t) -> output_t:
    """
    Iteration 1: Privatization with Shared Memory (Triton Implementation)
    
    Goal: Reduce global atomic contention
    - Each program block maintains its own private histogram in temporary storage
    - Blocks accumulate to their private histograms (no contention!)
    - At the end, reduce all private histograms to global memory
    - Expected: 5-10x speedup
    - Why: Eliminates atomic contention during the accumulation phase
    
    Two-phase approach:
    Phase 1: Each block computes its own private histogram for a chunk of data
    Phase 2: Reduce all private histograms into the final output
    
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
    
    # Ensure data is on CUDA
    if not array.is_cuda:
        array = array.cuda()
    
    # Configuration
    BLOCK_SIZE = 1024  # Each block processes 1024 elements
    num_blocks = triton.cdiv(length, BLOCK_SIZE)
    
    # Allocate temporary storage for private histograms
    # Shape: [num_blocks, num_channels, num_bins]
    temp_histogram = torch.zeros(
        num_blocks, num_channels, num_bins, 
        dtype=torch.int32, 
        device=array.device
    )
    
    # Allocate final output histogram
    histogram = torch.zeros(
        num_channels, num_bins, 
        dtype=torch.int32, 
        device=array.device
    )
    
    # Phase 1: Accumulate into private histograms
    # Grid: (num_blocks, num_channels)
    # Each program handles one block of data for one channel
    grid_phase1 = (num_blocks, num_channels)
    histogram_privatized_kernel[grid_phase1](
        array,
        temp_histogram,
        length,
        num_channels,
        num_bins,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Phase 2: Reduce private histograms to final output
    # Grid: (num_channels, num_bins // REDUCE_BLOCK_SIZE)
    # Each program reduces one bin across all blocks
    REDUCE_BLOCK_SIZE = 64
    num_bin_blocks = triton.cdiv(num_bins, REDUCE_BLOCK_SIZE)
    grid_phase2 = (num_channels, num_bin_blocks)
    histogram_reduce_kernel[grid_phase2](
        temp_histogram,
        histogram,
        num_blocks,
        num_channels,
        num_bins,
        BLOCK_SIZE=REDUCE_BLOCK_SIZE,
    )
    
    return histogram
