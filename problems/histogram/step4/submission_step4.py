import torch
import triton
import triton.language as tl
from task import input_t, output_t


@triton.jit
def histogram_per_thread_kernel(
    # Pointers to inputs/outputs
    data_ptr,              # [length, num_channels]
    temp_histogram_ptr,    # [num_programs, num_bins] - per-thread private histograms
    # Dimensions
    length,
    num_channels,
    num_bins,
    channel_id,
    # Block configuration
    ELEMENTS_PER_THREAD: tl.constexpr,
):
    """
    Iteration 2: Per-Thread Private Histograms - Phase 1
    
    Each thread (program) maintains its own private histogram in registers/local memory.
    Process multiple elements per thread sequentially with NO ATOMICS during accumulation!
    
    Key optimization: Complete elimination of atomic operations during accumulation phase.
    Each thread has exclusive access to its own histogram.
    """
    # Get unique program/thread ID
    program_id = tl.program_id(0)
    
    # Step 1: Each thread processes ELEMENTS_PER_THREAD elements sequentially
    # Calculate starting position for this thread
    start_idx = program_id * ELEMENTS_PER_THREAD
    
    # Step 2: Initialize output histogram location for this thread
    hist_base = temp_histogram_ptr + program_id * num_bins
    
    # Process elements sequentially and accumulate directly to temp storage
    # Since each thread has its own histogram section, no atomics needed!
    for i in range(ELEMENTS_PER_THREAD):
        idx = start_idx + i
        
        if idx < length:
            # Load value from data[idx, channel_id]
            data_offset = idx * num_channels + channel_id
            value = tl.load(data_ptr + data_offset).to(tl.int32)
            
            # Increment thread-private histogram (no atomics needed!)
            # Each thread writes to its own memory region
            bin_addr = hist_base + value
            old_count = tl.load(bin_addr)
            tl.store(bin_addr, old_count + 1)


@triton.jit
def histogram_reduce_per_thread_kernel(
    # Pointers
    temp_histogram_ptr,    # [num_programs, num_bins] - per-thread histograms
    histogram_ptr,         # [num_channels, num_bins]
    # Dimensions
    num_programs,
    num_bins,
    channel_id,
    # Block size for reduction
    BINS_PER_PROGRAM: tl.constexpr,
):
    """
    Iteration 2: Per-Thread Private Histograms - Phase 2
    
    Reduce all per-thread private histograms into the final global histogram.
    Each program handles a range of bins and sums across all threads.
    """
    program_id = tl.program_id(0)
    
    # Calculate which bins this program handles
    bin_start = program_id * BINS_PER_PROGRAM
    bin_offsets = bin_start + tl.arange(0, BINS_PER_PROGRAM)
    bin_mask = bin_offsets < num_bins
    
    # Initialize accumulator for these bins
    bin_sums = tl.zeros([BINS_PER_PROGRAM], dtype=tl.int32)
    
    # Sum across all per-thread histograms
    for thread_id in range(num_programs):
        # Load this thread's histogram for our bins
        temp_base = temp_histogram_ptr + thread_id * num_bins
        temp_indices = temp_base + bin_offsets
        counts = tl.load(temp_indices, mask=bin_mask, other=0)
        
        # Accumulate (no atomics needed during this phase)
        bin_sums += counts
    
    # Write final sums to global histogram
    hist_base = histogram_ptr + channel_id * num_bins
    hist_indices = hist_base + bin_offsets
    tl.store(hist_indices, bin_sums, mask=bin_mask)


def custom_kernel(data: input_t) -> output_t:
    """
    Iteration 2: Per-Thread Private Histograms (Triton Implementation)
    
    Goal: Eliminate atomics during accumulation phase
    - Each thread maintains its own histogram in registers/local memory
    - Process multiple elements per thread sequentially (NO ATOMICS!)
    - Reduce per-thread histograms to global memory
    - Expected: 2-3x additional speedup over Iteration 1
    - Why: Zero atomic contention during main accumulation
    - Tradeoff: More memory usage, but eliminates all contention
    
    Two-phase approach:
    Phase 1: Each thread builds its own private histogram (pure register operations)
    Phase 2: Reduce all per-thread histograms into final output
    
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
    
    # Allocate final output histogram
    histogram = torch.zeros(
        num_channels, num_bins, 
        dtype=torch.int32, 
        device=array.device
    )
    
    # Process each channel independently
    for channel_id in range(num_channels):
        # Configuration for this channel
        ELEMENTS_PER_THREAD = 256  # Each thread processes 256 elements
        num_programs = triton.cdiv(length, ELEMENTS_PER_THREAD)
        
        # Allocate temporary storage for per-thread private histograms
        # Shape: [num_programs, num_bins]
        # Each program (thread) gets its own histogram
        temp_histogram = torch.zeros(
            num_programs, num_bins,
            dtype=torch.int32,
            device=array.device
        )
        
        # Phase 1: Each thread builds its own private histogram
        # Grid: (num_programs,) - one program per thread
        # Each program processes ELEMENTS_PER_THREAD elements with NO ATOMICS
        grid_phase1 = (num_programs,)
        histogram_per_thread_kernel[grid_phase1](
            array,
            temp_histogram,
            length,
            num_channels,
            num_bins,
            channel_id,
            ELEMENTS_PER_THREAD=ELEMENTS_PER_THREAD,
        )
        
        # Phase 2: Reduce per-thread histograms to final output
        # Grid: (num_reduction_programs,)
        # Each program reduces a range of bins across all threads
        BINS_PER_PROGRAM = 32  # Each reduction program handles 32 bins
        num_reduction_programs = triton.cdiv(num_bins, BINS_PER_PROGRAM)
        grid_phase2 = (num_reduction_programs,)
        
        histogram_reduce_per_thread_kernel[grid_phase2](
            temp_histogram,
            histogram,
            num_programs,
            num_bins,
            channel_id,
            BINS_PER_PROGRAM=BINS_PER_PROGRAM,
        )
    
    return histogram
