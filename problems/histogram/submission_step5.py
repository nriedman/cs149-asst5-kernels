import torch
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline
import sys
import io

# CUDA source code - Iteration 1: Privatization with Shared Memory
cuda_source = """
#include <cuda_runtime.h>
#include <torch/extension.h>

// Kernel: Privatization with Shared Memory
// Each thread block maintains a private histogram in shared memory
// to reduce global atomic contention
__global__ void histogram_kernel_shared(
    const uint8_t* __restrict__ data,  // [length, num_channels]
    int32_t* __restrict__ histogram,   // [num_channels, num_bins]
    int length,
    int num_channels,
    int num_bins
) {
    // Allocate shared memory for private histogram
    // Each block has its own copy: [num_bins] entries
    extern __shared__ int32_t shared_hist[];
    
    // Each thread block processes one channel
    int channel = blockIdx.x;
    
    if (channel >= num_channels) return;
    
    // Step 1: Initialize shared memory histogram to zero
    // Use all threads in the block to collaboratively initialize
    for (int bin = threadIdx.x; bin < num_bins; bin += blockDim.x) {
        shared_hist[bin] = 0;
    }
    __syncthreads();  // Ensure all threads finish initialization
    
    // Step 2: Each thread processes a portion of the channel data
    // and accumulates into shared memory using atomics
    for (int i = threadIdx.x; i < length; i += blockDim.x) {
        uint8_t value = data[i * num_channels + channel];
        
        // Atomic add to shared memory (much faster than global memory atomics)
        atomicAdd(&shared_hist[value], 1);
    }
    __syncthreads();  // Ensure all threads finish accumulation
    
    // Step 3: Reduce shared histogram to global memory
    // Each thread writes out a portion of the bins
    int* global_hist = histogram + channel * num_bins;
    for (int bin = threadIdx.x; bin < num_bins; bin += blockDim.x) {
        // Use atomic add to global memory (but only once per bin per block)
        atomicAdd(&global_hist[bin], shared_hist[bin]);
    }
}

// Host function to launch kernel
torch::Tensor histogram_kernel(
    torch::Tensor data,  // [length, num_channels], dtype=uint8
    int num_bins
) {
    TORCH_CHECK(data.device().is_cuda(), "Tensor data must be a CUDA tensor");
    TORCH_CHECK(data.dtype() == torch::kUInt8, "Data must be uint8");

    const int length = data.size(0);
    const int num_channels = data.size(1);
    
    // Allocate output tensor
    auto options = torch::TensorOptions()
        .dtype(torch::kInt32)
        .device(data.device());
    torch::Tensor histogram = torch::zeros({num_channels, num_bins}, options);
    
    // Get raw pointers
    const uint8_t* data_ptr = data.data_ptr<uint8_t>();
    int32_t* hist_ptr = histogram.data_ptr<int32_t>();
    
    // Launch configuration
    // One block per channel for simplicity
    int threads_per_block = 256;  // Good balance for most GPUs
    int num_blocks = num_channels;
    
    // Shared memory size: num_bins * sizeof(int32_t)
    size_t shared_mem_size = num_bins * sizeof(int32_t);
    
    // Launch kernel
    histogram_kernel_shared<<<num_blocks, threads_per_block, shared_mem_size>>>(
        data_ptr,
        hist_ptr,
        length,
        num_channels,
        num_bins
    );
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    
    return histogram;
}
"""

# C++ header declaration
cpp_source = """
#include <torch/extension.h>
torch::Tensor histogram_kernel(torch::Tensor data, int num_bins);
"""

# Ensure stdout and stderr exist
if sys.stdout is None:
    sys.stdout = io.StringIO()
if sys.stderr is None:
    sys.stderr = io.StringIO()

# Compile CUDA module
cuda_module = load_inline(
    name='histogram_cuda_shared_memory',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['histogram_kernel'],
    verbose=False,
)

def custom_kernel(data: input_t) -> output_t:
    """
    Iteration 1: Privatization with Shared Memory
    
    Goal: Reduce global atomic contention
    - Each thread block maintains its own private histogram in shared memory
    - Threads accumulate to shared memory (still atomics, but faster)
    - At the end, reduce shared histograms to global memory
    - Expected: 5-10x speedup
    - Why: Shared memory atomics are much faster than global memory atomics
    
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
    
    if not array.is_cuda:
        array = array.cuda()
    
    # Call CUDA kernel with shared memory privatization
    histogram = cuda_module.histogram_kernel(array, num_bins)

    return histogram
