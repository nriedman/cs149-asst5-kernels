import torch
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline
import sys
import io

# CUDA source code loaded from submission.cu
cuda_source = """

#include <cuda_runtime.h>
// Kernel using shared memory privatization to reduce atomic contention
__global__ void histogram_kernel_shared(
    const uint8_t* __restrict__ data,
    int* __restrict__ histogram,
    int length,
    int num_channels,
    int num_bins
) {
    // Each block handles one channel
    int channel = blockIdx.x;
    if (channel >= num_channels) return;
    
    // Shared memory for privatized histogram (per block)
    extern __shared__ int shared_hist[];
    
    // Initialize shared memory histogram to zero
    for (int bin = threadIdx.x; bin < num_bins; bin += blockDim.x) {
        shared_hist[bin] = 0;
    }
    __syncthreads();
    
    // Each thread processes elements from this channel
    int tid = threadIdx.x;
    int stride = blockDim.x;
    
    // Process data elements for this channel
    for (int i = tid; i < length; i += stride) {
        uint8_t value = data[i * num_channels + channel];
        atomicAdd(&shared_hist[value], 1);
    }
    __syncthreads();
    
    // Write shared histogram to global memory
    int* global_hist = histogram + channel * num_bins;
    for (int bin = threadIdx.x; bin < num_bins; bin += blockDim.x) {
        atomicAdd(&global_hist[bin], shared_hist[bin]);
    }
}

// Host function to launch kernel
torch::Tensor histogram_kernel(
    torch::Tensor data,  // [length, num_channels], dtype=uint8
    int num_bins
) {
    TORCH_CHECK(data.device().is_cuda(), "Tensor data must be a CUDA tensor");
    const int length = data.size(0);
    const int num_channels = data.size(1);
    
    // Allocate output tensor
    auto options = torch::TensorOptions()
        .dtype(torch::kInt32)
        .device(data.device());
    torch::Tensor histogram = torch::zeros({num_channels, num_bins}, options);
    
    // Get pointers
    const uint8_t* data_ptr = data.data_ptr<uint8_t>();
    int* hist_ptr = histogram.data_ptr<int>();
    
    // Launch configuration
    int threads_per_block = 256;
    int num_blocks = num_channels;
    int shared_mem_size = num_bins * sizeof(int);
    
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

cuda_module = load_inline(
    name='submission_cuda_histogram_test_sunet',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['histogram_kernel'],
    verbose=True,  # Enable verbose to see compilation details
    # with_cuda=True,
    # build_directory=".",
)

def custom_kernel(data: input_t) -> output_t:
    """
    Wrapper function matching the required signature.
    
    Args:
        data: Tuple of (array, num_bins) where:
            array:    Tensor of shape [length, num_channels] with integer values in [0, num_bins-1]
            num_bins: Number of bins for the histogram
    
    Returns:
        histogram: Tensor of shape [num_channels, num_bins] containing histogram counts for each channel
    """

    array, num_bins = data
    
    if not array.is_cuda:
        array = array.cuda()
    
    # Call CUDA kernel
    histogram = cuda_module.histogram_kernel(array, num_bins)

    return histogram
