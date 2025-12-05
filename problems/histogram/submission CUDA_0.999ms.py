import torch
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline
import sys
import io

# CUDA source code loaded from submission.cu
cuda_source = """
#include <cuda_runtime.h>

// Hybrid approach: blocks handle portions of data and multiple channels
__global__ __launch_bounds__(1024, 1) void histogram_kernel_hybrid(
    const uint8_t* __restrict__ data,
    int* __restrict__ histogram,
    int length,
    int num_channels,
    int num_bins,
    int channels_per_block
) {
    // Each block handles a group of channels and all data
    int channel_group = blockIdx.x;
    int start_channel = channel_group * channels_per_block;
    int end_channel = min(start_channel + channels_per_block, num_channels);
    
    extern __shared__ int shared_hist[];
    
    // Initialize shared memory using vectorized stores
    int total_bins = (end_channel - start_channel) * num_bins;
    int4 zero_vec = make_int4(0, 0, 0, 0);
    for (int i = threadIdx.x * 4; i < total_bins; i += blockDim.x * 4) {
        if (i + 3 < total_bins) {
            *(int4*)&shared_hist[i] = zero_vec;
        } else {
            for (int j = 0; j < 4 && i + j < total_bins; j++) {
                shared_hist[i + j] = 0;
            }
        }
    }
    __syncthreads();
    
    // Process all data for channels in this group using vectorized loads
    int tid = threadIdx.x;
    int stride = blockDim.x;
    
    for (int i = tid; i < length; i += stride) {
        // Process all channels in this block's group with optimized vectorization
        int c = start_channel;
        for (; c + 7 < end_channel; c += 8) {
            // Load 8 consecutive channels at once using uint2 (8 bytes)
            uint2 values = *(uint2*)&data[i * num_channels + c];
            
            // Process 8 channels
            uint8_t* vals = (uint8_t*)&values;
            for (int j = 0; j < 8; j++) {
                int local_idx = (c + j - start_channel) * num_bins + vals[j];
                atomicAdd(&shared_hist[local_idx], 1);
            }
        }
        
        // Handle remaining channels in groups of 4
        for (; c + 3 < end_channel; c += 4) {
            // Load 4 consecutive channels at once
            uint32_t values = *(uint32_t*)&data[i * num_channels + c];
            
            // Process 4 channels
            uint8_t val0 = ((uint8_t*)&values)[0];
            uint8_t val1 = ((uint8_t*)&values)[1];
            uint8_t val2 = ((uint8_t*)&values)[2];
            uint8_t val3 = ((uint8_t*)&values)[3];
            
            int local_idx0 = (c - start_channel) * num_bins + val0;
            int local_idx1 = (c + 1 - start_channel) * num_bins + val1;
            int local_idx2 = (c + 2 - start_channel) * num_bins + val2;
            int local_idx3 = (c + 3 - start_channel) * num_bins + val3;
            
            atomicAdd(&shared_hist[local_idx0], 1);
            atomicAdd(&shared_hist[local_idx1], 1);
            atomicAdd(&shared_hist[local_idx2], 1);
            atomicAdd(&shared_hist[local_idx3], 1);
        }
        
        // Handle remaining channels
        for (; c < end_channel; c++) {
            uint8_t value = data[i * num_channels + c];
            int local_idx = (c - start_channel) * num_bins + value;
            atomicAdd(&shared_hist[local_idx], 1);
        }
    }
    __syncthreads();
    
    // Write results to global memory
    for (int c = start_channel; c < end_channel; c++) {
        int* global_hist = histogram + c * num_bins;
        int local_offset = (c - start_channel) * num_bins;
        for (int bin = threadIdx.x; bin < num_bins; bin += blockDim.x) {
            global_hist[bin] = shared_hist[local_offset + bin];
        }
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
    
    // Hybrid approach: each block handles multiple channels
    int channels_per_block = 4;
    int threads_per_block = 1024;
    int num_blocks = (num_channels + channels_per_block - 1) / channels_per_block;
    int shared_mem_size = channels_per_block * num_bins * sizeof(int);
    
    // Launch kernel
    histogram_kernel_hybrid<<<num_blocks, threads_per_block, shared_mem_size>>>(
        data_ptr,
        hist_ptr,
        length,
        num_channels,
        num_bins,
        channels_per_block
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
