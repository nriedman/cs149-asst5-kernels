#include <cuda_runtime.h>

//
// craete your function: __global__ void kernel(...) here
// Note: input data is of type uint8_t
//

// Launch with N threads
__global__ void find_starts(const uint8_t* sorted_bin_ids, int* bin_starts, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    if (tid == 0 || (sorted_bin_ids[tid] != sorted_bin_ids[tid-1])) {
        bin_starts[sorted_bin_ids[tid]] = tid;
    }
}

// Launch with num_bins threads
__global__ void bin_sizes(const int* bin_starts, int* histogram_bins, int num_items, int num_bins) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_bins) return;

    if (bin_starts[tid] == 0) {
        histogram_bins[tid] = 0;
    } else {
        int next_id = tid + 1;
        while (next_id < num_bins && bin_starts[next_id] == -1) {
            next_id++;
        }
        
        if (next_id < num_bins) {
            histogram_bins[tid] = bin_starts[next_id] - bin_starts[tid];
        } else {
            histogram_bins[tid] = num_bins - bin_starts[tid];
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

    // Reshape input tensor so channel slices are contiguous and sort into ascending order
    data = data.transpose(0, 1).contiguous();  // shape: (C, L)

    torch::Tensor sorted_ids_tnsr = std::get<0>(torch::sort(data, dim=1));
    
    // Allocate output tensor
    auto options = torch::TensorOptions()
        .dtype(torch::kInt32)
        .device(data.device());
    torch::Tensor histogram = torch::zeros({num_channels, num_bins}, options);
    
    int BLOCK_SIZE = 1024;

    int num_blocks_items = (length + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int num_blocks_bins = (num_bins + BLOCK_SIZE - 1) / BLOCK_SIZE;

    ////
    // Launch your kernel here
    for (int c = 0; c < num_channels; c++) {
        // find starting point of each bin
        torch::Tensor bin_starts_tensor = torch::full({num_bins, }, -1, options);
        int* bin_starts = bin_starts_tensor.data_ptr<int>();

        torch::Tensor sorted_ids_slice = sorted_ids_tnsr.slice(0, c, c + 1).squeeze(0);
        uint8_t* sorted_ids = sorted_ids_slice.data_ptr<uint8_t>();

        find_starts<<<num_blocks_items, BLOCK_SIZE>>>(sorted_ids, bin_starts, length);

        torch::Tensor histogram_slice = histogram.slice(0, c, c + 1).squeeze(0);
        int* out_ptr = histogram_slice.data_ptr<int>();

        bin_sizes<<<num_blocks_bins, BLOCK_SIZE>>>(bin_starts, out_ptr, length, num_bins);
    }
    ////


    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    
    return histogram;
}