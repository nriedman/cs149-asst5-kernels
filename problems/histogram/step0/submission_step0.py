from task import input_t, output_t
import torch


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
    
    # allocate output histogram
    output_dtype = torch.int32
    histogram = torch.zeros(num_channels, num_bins, dtype=output_dtype, device=array.device)

    # compute histogram for each channel sequentially (very slow)
    for b in range(length):
        for c in range(num_channels):
            histogram[c][array[b][c]] += 1
    
    return histogram
