#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include "folding3d.hpp"

// for indexing
using namespace torch::indexing;
namespace py = pybind11;

// Assuming the function definitions stay the same...

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

    // Ensure that PyTorch is loaded when the module is imported
    py::module::import("torch");

    m.def("ovadd_3d", &ovadd_3d,
          R"doc(
            Overlapp-Add/AVG of 3D tensor patches.

            This function performs an overlapping add operation on 3D tensor patches.
            The input tensor is divided into patches of size kernel_size with the given stride.
            These patches are then added (or averaged) back into a tensor of size roi.

            Args:
              tensor: A windowed 3D voxel tensor of shape (N, C, K, D, D, D) or (N, C, K, D, D, D, _, ...).
              roi: The size of the region of interest.
              kernel_size: The size of the kernel (patch).
              stride: The stride of the kernel.
              average: Whether to average the overlapped areas (default: true).

            Returns:
              A tensor of size (N, C, roi, roi, roi) with the overlapped areas added (or averaged).
          )doc");

    m.def("divisor_3d", &divisor_3d,
          R"doc(
            Overlapp-Add Grid of 3D tensor patches.

            This function creates a divisor grid for a 3D voxel tensor of size roi, given a kernel size and stride.
            The grid is used to normalize overlapping additions.

            Args:
              roi: The size of the region of interest.
              kernel_size: The size of the kernel (patch).
              stride: The stride of the kernel.

            Returns:
              A tensor of size (roi, roi, roi) representing the divisor grid.
          )doc");

    m.def("window_3d", &window_3d,
          R"doc(
            Windowing of 3D tensor patches.

            This function extracts patches from a 3D tensor using a sliding window approach.
            The patches are extracted with the given kernel size and stride.

            Args:
              tensor: A 3D voxel tensor of shape (N, C, D, D, D) or (N, C, D, D, D, _, ...).
              kernel_size: The size of the kernel (block).
              stride: The stride of the kernel.

            Returns:
              A tensor of shape (N, C, num_patches, kernel_size, kernel_size, kernel_size) containing the extracted patches.
          )doc");

    m.def("ovadd_3d_noblock", &ovadd_3d_noblock,
          R"doc(
            Overlapp-Add/AVG of 3D tensor patches (No-Block Version).

            This function performs an overlapping add operation on 3D tensor patches.
            The input tensor is divided into patches of size kernel_size with the given stride.
            These patches are then added (or averaged) back into a tensor of size roi.
            This version does not assume that the roi is a cubic region.

            Args:
              tensor: A 5D tensor of shape (N, C, D, H, W) or (N, C, D, H, W, 1).
              roi: The size of the region of interest in each dimension.
              kernel_size: The size of the kernel (patch).
              stride: The stride of the kernel.
              average: Whether to average the overlapped areas (default: true).

            Returns:
              A tensor of size (N, C, roi[0], roi[1], roi[2]) with the overlapped areas added (or averaged).
          )doc");

    m.def("divisor_3d_noblock", &divisor_3d_noblock,
          R"doc(
            Overlapp-Add Grid of 3D tensor patches (No-Block Version).

            This function creates a divisor grid for a 3D tensor of size roi, given a kernel size and stride.
            The grid is used to normalize overlapping additions. This version does not assume that the roi is a cubic region.

            Args:
              roi: The size of the region of interest in each dimension.
              kernel_size: The size of the kernel (patch).
              stride: The stride of the kernel.

            Returns:
              A tensor of size (roi[0], roi[1], roi[2]) representing the divisor grid.
          )doc");

    m.def("window_3d_noblock", &window_3d_noblock,
          R"doc(
            Windowing of 3D tensor patches (No-Block Version).

            This function extracts patches from a 3D tensor using a sliding window approach.
            The patches are extracted with the given kernel size and stride. This version does not assume that the roi is a cubic region.

            Args:
              tensor: A 5D tensor of shape (N, C, D, H, W) or (N, C, D, H, W, 1).
              kernel_size: The size of the kernel (patch).
              stride: The stride of the kernel.

            Returns:
              A tensor of shape (N, C, num_patches, kernel_size, kernel_size, kernel_size) containing the extracted patches.
          )doc");
}
