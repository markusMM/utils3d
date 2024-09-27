#ifndef FOLDING3D_H
#define FOLDING3D_H

#include <torch/torch.h>
#include <torch/extension.h>
#include <cmath>

// Function declarations

torch::Tensor ovadd_3d(torch::Tensor tensor, int64_t roi, int64_t kernel_size, int64_t stride, bool average = true);
torch::Tensor divisor_3d(int64_t roi, int64_t kernel_size, int64_t stride);
torch::Tensor window_3d(torch::Tensor tensor, int64_t kernel_size, int64_t stride);

torch::Tensor ovadd_3d_noblock(torch::Tensor tensor, int64_t roi[3], int64_t kernel_size, int64_t stride, bool average = true);
torch::Tensor divisor_3d_noblock(int64_t roi[3], int64_t kernel_size, int64_t stride);
torch::Tensor window_3d_noblock(torch::Tensor tensor, int64_t kernel_size, int64_t stride);

#endif // FOLDING3D_H
