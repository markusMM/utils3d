#include "folding3d.hpp"

// for indexing
using namespace torch::indexing;

torch::Tensor ovadd_3d(
    torch::Tensor tensor,
    int64_t roi,
    int64_t kernel_size,
    int64_t stride,
    bool average)
{
    if (tensor.dim() == 5)
    {
        tensor = tensor.unsqueeze(0);
    }
    tensor = tensor.to(torch::kFloat32);

    auto n = tensor.size(0);
    auto c = tensor.size(1);
    auto s = stride;
    auto z = kernel_size;
    auto k = int((roi - z) / s + 1);

    // final overlapped tensor
    torch::Tensor ova = torch::zeros(
        {n, c, roi, roi, roi},
        torch::TensorOptions(torch::kFloat));
    // overlapp add insertion loop
    for (int64_t i = 0; i < k; i++)
    {
        for (int64_t j = 0; j < k; j++)
        {
            for (int64_t l = 0; l < k; l++)
            {
                int64_t w_id = i * k * k + j * k + l;
                // int64_t w_id = int64_t((i * k + j * pow(k, 2) + l * pow(k, 3)) / k);
                torch::Tensor a = ova.index({Slice(None, None),
                                             Slice(None, None),
                                             Slice((s * i), (s * i + z)),
                                             Slice((s * j), (s * j + z)),
                                             Slice((s * l), (s * l + z))});
                ova.index_put_(
                    {Slice(None, None),
                     Slice(None, None),
                     Slice((s * i), (s * i + z)),
                     Slice((s * j), (s * j + z)),
                     Slice((s * l), (s * l + z))},
                    tensor.index({Slice(None, None),
                                  Slice(None, None),
                                  w_id,
                                  Slice(None, None),
                                  Slice(None, None),
                                  Slice(None, None)}) +
                        a);
            }
        }
    }
    // averaging final tensor
    if (average == true)
    {
        torch::Tensor one = torch::zeros(
            {roi, roi, roi},
            torch::TensorOptions(torch::kFloat));
        // partial constructor for averaging tensor
        torch::Tensor o_p = torch::zeros(
            {roi},
            torch::TensorOptions(torch::kFloat));
        for (int l = 0; l < k; l++)
        {
            torch::Tensor a = torch::ones(
                {z},
                torch::TensorOptions(torch::kFloat));
            o_p.index_put_(
                {Slice(l * s, l * s + z)},
                o_p.index({Slice(s * l, s * l + z)}) + a);
        }
        one = o_p.index({None, Slice(None, None), None});
        one = one * o_p.index({Slice(None, None), None, None});
        one = one * o_p.index({None, None, Slice(None, None)});
        ova = ova / one;
    }
    return ova;
}

torch::Tensor divisor_3d(
    int64_t roi,
    int64_t kernel_size,
    int64_t stride)
{
    auto s = stride;
    auto z = kernel_size;
    auto k = int((roi - z) / s + 1);

    torch::Tensor one = torch::zeros(
        {roi, roi, roi},
        torch::TensorOptions(torch::kFloat));
    // partial constructor for averaging tensor
    torch::Tensor o_p = torch::zeros(
        {roi},
        torch::TensorOptions(torch::kFloat));
    for (int l = 0; l < k; l++)
    {
        torch::Tensor a = torch::ones(
            {z},
            torch::TensorOptions(torch::kFloat));
        o_p.index_put_(
            {Slice(l * s, l * s + z)},
            o_p.index({Slice(s * l, s * l + z)}) + a);
    }
    one = o_p.index({None, Slice(None, None), None});
    one = one * o_p.index({Slice(None, None), None, None});
    one = one * o_p.index({None, None, Slice(None, None)});
    return one;
}

torch::Tensor window_3d(
    torch::Tensor tensor,
    int64_t kernel_size,
    int64_t stride)
{
    if (tensor.dim() == 4)
    {
        tensor = tensor.unsqueeze(0);
    }
    tensor = tensor.to(torch::kFloat32);

    auto n = tensor.size(0);
    auto c = tensor.size(1);
    auto d = tensor.size(2);
    auto s = stride;
    auto z = kernel_size;

    // Calculate the number of blocks per dimension
    auto k = (d - z + s) / s;

    torch::Tensor ova = torch::zeros(
        {n, c, k * k * k, kernel_size, kernel_size, kernel_size},
        torch::TensorOptions(torch::kFloat));

    int64_t block_id = 0;
    for (int64_t i = 0; i < k; i++)
    {
        for (int64_t j = 0; j < k; j++)
        {
            for (int64_t l = 0; l < k; l++)
            {
                // Handle out-of-bounds slicing
                int64_t start_i = s * i;
                int64_t end_i = std::min(s * i + z, d);
                int64_t start_j = s * j;
                int64_t end_j = std::min(s * j + z, d);
                int64_t start_l = s * l;
                int64_t end_l = std::min(s * l + z, d);

                ova.index_put_(
                    {Slice(None, None),
                     Slice(None, None),
                     block_id,
                     Slice(None, end_i - start_i),
                     Slice(None, end_j - start_j),
                     Slice(None, end_l - start_l)},
                    tensor.index({Slice(None, None),
                                  Slice(None, None),
                                  Slice(start_i, end_i),
                                  Slice(start_j, end_j),
                                  Slice(start_l, end_l)}));

                block_id++;
            }
        }
    }

    return ova;
}

// ----- No-Block Version ----------------------------------------------------- //

torch::Tensor ovadd_3d_noblock(
    torch::Tensor tensor,
    int64_t roi[3],
    int64_t kernel_size,
    int64_t stride,
    bool average)
{
    if (tensor.dim() == 5)
    {
        tensor = tensor.unsqueeze(0);
    }
    tensor = tensor.to(torch::kFloat32);

    auto n = tensor.size(0);
    auto c = tensor.size(1);
    auto s = stride;
    auto z = kernel_size;
    auto kx = int((roi[0] - z) / s + 1);
    auto ky = int((roi[1] - z) / s + 1);
    auto kz = int((roi[2] - z) / s + 1);

    // final overlapped tensor
    torch::Tensor ova = torch::zeros(
        {n, c, roi[0], roi[1], roi[2]},
        torch::TensorOptions(torch::kFloat));
    // overlapp add insertion loop
    for (int64_t i = 0; i < kx; i++)
    {
        for (int64_t j = 0; j < ky; j++)
        {
            for (int64_t l = 0; l < kz; l++)
            {
                int64_t w_id = int64_t((i * kz + j * pow(kz, 2) + l * pow(kz, 3)) / kz);
                torch::Tensor a = ova.index({Slice(None, None),
                                             Slice(None, None),
                                             Slice((s * i), (s * i + z)),
                                             Slice((s * j), (s * j + z)),
                                             Slice((s * l), (s * l + z))});
                ova.index_put_(
                    {Slice(None, None),
                     Slice(None, None),
                     Slice((s * i), (s * i + z)),
                     Slice((s * j), (s * j + z)),
                     Slice((s * l), (s * l + z))},
                    tensor.index({Slice(None, None),
                                  Slice(None, None),
                                  w_id,
                                  Slice(None, None),
                                  Slice(None, None),
                                  Slice(None, None)}) +
                        a);
            }
        }
    }
    // averaging final tensor
    if (average == true)
    {
        torch::Tensor one = torch::zeros(
            {roi[0], roi[1], roi[2]},
            torch::TensorOptions(torch::kFloat));
        // partial constructor for averaging tensor
        torch::Tensor o_p = torch::zeros(
            {roi[0]},
            torch::TensorOptions(torch::kFloat));
        for (int l = 0; l < kx; l++)
        {
            torch::Tensor a = torch::ones(
                {z},
                torch::TensorOptions(torch::kFloat));
            o_p.index_put_(
                {Slice(l * s, l * s + z)},
                o_p.index({Slice(s * l, s * l + z)}) + a);
        }
        one = o_p.index({None, Slice(None, None), None});
        one = one * o_p.index({Slice(None, None), None, None});
        one = one * o_p.index({None, None, Slice(None, None)});
        ova = ova / one;
    }
    return ova;
}

torch::Tensor divisor_3d_noblock(
    int64_t roi[3],
    int64_t kernel_size,
    int64_t stride)
{
    auto s = stride;
    auto z = kernel_size;
    auto kx = int((roi[0] - z) / s + 1);
    auto ky = int((roi[1] - z) / s + 1);
    auto kz = int((roi[2] - z) / s + 1);

    torch::Tensor one = torch::zeros(
        {roi[0], roi[1], roi[2]},
        torch::TensorOptions(torch::kFloat));
    // partial constructor for averaging tensor
    torch::Tensor o_p = torch::zeros(
        {roi[0]},
        torch::TensorOptions(torch::kFloat));
    for (int l = 0; l < kx; l++)
    {
        torch::Tensor a = torch::ones(
            {z},
            torch::TensorOptions(torch::kFloat));
        o_p.index_put_(
            {Slice(l * s, l * s + z)},
            o_p.index({Slice(s * l, s * l + z)}) + a);
    }
    one = o_p.index({None, Slice(None, None), None});
    one = one * o_p.index({Slice(None, None), None, None});
    one = one * o_p.index({None, None, Slice(None, None)});
    return one;
}

torch::Tensor window_3d_noblock(
    torch::Tensor tensor,
    int64_t kernel_size,
    int64_t stride)
{
    if (tensor.dim() == 4)
    {
        tensor = tensor.unsqueeze(0);
    }
    tensor = tensor.to(torch::kFloat32);

    auto n = tensor.size(0);
    auto c = tensor.size(1);
    auto d = tensor.size(2);
    auto h = tensor.size(3);
    auto w = tensor.size(4);
    auto s = stride;
    auto z = kernel_size;
    auto kx = int((d - z) / s) + 1;
    auto ky = int((h - z) / s) + 1;
    auto kz = int((w - z) / s) + 1;

    torch::Tensor ova = torch::zeros(
        {n, c, int(pow(kx, 3)), kernel_size, kernel_size, kernel_size},
        torch::TensorOptions(torch::kFloat));
    for (int64_t i = 0; i < kx; i++)
    {
        for (int64_t j = 0; j < ky; j++)
        {
            for (int64_t l = 0; l < kz; l++)
            {
                int64_t w_id = long((i * kz + j * pow(kz, 2) + l * pow(kz, 3)) / kz);
                ova.index_put_(
                    {Slice(None, None),
                     Slice(None, None),
                     w_id,
                     Slice(None, None),
                     Slice(None, None),
                     Slice(None, None)},
                    tensor.index({Slice(None, None),
                                  Slice(None, None),
                                  Slice((s * i), (s * i + z)),
                                  Slice((s * j), (s * j + z)),
                                  Slice((s * l), (s * l + z))}));
            }
        }
    }
    return ova;
}
