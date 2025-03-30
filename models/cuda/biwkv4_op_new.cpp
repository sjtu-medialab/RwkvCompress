/******************************************************************************
 * Copyright (c) 2025 Shanghai AI Lab.
 ******************************************************************************/

#include <torch/extension.h>
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor biwkv4_cuda_forward(
    torch::Tensor w, 
    torch::Tensor u, 
    torch::Tensor k, 
    torch::Tensor v);

std::vector<torch::Tensor> biwkv4_cuda_backward(
    torch::Tensor w, 
    torch::Tensor u, 
    torch::Tensor k, 
    torch::Tensor v, 
    torch::Tensor gy);

torch::Tensor biwkv4_forward(
    torch::Tensor w, 
    torch::Tensor u, 
    torch::Tensor k, 
    torch::Tensor v) {
    CHECK_INPUT(w);
    CHECK_INPUT(u);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    return biwkv4_cuda_forward(w, u, k, v);
}

std::vector<torch::Tensor> biwkv4_backward(
    torch::Tensor w, 
    torch::Tensor u, 
    torch::Tensor k, 
    torch::Tensor v, 
    torch::Tensor gy) {
    CHECK_INPUT(w);
    CHECK_INPUT(u);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    CHECK_INPUT(gy);
    return biwkv4_cuda_backward(w, u, k, v, gy);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &biwkv4_forward, "BiWKV4 Forward(CUDA)");
    m.def("backward", &biwkv4_backward, "BiWKV4 Backward(CUDA)");
}

TORCH_LIBRARY(biwkv4, m) {
    m.def("forward", biwkv4_forward);
    m.def("backward", biwkv4_backward);
}