#include <torch/extension.h>
#include "seg_loss_kernel.h"
#include <stdlib.h>
#include <c10/cuda/CUDAGuard.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a GPU tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

namespace py = pybind11;

class SegLossTorch{
private:
  SegLoss segLoss;
public:
  SegLossTorch(const int chunk_size, const bool verbose = true): segLoss(chunk_size, verbose){}
  torch::Tensor cuda_forward(torch::Tensor W, torch::Tensor loss,
                             torch::Tensor label_sizes, torch::Tensor prob_sizes, torch::Tensor label_batch,
                             torch::Tensor alpha_1d, torch::Tensor alpha_2d,
                             torch::Tensor beta_1d, torch::Tensor beta_2d,
                             const bool is_sum);
  torch::Tensor cuda_backward(torch::Tensor W, torch::Tensor grad_output,
                              torch::Tensor label_sizes, torch::Tensor prob_sizes, torch::Tensor label_batch,
                              torch::Tensor alpha_1d, torch::Tensor alpha_2d,
                              torch::Tensor beta_1d, torch::Tensor beta_2d,
                              const bool is_sum);
};


torch::Tensor SegLossTorch::cuda_forward(torch::Tensor W, torch::Tensor loss,
                                         torch::Tensor label_sizes, torch::Tensor prob_sizes, torch::Tensor label_batch,
                                         torch::Tensor alpha_1d, torch::Tensor alpha_2d,
                                         torch::Tensor beta_1d, torch::Tensor beta_2d,
                                         const bool is_sum){
  CHECK_INPUT(W);
  CHECK_INPUT(loss);
  CHECK_INPUT(label_sizes);
  CHECK_INPUT(prob_sizes);
  CHECK_INPUT(label_batch);
  CHECK_INPUT(alpha_1d);
  CHECK_INPUT(alpha_2d);
  CHECK_INPUT(beta_1d);
  CHECK_INPUT(beta_2d);
  int B = W.size(0);
  int T = W.size(1);
  int S = W.size(2);
  int V = W.size(3);
  int L = label_batch.size(1);
  torch::Tensor loss_ = torch::zeros_like(loss);
  segLoss.seg_forward(W.data<float>(), loss_.data<float>(),
                      label_sizes.data<int>(), prob_sizes.data<int>(), label_batch.data<int>(),
                      alpha_1d.data<float>(), alpha_2d.data<float>(),
                      beta_1d.data<float>(), beta_2d.data<float>(),
                      B, T, S, V, L, is_sum);
  return loss_;
}


torch::Tensor SegLossTorch::cuda_backward(torch::Tensor W, torch::Tensor grad_output,
                                          torch::Tensor label_sizes, torch::Tensor prob_sizes, torch::Tensor label_batch,
                                          torch::Tensor alpha_1d, torch::Tensor alpha_2d,
                                          torch::Tensor beta_1d, torch::Tensor beta_2d,
                                          const bool is_sum){
  CHECK_INPUT(W);
  CHECK_INPUT(grad_output);
  CHECK_INPUT(label_sizes);
  CHECK_INPUT(prob_sizes);
  CHECK_INPUT(label_batch);
  CHECK_INPUT(alpha_1d);
  CHECK_INPUT(alpha_2d);
  CHECK_INPUT(beta_1d);
  CHECK_INPUT(beta_2d);
  torch::Tensor grad_W = torch::zeros_like(W);
  float grad_output_val = grad_output.cpu().data<float>()[0];
  int B = W.size(0);
  int T = W.size(1);
  int S = W.size(2);
  int V = W.size(3);
  int L = label_batch.size(1);
  
  segLoss.seg_backward(W.data<float>(), grad_W.data<float>(), grad_output_val,
                       label_sizes.data<int>(), prob_sizes.data<int>(), label_batch.data<int>(),
                       alpha_1d.data<float>(), alpha_2d.data<float>(),
                       beta_1d.data<float>(), beta_2d.data<float>(),
                       B, T, S, V, L, is_sum);
  return grad_W;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
  m.doc() = "Seg Loss Torch Module";
  py::class_<SegLossTorch>(m, "SegLossTorch", py::module_local())
    .def(py::init<const int, const bool>())
    .def("cuda_forward", &SegLossTorch::cuda_forward, "Seg Forward")
    .def("cuda_backward", &SegLossTorch::cuda_backward, "Seg Backward");
}
