// Autogenerated file by gen-common-ops.py. Do not edit directly!
#pragma once

#include <ATen/Tensor.h>

namespace torch_ipex {

class AtenIpexTypeExt {
 public:
  static void packed_add_(at::Tensor & top_half, at::Tensor & bot_half, const at::Tensor & grad, float alpha);
  static at::Tensor interaction_forward(const std::vector<at::Tensor> & input);
  static std::vector<at::Tensor> interaction_backward(const at::Tensor & grad_out, const std::vector<at::Tensor> & input);
  static std::vector<at::Tensor> embedding_bag(const at::Tensor & weight, const at::Tensor & indices, const at::Tensor & offsets, bool scale_grad_by_freq, int64_t mode, bool sparse, const c10::optional<at::Tensor>& per_sample_weights, bool include_last_offset);
  static at::Tensor linear(const at::Tensor& input, const at::Tensor& weight, const c10::optional<at::Tensor>& bias);
  static at::Tensor adaptive_avg_pool2d(at::Tensor const& input, at::IntArrayRef output_size);
  static at::Tensor max_pool2d(const at::Tensor& input, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode);
  static at::Tensor max_pool3d(const at::Tensor& input, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode);
  static std::vector<at::Tensor> lstm(const at::Tensor& input, std::vector<at::Tensor> hidden, std::vector<at::Tensor> params, bool has_biases, int64_t num_layers, double dropout_p, bool train, bool bidirectional, bool batch_first);
};

}  // namespace torch_ipex

