#include <ATen/NamedTensorUtils.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/core/TensorBody.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/TensorIterator.h>

#include <aten/Cumsum.h>

#include <immintrin.h>
#include "jit_blas_gemm.h"
#include "vec/vec.h"

namespace torch_ipex {
namespace cpu {

namespace {

using namespace at::vec;
using namespace torch_ipex::cpu::kernel;

inline int64_t divup(int64_t x, int64_t y) {
  return (x + y - 1) / y;
}

static inline void cumsum_lastdim_kernel(
    at::Tensor& activation,
    at::Tensor& weight,
    at::Tensor& output,
    int64_t m,
    int64_t n,
    int64_t k) {
  jblas::gemm::GemmCore_Row_NN_2x48_AVX2 gemm;
  auto a = activation.data_ptr<float>();
  auto b = weight.data_ptr<float>();
  auto c = output.data_ptr<float>();
  gemm.forward(a, b, c, m, n, k, 4 * k, 4 * n, 4 * n, 0);
}

bool cumsum_fast_path(
    const at::Tensor& self,
    const at::Tensor& result,
    int64_t dim,
    c10::optional<at::ScalarType> dtype) {
  // check contiguous
  bool is_contig = self.is_contiguous() && (result.is_contiguous());
  if (!is_contig)
    return false;
  // check dim
  auto wrap_dim = at::maybe_wrap_dim(dim, self.dim());
  if (wrap_dim != self.dim() - 1)
    return false;
  // check dtype matched
  auto out_dtype = result.scalar_type();
  if (dtype.has_value() && out_dtype != dtype.value())
    return false;
  // check dtype enabled
  bool is_dtype_enabled = out_dtype == at::ScalarType::Double ||
      out_dtype == at::ScalarType::Float || out_dtype == at::ScalarType::Long;
  if (!is_dtype_enabled)
    return false;
  return true;
}

class NewCumSumOp : public torch::autograd::Function<NewCumSumOp> {
 public:
  static at::Tensor& _forward(
      at::Tensor& activation,
      at::Tensor& weight,
      at::Tensor& output,
      int64_t m,
      int64_t n,
      int64_t k) {
    // RECORD_FUNCTION("IPEXCumSumOp::_forward",
    // c10::ArrayRef<c10::IValue>({}));

    // if (result.sizes() != self.sizes()) {
    //   at::native::resize_output(result, self.sizes());
    // }
    // if (cumsum_fast_path(result, self, dim, dtype)) {
    //   AT_DISPATCH_FLOATING_TYPES_AND(
    //       at::ScalarType::Long, self.scalar_type(), "cumsum_lastdim_cpu", [&]
    //       {
    std::cout << "enter fwd" << std::endl;
    cumsum_lastdim_kernel(activation, weight, output, m, n, k);
    return output;
    //       });
    //   return result;
    // }
    // return at::cumsum_out(result, self, dim, dtype);
  }

  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      at::Tensor& result,
      const at::Tensor& self,
      int64_t dim,
      c10::optional<at::ScalarType> dtype) {
    RECORD_FUNCTION("IPEXCumSumOp::forward", c10::ArrayRef<c10::IValue>({}));

    at::AutoDispatchBelowADInplaceOrView g;
    ctx->saved_data["dim"] = dim;
    // auto ret = _forward(result, self, dim, dtype);
    return result;
  }

  static torch::autograd::tensor_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::tensor_list grad_outputs) {
    RECORD_FUNCTION("IPEXCumSumOp::backward", c10::ArrayRef<c10::IValue>({}));

    at::AutoDispatchBelowADInplaceOrView g;
    int64_t dim = ctx->saved_data["dim"].toInt();

    at::Tensor grad_out = grad_outputs[0];
    at::Tensor grad_self;
    if (grad_out.numel() <= 1 || grad_out.size(dim) == 1) {
      grad_self = grad_out;
    }
    grad_self = grad_out.flip(dim).cumsum(dim).flip(dim);
    return {at::Tensor(), grad_self, at::Tensor(), at::Tensor()};
  }
};

at::Tensor& cumsum_kernel_impl(
    at::Tensor& activation,
    at::Tensor& weight,
    at::Tensor& output,
    int64_t m,
    int64_t n,
    int64_t k) {
  // if (at::GradMode::is_enabled() && self.requires_grad())
  //   return NewCumSumOp::apply(result, self, dim, dtype);
  std::cout << "enter cumsum kernel impl" << std::endl;
  return NewCumSumOp::_forward(activation, weight, output, m, n, k);
}

} // anonymous namespace

REGISTER_DISPATCH(cumsum_kernel_stub, &cumsum_kernel_impl);

} // namespace cpu
} // namespace torch_ipex
