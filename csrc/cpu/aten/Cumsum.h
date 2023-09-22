#include <ATen/Tensor.h>
#include <ATen/core/TensorBody.h>
#include <dyndisp/DispatchStub.h>
#include <torch/all.h>
#include <cstdint>

namespace torch_ipex {
namespace cpu {

namespace {

// naive gemm, not jblas version
at::Tensor& cumsum_kernel_impl(
    at::Tensor& activation,
    at::Tensor& weight,
    at::Tensor& output,
    int64_t m,
    int64_t n,
    int64_t k);
} // namespace

using cumsum_kernel_fn = at::Tensor& (*)(at::Tensor&,
                                         at::Tensor&,
                                         at::Tensor&,
                                         int64_t,
                                         int64_t,
                                         int64_t);
DECLARE_DISPATCH(cumsum_kernel_fn, cumsum_kernel_stub);

} // namespace cpu
} // namespace torch_ipex
