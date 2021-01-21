#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/mish_op.h"

namespace caffe2 {
namespace {
template <typename T>
__global__ void MishKernel(const int N, const T* X, T* Y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    Y[i] = tanhf(log(exp(X[i]) + 1.0f)) * X[i];
  }
}
} // namespace

template <>
template <typename T>
bool MishFunctor<CUDAContext>::
operator()(const int N, const T* X, T* Y, CUDAContext* context) const {
  MishKernel<float>
      <<<CAFFE_GET_BLOCKS(N),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context->cuda_stream()>>>(N, X, Y);


  return true;
}

REGISTER_CUDA_OPERATOR(
    Mish,
    UnaryElementwiseOp<
        TensorTypes<float>,
        CUDAContext,
        MishFunctor<CUDAContext>>);
} // namespace caffe2
