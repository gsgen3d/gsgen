#ifndef _COMMON_H
#define _COMMON_H
#pragma once

// test
// #ifdef __CUDACC__
// #define CUDA_HOSTDEV __host__ __device__
// #else
// #define CUDA_HOSTDEV
// #endif

#include <assert.h>
#include <cooperative_groups.h>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_atomic_functions.h>
#include <device_launch_parameters.h>
#include <torch/torch.h>

#define GLOBAL __global__
#define DEVICE __device__
#define HOST __host__

#define N_THREADS 256

#define tid_1d(tid) const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

#define CHECK_CUDA(x)                                                          \
  TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
  TORCH_CHECK(x.is_contiguous(), #x " must be a contiguous tensor")
#define CHECK_IS_BOOL(x)                                                       \
  TORCH_CHECK(x.scalar_type() == at::ScalarType::Bool,                         \
              #x " must be an bool tensor")
#define CHECK_IS_INT(x)                                                        \
  TORCH_CHECK(x.scalar_type() == at::ScalarType::Int,                          \
              #x " must be an int tensor")
#define CHECK_IS_FLOATING(x)                                                   \
  TORCH_CHECK(x.scalar_type() == at::ScalarType::Float,                        \
              #x " must be a floating tensor")
#define CHECK_IS_DOUBLE(x)                                                     \
  TORCH_CHECK(x.scalar_type() == at::ScalarType::Double,                       \
              #x " must be a floating tensor")

#define CHECK_DC_FLOAT(x)                                                      \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x);                                                         \
  CHECK_IS_FLOATING(x)

#define CHECK_DC_INT(x)                                                        \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x);                                                         \
  CHECK_IS_INT(x)

#define checkLastCudaError(error)                                              \
  error = cudaGetLastError();                                                  \
  if (error != cudaSuccess) {                                                  \
    printf("At line %s file %s:\nCUDA error: %s\n", __LINE__, __FILE__,        \
           cudaGetErrorString(error));                                         \
    exit(-1);                                                                  \
  }

#define cudaCheck(call)                                                        \
  {                                                                            \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess) {                                                \
      printf("ERROR: %s:%d,", __FILE__, __LINE__);                             \
      printf("code:%d, reason:%s\n", error, cudaGetErrorString(error));        \
      exit(1);                                                                 \
    }                                                                          \
  }

#define checkValue(val)                                                        \
  if (isnan(val) || isinf(val)) {                                              \
    printf("SINGULAR VAL ERROR: %s:%d,", __FILE__, __LINE__);                  \
    printf("value: %f\n", val);                                                \
  }

using torch::Tensor;
// namespace cg = cooperative_groups;

#define EPS 1e-6
#define MIN_RADIAL -85.0f
#define gs_coeff_3d 0.06349363593424097f // 1 / (2 * pi) ** (3/2)
#define gs_coeff_2d 0.15915494309189535f // 1 / (2 * pi)

#define MAX_N_FLOAT_SM 12000 // 48KB
#define MIN_RENDER_ALPHA 0.00392156862745098f // 1 / 255.0f

template <typename T>
__host__ __device__ inline T div_round_up(T val, T divisor) {
  return (val + divisor - 1) / divisor;
}

struct GpuTimer {
  cudaEvent_t start;
  cudaEvent_t stop;

  GpuTimer() {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
  }

  ~GpuTimer() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  void Start() { cudaEventRecord(start, 0); }

  void Stop() { cudaEventRecord(stop, 0); }

  void Elapsed(const char *msg) {
    const char *timing = std::getenv("TIMING");
    if (timing != NULL) {
      float elapsed;
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&elapsed, start, stop);
      printf("%s: %.3f ms\n", msg, elapsed);
    }
  }
};

#define FULL_MASK 0xffffffff

#endif
