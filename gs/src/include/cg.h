#include "common.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

using namespace cooperative_groups;

__device__ inline void
kernel_gaussian_2d_backward_cg(float *mean, float *cov, float *query,
                               float *grad_mean, float *grad_cov, float grad) {
  // nan
  double d_grad = (double)grad;
  double c0 = (double)cov[0];
  double c1 = (double)cov[1];
  double c2 = (double)cov[2];
  double c3 = (double)cov[3];
  double det = c0 * c3 - c1 * c2;
  double x = query[0] - mean[0];
  double y = query[1] - mean[1];
  double tmpx = (x * c3 - y * c2) / det;
  double tmpy = (-x * c1 + y * c0) / det;
  // note: val has been multiplied in the grad
  // atomic ops here
  checkValue((float)(d_grad * tmpx));
  checkValue((float)(d_grad * tmpy * tmpy));

  auto g = coalesced_threads();

  atomicAdd(grad_mean, (float)(d_grad * tmpx));
  atomicAdd(grad_mean + 1, (float)(d_grad * tmpy));
  atomicAdd(grad_cov, 0.5 * (float)(d_grad * tmpx * tmpx));
  atomicAdd(grad_cov + 1, 0.5 * (float)(d_grad * tmpx * tmpy));
  atomicAdd(grad_cov + 2, 0.5 * (float)(d_grad * tmpy * tmpx));
  atomicAdd(grad_cov + 3, 0.5 * (float)(d_grad * tmpy * tmpy));
}