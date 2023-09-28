// render a scalar along ray, differentiable on the per gaussian scalar

#pragma once
#include "common.h"
#include "data_spec.h"
#include "kernels.h"
#include <cub/cub.cuh>
#include <device_atomic_functions.h>
#include <device_launch_parameters.h>
#include <stdint.h>

#include "vol_render.h"

__device__ void vol_render_scalar_one_batch(
    uint32_t N_gaussians_this_time, float *mean, float *cov,
    float *scalar_to_render, float *alpha, float &out, float &cum_alpha,
    float *topleft, uint32_t tile_size, uint32_t n_tiles_h, uint32_t n_tiles_w,
    float pixel_size_x, float pixel_size_y, uint32_t H, uint32_t W,
    float thresh) {
  int local_id = threadIdx.x;
  int local_y = local_id / tile_size;
  int local_x = local_id % tile_size;
  int global_y = blockIdx.y * tile_size + local_y;
  int global_x = blockIdx.x * tile_size + local_x;

  if (global_y >= H || global_x >= W)
    return;

  float pos[2] = {topleft[0] + global_x * pixel_size_x,
                  topleft[1] + global_y * pixel_size_y};

  for (int i = 0; i < N_gaussians_this_time; ++i) {
    if (cum_alpha < thresh)
      break;
    float alpha_ = fminf(alpha[i], 0.99f);
    float coeff = alpha_ * cum_alpha;
    float val = kernel_gaussian_2d(mean + 2 * i, cov + 4 * i, pos);
    coeff *= val;
    if (alpha_ * val < MIN_RENDER_ALPHA) {
      continue;
    }
    out += coeff * scalar_to_render[i];
    cum_alpha *= (1 - alpha_ * val);
  }
}

__global__ void tile_based_vol_rendering_scalar(
    uint32_t N, uint32_t N_with_dub, float *__restrict__ mean,
    float *__restrict__ cov, float *__restrict__ scalar_to_render,
    float *__restrict__ alpha, int *__restrict__ start, int *__restrict__ end,
    int *__restrict__ gaussian_ids, float *__restrict__ out,
    float *__restrict__ topleft, uint32_t tile_size, uint32_t n_tiles_h,
    uint32_t n_tiles_w, float pixel_size_x, float pixel_size_y, uint32_t H,
    uint32_t W, float thresh, float *__restrict__ T) {
  int local_id = threadIdx.x;
  int local_y = local_id / tile_size;
  int local_x = local_id % tile_size;
  int tile_id = blockIdx.y * gridDim.x + blockIdx.x;
  if (start[tile_id] == -1) {
    // skip
    return;
  }
  int n_gaussians_this_tile = end[tile_id] - start[tile_id];
  if (n_gaussians_this_tile == 0) {
    return;
  }

  int n_float_per_gaussian = 2 + 4 + 1 + 1;
  int max_gaussian_sm = (MAX_N_FLOAT_SM) / n_float_per_gaussian;
  __shared__ float sm[MAX_N_FLOAT_SM];
  float *sm_mean = sm;
  float *sm_cov = sm_mean + 2 * max_gaussian_sm;
  float *sm_scalar = sm_cov + 4 * max_gaussian_sm;
  float *sm_alpha = sm_scalar + 1 * max_gaussian_sm;

  float tmp_out = 0.0f;
  float cum_alpha = 1.0f;

  gaussian_ids += start[tile_id];

  for (int n = 0; n < n_gaussians_this_tile; n += max_gaussian_sm) {
    int num_gaussian_sm = min(max_gaussian_sm, n_gaussians_this_tile - n);
    carry(num_gaussian_sm, 2, sm_mean, mean, NULL, gaussian_ids);
    carry(num_gaussian_sm, 4, sm_cov, cov, NULL, gaussian_ids);
    carry(num_gaussian_sm, 1, sm_scalar, scalar_to_render, NULL, gaussian_ids);
    carry(num_gaussian_sm, 1, sm_alpha, alpha, NULL, gaussian_ids);
    __syncthreads();
    vol_render_scalar_one_batch(num_gaussian_sm, sm_mean, sm_cov, sm_scalar,
                                sm_alpha, tmp_out, cum_alpha, topleft,
                                tile_size, n_tiles_h, n_tiles_w, pixel_size_x,
                                pixel_size_y, H, W, thresh);
    __syncthreads();
    gaussian_ids += num_gaussian_sm;
  }
  int global_y = blockIdx.y * tile_size + local_y;
  int global_x = blockIdx.x * tile_size + local_x;
  if (global_y >= H || global_x >= W) {
    return;
  }
  out[global_y * W + global_x] = tmp_out;
  T[global_y * W + global_x] = cum_alpha;
}

__device__ void vol_render_scalar_one_batch_backward(
    uint32_t N_gaussians_this_time, float *mean, float *cov, float *scalar,
    float *alpha, float *grad_mean, float *grad_cov, float *grad_scalar,
    float *grad_alpha, float grad_out, float final, float &out,
    float &cum_alpha, float *topleft, uint32_t tile_size, uint32_t n_tiles_h,
    uint32_t n_tiles_w, float pixel_size_x, float pixel_size_y, uint32_t H,
    uint32_t W, float thresh) {
  int local_id = threadIdx.x;
  // check row major here
  int local_y = local_id / tile_size;
  int local_x = local_id % tile_size;
  int global_y = blockIdx.y * tile_size + local_y;
  int global_x = blockIdx.x * tile_size + local_x;
  if (global_y >= H || global_x >= W) {
    return;
  }

  float pos[2] = {topleft[0] + global_x * pixel_size_x,
                  topleft[1] + global_y * pixel_size_y};

  for (int i = 0; i < N_gaussians_this_time; ++i) {
    if (cum_alpha < thresh) {
      break;
    }
    float alpha_ = fminf(alpha[i], 0.99f);
    float G = kernel_gaussian_2d(mean + 2 * i, cov + 4 * i, pos); // a * G
    assert(G >= 0.0f && G <= 1.0f);
    if (alpha_ * G < MIN_RENDER_ALPHA) {
      continue;
    }
    float coeff = alpha_ * cum_alpha * G; // a * T * G
    out += scalar[i] * coeff;
    atomicAdd(grad_scalar + i, coeff * grad_out);
    float partial_aG = 0.0f;
    partial_aG +=
        grad_out * (scalar[i] * cum_alpha - (final - out) / (1 - alpha_ * G));
    kernel_gaussian_2d_backward(mean + 2 * i, cov + 4 * i, pos,
                                grad_mean + 2 * i, grad_cov + 4 * i,
                                partial_aG * alpha_ * G);
    atomicAdd(grad_alpha + i, partial_aG * G);
    cum_alpha *= (1 - alpha_ * G);
  }
}

__global__ void tile_based_vol_rendering_scalar_backward(
    uint32_t N, uint32_t N_with_dub, float *__restrict__ mean,
    float *__restrict__ cov, float *__restrict__ scalar,
    float *__restrict__ alpha, int *__restrict__ start, int *__restrict__ end,
    int *__restrict__ gaussian_ids, float *__restrict__ out,
    float *__restrict__ grad_mean, float *__restrict__ grad_cov,
    float *__restrict__ grad_scalar, float *__restrict__ grad_alpha,
    float *__restrict__ grad_out, float *__restrict__ topleft,
    uint32_t tile_size, uint32_t n_tiles_h, uint32_t n_tiles_w,
    float pixel_size_x, float pixel_size_y, uint32_t H, uint32_t W,
    float thresh) {
  int local_id = threadIdx.x;
  int local_y = local_id / tile_size;
  int local_x = local_id % tile_size;
  int tile_id = blockIdx.y * gridDim.x + blockIdx.x;
  int global_y = blockIdx.y * tile_size + local_y;
  int global_x = blockIdx.x * tile_size + local_x;
  if (start[tile_id] == -1) {
    // skip
    return;
  }
  int n_gaussians_this_tile = end[tile_id] - start[tile_id];
  if (n_gaussians_this_tile == 0) {
    return;
  }

  // compute memory need for this tile
  int n_float_per_gaussian = 2 + 4 + 1 + 1;
  n_float_per_gaussian *= 2; // for backward
  // mean + cov + scalar + alpha
  int n_pixel_per_tile = tile_size * tile_size;
  int n_float_per_pixel = 3 + 1;
  n_float_per_pixel +=
      (3 + 3); // for backward: 3 for out_rgb, 3 for grad_out_rgb
  // output rgb + cum_alpha
  int max_gaussian_sm = (MAX_N_FLOAT_SM) / n_float_per_gaussian;
  __shared__ float sm[MAX_N_FLOAT_SM];
  float *sm_mean = sm;
  float *sm_cov = sm_mean + 2 * max_gaussian_sm;
  float *sm_scalar = sm_cov + 4 * max_gaussian_sm;
  float *sm_alpha = sm_scalar + 1 * max_gaussian_sm;

  float *sm_grad_mean = sm_alpha + 1 * max_gaussian_sm;
  float *sm_grad_cov = sm_grad_mean + 2 * max_gaussian_sm;
  float *sm_grad_scalar = sm_grad_cov + 4 * max_gaussian_sm;
  float *sm_grad_alpha = sm_grad_scalar + 1 * max_gaussian_sm;

  // float *sm_grad_out = sm_grad_alpha + 1 * max_gaussian_sm;
  // float *sm_final = sm_grad_out + 3 * n_pixel_per_tile;

  gaussian_ids += start[tile_id];
  float cum_alpha = 1.0f;
  float grad_out_local = grad_out[global_y * W + global_x];
  float out_local = out[global_y * W + global_x];
  float out_recomp = 0.0f;

  for (int n = 0; n < n_gaussians_this_tile; n += max_gaussian_sm) {
    int num_gaussian_sm = min(max_gaussian_sm, n_gaussians_this_tile - n);
    carry(num_gaussian_sm, 2, sm_mean, mean, NULL, gaussian_ids);
    carry(num_gaussian_sm, 4, sm_cov, cov, NULL, gaussian_ids);
    carry(num_gaussian_sm, 1, sm_scalar, scalar, NULL, gaussian_ids);
    carry(num_gaussian_sm, 1, sm_alpha, alpha, NULL, gaussian_ids);
    set_zero(2 * num_gaussian_sm, sm_grad_mean);
    set_zero(4 * num_gaussian_sm, sm_grad_cov);
    set_zero(1 * num_gaussian_sm, sm_grad_scalar);
    set_zero(1 * num_gaussian_sm, sm_grad_alpha);
    __syncthreads();
    vol_render_scalar_one_batch_backward(
        num_gaussian_sm, sm_mean, sm_cov, sm_scalar, sm_alpha, sm_grad_mean,
        sm_grad_cov, sm_grad_scalar, sm_grad_alpha, grad_out_local, out_local,
        out_recomp, cum_alpha, topleft, tile_size, n_tiles_h, n_tiles_w,
        pixel_size_x, pixel_size_y, H, W, thresh);
    __syncthreads();
    carry_back(num_gaussian_sm, 2, sm_grad_mean, grad_mean, NULL, gaussian_ids);
    carry_back(num_gaussian_sm, 4, sm_grad_cov, grad_cov, NULL, gaussian_ids);
    carry_back(num_gaussian_sm, 1, sm_grad_scalar, grad_scalar, NULL,
               gaussian_ids);
    carry_back(num_gaussian_sm, 1, sm_grad_alpha, grad_alpha, NULL,
               gaussian_ids);
    __syncthreads();
    gaussian_ids += num_gaussian_sm;
  }
  if (global_x >= W || global_y >= H) {
    return;
  }
  assert(out_recomp == out_local);
}

void vol_rendering_scalar_cuda(
    uint32_t N, uint32_t N_with_dub, float *__restrict__ mean,
    float *__restrict__ cov, float *__restrict__ scalar,
    float *__restrict__ alpha, int *__restrict__ start, int *__restrict__ end,
    int *__restrict__ gaussian_ids, float *__restrict__ out,
    float *__restrict__ topleft, uint32_t tile_size, uint32_t n_tiles_h,
    uint32_t n_tiles_w, float pixel_size_x, float pixel_size_y, uint32_t H,
    uint32_t W, float thresh, float *__restrict__ T) {
  const dim3 block(n_tiles_w, n_tiles_h, 1);
  const int n_pixel_per_tile = tile_size * tile_size;
  tile_based_vol_rendering_scalar<<<block, n_pixel_per_tile>>>(
      N, N_with_dub, mean, cov, scalar, alpha, start, end, gaussian_ids, out,
      topleft, tile_size, n_tiles_h, n_tiles_w, pixel_size_x, pixel_size_y, H,
      W, thresh, T);
  cudaError_t last_error;
  checkLastCudaError(last_error);
}

void vol_rendering_scalar_backward_cuda(
    uint32_t N, uint32_t N_with_dub, float *__restrict__ mean,
    float *__restrict__ cov, float *__restrict__ scalar,
    float *__restrict__ alpha, int *__restrict__ start, int *__restrict__ end,
    int *__restrict__ gaussian_ids, float *__restrict__ out,
    float *__restrict__ grad_mean, float *__restrict__ grad_cov,
    float *__restrict__ grad_scalar, float *__restrict__ grad_alpha,
    float *__restrict__ grad_out, float *__restrict__ topleft,
    uint32_t tile_size, uint32_t n_tiles_h, uint32_t n_tiles_w,
    float pixel_size_x, float pixel_size_y, uint32_t H, uint32_t W,
    float thresh) {
  const dim3 block(n_tiles_w, n_tiles_h, 1);
  const int n_pixel_per_tile = tile_size * tile_size;
  tile_based_vol_rendering_scalar_backward<<<block, n_pixel_per_tile>>>(
      N, N_with_dub, mean, cov, scalar, alpha, start, end, gaussian_ids, out,
      grad_mean, grad_cov, grad_scalar, grad_alpha, grad_out, topleft,
      tile_size, n_tiles_h, n_tiles_w, pixel_size_x, pixel_size_y, H, W,
      thresh);
  cudaError_t last_error;
  checkLastCudaError(last_error);
}