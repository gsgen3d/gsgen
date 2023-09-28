#pragma once
#include "common.h"
#include "data_spec.h"
#include "kernels.h"
#include "shencoder.h"
#include "vol_render_sh.h"
#include <cub/cub.cuh>
#include <device_atomic_functions.h>
#include <device_launch_parameters.h>
#include <stdint.h>

template <uint32_t C>
__global__ void tile_based_vol_rendering_sh_entry_with_bg(
    uint32_t N, uint32_t N_with_dub, float *__restrict__ mean,
    float *__restrict__ cov, float *__restrict__ sh_coeffs,
    float *__restrict__ alpha, int *__restrict__ start, int *__restrict__ end,
    int *__restrict__ gaussian_ids, float *__restrict__ out_rgb,
    float *__restrict__ topleft, float *__restrict__ c2w, uint32_t tile_size,
    uint32_t n_tiles_h, uint32_t n_tiles_w, float pixel_size_x,
    float pixel_size_y, uint32_t H, uint32_t W, float thresh, float *bg_rgb) {
  // C stands for sh degree
  int local_id = threadIdx.x;
  int local_y = local_id / tile_size;
  int local_x = local_id % tile_size;
  int tile_id = blockIdx.y * gridDim.x + blockIdx.x;
  int global_y = blockIdx.y * tile_size + local_y;
  int global_x = blockIdx.x * tile_size + local_x;

  float sh_consts[C * C];
  float direction[3];
  float out[3] = {0.0f, 0.0f, 0.0f};
  float cum_alpha = 1.0f;

  if (start[tile_id] == -1) {
    // skip
    if (global_y >= H || global_x >= W) {
      return;
    }
    out_rgb[3 * (global_y * W + global_x) + 0] = bg_rgb[0];
    out_rgb[3 * (global_y * W + global_x) + 1] = bg_rgb[1];
    out_rgb[3 * (global_y * W + global_x) + 2] = bg_rgb[2];
    return;
  }
  int n_gaussians_this_tile = end[tile_id] - start[tile_id];
  if (n_gaussians_this_tile == 0) {
    if (global_y >= H || global_x >= W) {
      return;
    }
    out_rgb[3 * (global_y * W + global_x) + 0] = bg_rgb[0];
    out_rgb[3 * (global_y * W + global_x) + 1] = bg_rgb[1];
    out_rgb[3 * (global_y * W + global_x) + 2] = bg_rgb[2];
    return;
  }

  // compute memory need for this tile
  int n_float_per_gaussian = 2 + 4 + 3 * C * C + 1;
  // mean + cov + color + alpha
  int n_pixel_per_tile = tile_size * tile_size;
  // output rgb + cum_alpha
  int max_gaussian_sm = (MAX_N_FLOAT_SM) / n_float_per_gaussian;
  __shared__ float sm[MAX_N_FLOAT_SM];
  float *sm_mean = sm;
  float *sm_cov = sm_mean + 2 * max_gaussian_sm;
  float *sm_sh_coeffs = sm_cov + 4 * max_gaussian_sm;
  float *sm_alpha = sm_sh_coeffs + 3 * C * C * max_gaussian_sm;

  float pos[3] = {topleft[0] + global_x * pixel_size_x,
                  topleft[1] + global_y * pixel_size_y, 1.0f};
  // calculate direction for sh rendering
  calc_direction(direction, pos, c2w);
  spherical_harmonic(direction, sh_consts, C);

  // DEBUG
  // if (threadIdx.x == 0) {
  //   printf("sh_consts[0] = %f\n", sh_consts[0]);
  // }

  gaussian_ids += start[tile_id];

  for (int n = 0; n < n_gaussians_this_tile; n += max_gaussian_sm) {
    int num_gaussian_sm = min(max_gaussian_sm, n_gaussians_this_tile - n);
    carry(num_gaussian_sm, 2, sm_mean, mean, gaussian_ids);
    carry(num_gaussian_sm, 4, sm_cov, cov, gaussian_ids);
    carry(num_gaussian_sm, 3 * C * C, sm_sh_coeffs, sh_coeffs, gaussian_ids);
    carry(num_gaussian_sm, 1, sm_alpha, alpha, gaussian_ids);
    __syncthreads();
    vol_render_one_batch_sh<C>(num_gaussian_sm, sm_mean, sm_cov, sm_sh_coeffs,
                               sm_alpha, sh_consts, out, cum_alpha, topleft,
                               tile_size, n_tiles_h, n_tiles_w, pixel_size_x,
                               pixel_size_y, H, W, thresh);
    __syncthreads();
    gaussian_ids += num_gaussian_sm;
  }
  if (global_y >= H || global_x >= W) {
    return;
  }
  // if (cum_alpha > thresh) {
  //   out[0] = out[0] * cum_alpha + bg_rgb[0] * (1.0f - cum_alpha);
  //   out[1] = out[1] * cum_alpha + bg_rgb[1] * (1.0f - cum_alpha);
  //   out[2] = out[2] * cum_alpha + bg_rgb[2] * (1.0f - cum_alpha);
  // }
  out[0] = out[0] + bg_rgb[0] * cum_alpha;
  out[1] = out[1] + bg_rgb[1] * cum_alpha;
  out[2] = out[2] + bg_rgb[2] * cum_alpha;
  out_rgb[3 * (global_y * W + global_x) + 0] = out[0];
  out_rgb[3 * (global_y * W + global_x) + 1] = out[1];
  out_rgb[3 * (global_y * W + global_x) + 2] = out[2];

  /* test kernel 2d */
}

template <uint32_t C>
void tile_based_vol_rendering_sh_cuda_with_bg(
    uint32_t N, uint32_t N_with_dub, float *mean, float *cov, float *sh_coeffs,
    float *alpha, int *start, int *end, int *gaussian_ids, float *out_rgb,
    float *topleft, float *c2w, uint32_t tile_size, uint32_t n_tiles_h,
    uint32_t n_tiles_w, float pixel_size_x, float pixel_size_y, uint32_t H,
    uint32_t W, float thresh, float *bg_rgb, cudaStream_t stream) {
  const dim3 block(n_tiles_w, n_tiles_h, 1);
  const int n_pixel_per_tile = tile_size * tile_size;
  tile_based_vol_rendering_sh_entry_with_bg<C>
      <<<block, n_pixel_per_tile, 0, stream>>>(
          N, N_with_dub, mean, cov, sh_coeffs, alpha, start, end, gaussian_ids,
          out_rgb, topleft, c2w, tile_size, n_tiles_h, n_tiles_w, pixel_size_x,
          pixel_size_y, H, W, thresh, bg_rgb);

  cudaError_t last_error;
  checkLastCudaError(last_error);
}

template <uint32_t C>
__global__ void tile_based_vol_rendering_backward_sh_entry_with_bg(
    uint32_t N, uint32_t N_with_dub, float *__restrict__ mean,
    float *__restrict__ cov, float *__restrict__ sh_coeffs,
    float *__restrict__ alpha, int *__restrict__ start, int *__restrict__ end,
    int *__restrict__ gaussian_ids, float *__restrict__ out_rgb,
    float *__restrict__ grad_mean, float *__restrict__ grad_cov,
    float *__restrict__ grad_sh_coeffs, float *__restrict__ grad_alpha,
    float *__restrict__ grad_out_rgb, float *__restrict__ topleft,
    float *__restrict__ c2w, uint32_t tile_size, uint32_t n_tiles_h,
    uint32_t n_tiles_w, float pixel_size_x, float pixel_size_y, uint32_t H,
    uint32_t W, float thresh, float *bg_rgb) {
  int local_id = threadIdx.x;
  int local_y = local_id / tile_size;
  int local_x = local_id % tile_size;
  int tile_id = blockIdx.y * gridDim.x + blockIdx.x;
  int global_y = blockIdx.y * tile_size + local_y;
  int global_x = blockIdx.x * tile_size + local_x;
  if (start[tile_id] == -1) {
    // skip
    if (global_y >= H || global_x >= W) {
      return;
    }
    // out[0] = bg_rgb[0];
    // out[1] = bg_rgb[1];
    // out[2] = bg_rgb[2];
    return;
  }
  int n_gaussians_this_tile = end[tile_id] - start[tile_id];
  if (n_gaussians_this_tile == 0) {
    return;
  }

  int n_float_per_gaussian = 2 + 4 + 3 * C * C + 1;
  n_float_per_gaussian *= 2; // for backward
  // mean + cov + color + alpha
  int n_pixel_per_tile = tile_size * tile_size;
  int max_gaussian_sm = (MAX_N_FLOAT_SM) / n_float_per_gaussian;

  float grad_out[3];
  float final[3];
  float out[3] = {0.0f, 0.0f, 0.0f};
  float cum_alpha = 1.0f;

  float sh_consts[C * C];
  float direction[3];
  float pos[3] = {topleft[0] + global_x * pixel_size_x,
                  topleft[1] + global_y * pixel_size_y, 1.0f};
  calc_direction(direction, pos, c2w);
  spherical_harmonic(direction, sh_consts, C);

  __shared__ float sm[MAX_N_FLOAT_SM];
  float *sm_mean = sm;
  float *sm_cov = sm_mean + 2 * max_gaussian_sm;
  float *sm_sh_coeffs = sm_cov + 4 * max_gaussian_sm;
  float *sm_alpha = sm_sh_coeffs + 3 * C * C * max_gaussian_sm;

  float *sm_grad_mean = sm_alpha + 1 * max_gaussian_sm;
  float *sm_grad_cov = sm_grad_mean + 2 * max_gaussian_sm;
  float *sm_grad_sh_coeffs = sm_grad_cov + 4 * max_gaussian_sm;
  float *sm_grad_alpha = sm_grad_sh_coeffs + 3 * C * C * max_gaussian_sm;

  if (global_x < W && global_y < H) {
#pragma unroll
    for (size_t i = 0; i < 3; ++i) {
      grad_out[i] = grad_out_rgb[3 * (global_y * W + global_x) + i];
      final[i] = out_rgb[3 * (global_y * W + global_x) + i];
    }
  }

  gaussian_ids += start[tile_id];

  for (int n = 0; n < n_gaussians_this_tile; n += max_gaussian_sm) {
    int num_gaussian_sm = min(max_gaussian_sm, n_gaussians_this_tile - n);
    carry(num_gaussian_sm, 2, sm_mean, mean, gaussian_ids);
    carry(num_gaussian_sm, 4, sm_cov, cov, gaussian_ids);
    carry(num_gaussian_sm, 3 * C * C, sm_sh_coeffs, sh_coeffs, gaussian_ids);
    carry(num_gaussian_sm, 1, sm_alpha, alpha, gaussian_ids);
    set_zero(2 * num_gaussian_sm, sm_grad_mean);
    set_zero(4 * num_gaussian_sm, sm_grad_cov);
    set_zero(3 * num_gaussian_sm * C * C, sm_grad_sh_coeffs);
    set_zero(1 * num_gaussian_sm, sm_grad_alpha);
    __syncthreads();
    vol_render_one_batch_sh_backward<C>(
        num_gaussian_sm, sm_mean, sm_cov, sm_sh_coeffs, sm_alpha, sm_grad_mean,
        sm_grad_cov, sm_grad_sh_coeffs, sm_grad_alpha, grad_out, final, out,
        cum_alpha, sh_consts, topleft, tile_size, n_tiles_h, n_tiles_w,
        pixel_size_x, pixel_size_y, H, W, thresh);
    __syncthreads();
    carry_back(num_gaussian_sm, 2, sm_grad_mean, grad_mean, NULL, gaussian_ids);
    carry_back(num_gaussian_sm, 4, sm_grad_cov, grad_cov, NULL, gaussian_ids);
    carry_back(num_gaussian_sm, 3 * C * C, sm_grad_sh_coeffs, grad_sh_coeffs,
               NULL, gaussian_ids);
    carry_back(num_gaussian_sm, 1, sm_grad_alpha, grad_alpha, NULL,
               gaussian_ids);
    __syncthreads();
    gaussian_ids += num_gaussian_sm;
  }
  if (global_x >= W || global_y >= H) {
    return;
  }
  out[0] = out[0] + bg_rgb[0] * cum_alpha;
  out[1] = out[1] + bg_rgb[1] * cum_alpha;
  out[2] = out[2] + bg_rgb[2] * cum_alpha;
  if (out_rgb[3 * (global_y * W + global_x) + 0] != out[0]) {
    printf("out_rgb[3 * (global_y * W + global_x) + 0] = %f, out[0] = %f\n",
           out_rgb[3 * (global_y * W + global_x) + 0], out[0]);
  }
  assert(out_rgb[3 * (global_y * W + global_x) + 0] == out[0]);
  assert(out_rgb[3 * (global_y * W + global_x) + 1] == out[1]);
  assert(out_rgb[3 * (global_y * W + global_x) + 2] == out[2]);
}

template <uint32_t C>
void tile_based_vol_rendering_backward_sh_cuda_with_bg(
    uint32_t N, uint32_t N_with_dub, float *__restrict__ mean,
    float *__restrict__ cov, float *__restrict__ sh_coeffs,
    float *__restrict__ alpha, int *__restrict__ start, int *__restrict__ end,
    int *__restrict__ gaussian_ids, float *__restrict__ out_rgb,
    float *__restrict__ grad_mean, float *__restrict__ grad_cov,
    float *__restrict__ grad_sh_coeffs, float *__restrict__ grad_alpha,
    float *__restrict__ grad_out_rgb, float *__restrict__ topleft,
    float *__restrict__ c2w, uint32_t tile_size, uint32_t n_tiles_h,
    uint32_t n_tiles_w, float pixel_size_x, float pixel_size_y, uint32_t H,
    uint32_t W, float thresh, float *bg_rgb, cudaStream_t stream) {
  const dim3 block(n_tiles_w, n_tiles_h, 1);
  const int n_pixel_per_tile = tile_size * tile_size;
  tile_based_vol_rendering_backward_sh_entry_with_bg<C>
      <<<block, n_pixel_per_tile, 0, stream>>>(
          N, N_with_dub, mean, cov, sh_coeffs, alpha, start, end, gaussian_ids,
          out_rgb, grad_mean, grad_cov, grad_sh_coeffs, grad_alpha,
          grad_out_rgb, topleft, c2w, tile_size, n_tiles_h, n_tiles_w,
          pixel_size_x, pixel_size_y, H, W, thresh, bg_rgb);

  cudaError_t last_error;
  checkLastCudaError(last_error);
}