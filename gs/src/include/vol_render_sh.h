#pragma once
#include "common.h"
#include "data_spec.h"
#include "kernels.h"
#include "shencoder.h"
#include <cub/cub.cuh>
#include <device_atomic_functions.h>
#include <device_launch_parameters.h>
#include <stdint.h>

__device__ __forceinline__ float warpSum(float val, unsigned int mask) {
#pragma unroll
  for (int i = 1; i < 32; i *= 2)
    val += __shfl_xor_sync(mask, val, i);
  return val;
}

template <uint32_t C>
__device__ __forceinline__ float sum_C(float *a, float *b) {
  float summation = 0.0f;
#pragma unroll
  for (int i = 0; i < C * C; ++i) {
    summation += a[i] * b[i];
  }
  return summation;
}

template <uint32_t C>
__device__ __forceinline__ void backward_C(float *__restrict__ grad_sh_coeffs,
                                           float *__restrict__ sh_consts,
                                           float grad) {
#pragma unroll
  for (int i = 0; i < C * C; ++i) {
    atomicAdd(grad_sh_coeffs + i, grad * sh_consts[i]);
  }
}

template <uint32_t C>
__device__ __forceinline__ void
backward_C_nonatomic(float *__restrict__ grad_sh_coeffs,
                     float *__restrict__ sh_consts, float grad) {
#pragma unroll
  for (int i = 0; i < C * C; ++i) {
    grad_sh_coeffs[i] = grad * sh_consts[i];
  }
}

__device__ __forceinline__ void calc_direction(float *__restrict__ direction,
                                               float *__restrict__ pos,
                                               float *__restrict__ c2w) {
  float3 *pos_ = reinterpret_cast<float3 *>(pos);
  float3 *c2w_ = reinterpret_cast<float3 *>(c2w);
  direction[0] = dot(c2w_[0], *pos_);
  direction[1] = dot(c2w_[1], *pos_);
  direction[2] = dot(c2w_[2], *pos_);
  float length =
      sqrtf(direction[0] * direction[0] + direction[1] * direction[1] +
            direction[2] * direction[2]);
  direction[0] /= length;
  direction[1] /= length;
  direction[2] /= length;
  checkValue(direction[0]);
  checkValue(direction[1]);
  checkValue(direction[2]);
}

__device__ __forceinline__ void carry(uint32_t N, uint32_t dsize, float *sm,
                                      float *gm, int *gaussian_ids) {
  /*carry from global memory to scratchpad memory with (N * dsize) float32s */
  int local_id = threadIdx.x;
  int n_turns = (dsize * N) / blockDim.x;
  int n_left = (dsize * N) % blockDim.x;
  for (int i = 0; i < n_turns; i++) {
    sm[local_id + i * blockDim.x] =
        gm[dsize * gaussian_ids[(local_id + i * blockDim.x) / dsize] +
           ((local_id + i * blockDim.x) % dsize)];
  }
  if (local_id < n_left) {
    sm[local_id + n_turns * blockDim.x] =
        gm[dsize * gaussian_ids[(local_id + n_turns * blockDim.x) / dsize] +
           ((local_id + n_turns * blockDim.x) % dsize)];
  }
}

template <uint32_t D>
__device__ __forceinline__ void carry_items(uint32_t N, float *sm, float *gm,
                                            int *gaussian_ids) {
  for (uint32_t i = 0; i < N; ++i) {
#pragma unroll
    for (uint32_t d = 0; d < D; ++d) {
      sm[i * D + d] = gm[gaussian_ids[i] * D + d];
    }
  }
  return;
}

template <uint32_t C>
__device__ void vol_render_one_batch_sh(
    uint32_t N_gaussians_this_time, float *mean, float *cov, float *sh_coeffs,
    float *alpha, float *sh_consts, float *out, float &cum_alpha,
    float *topleft, uint32_t tile_size, uint32_t n_tiles_h, uint32_t n_tiles_w,
    float pixel_size_x, float pixel_size_y, uint32_t H, uint32_t W,
    float thresh) {
  int local_id = threadIdx.x;
  // check row major here
  int local_y = local_id / tile_size;
  int local_x = local_id % tile_size;
  int global_y = blockIdx.y * tile_size + local_y;
  int global_x = blockIdx.x * tile_size + local_x;
  if (global_y >= H || global_x >= W) {
    return;
  }
  // float color_this_time[3];
  // float cum_alpha_this_time;

  // position in camera coordinate
  float pos[2] = {topleft[0] + global_x * pixel_size_x,
                  topleft[1] + global_y * pixel_size_y};

  for (int i = 0; i < N_gaussians_this_time; ++i) {
    if (cum_alpha < thresh) {
      break;
    }
    float alpha_ = fminf(alpha[i], 0.99f);
    float coeff = alpha_ * cum_alpha;
    // float val = kernel_gaussian_2d(mean + 2 * i, cov + 4 * i, pos);
    float val = kernel_gaussian_2d_float(mean + 2 * i, cov + 4 * i, pos);
    // float val = 1.0f;
    coeff *= val;
    if (alpha_ * val < MIN_RENDER_ALPHA) {
      continue;
    }
    if (isnan(coeff)) {
      coeff = 0.0f;
    }
    // assert(alpha_ * val >= 0.0f && alpha_ * val <= 1.0f);
    // checkValue(coeff);
    // assert(coeff >= 0.0f && coeff <= 1.0f);
    float y0 = SIGMOID(sum_C<C>(sh_coeffs + 3 * i * C * C, sh_consts));
    float y1 = SIGMOID(sum_C<C>(sh_coeffs + C * C * (3 * i + 1), sh_consts));
    float y2 = SIGMOID(sum_C<C>(sh_coeffs + (3 * i + 2) * C * C, sh_consts));
    // assert(y0 >= 0.0f && y0 <= 1.0f);
    // assert(y1 >= 0.0f && y1 <= 1.0f);
    // assert(y2 >= 0.0f && y2 <= 1.0f);
    // assert(coeff * y0 >= 0.0f && coeff * y0 <= 1.0f);
    // assert(coeff * y1 >= 0.0f && coeff * y1 <= 1.0f);
    // assert(coeff * y2 >= 0.0f && coeff * y2 <= 1.0f);
    // if (isnan(coeff * y0) || isnan(coeff * y1) || isnan(coeff * y2)) {
    //   continue;
    // }
    if (isnan(y0 * coeff)) {
      y0 = 0.0f;
    }
    if (isnan(y1 * coeff)) {
      y1 = 0.0f;
    }
    if (isnan(y2 * coeff)) {
      y2 = 0.0f;
    }
    out[0] += coeff * y0;
    out[1] += coeff * y1;
    out[2] += coeff * y2;
    // checkValue(out[0]);
    // checkValue(out[1]);
    // checkValue(out[2]);
    // printf("out: %f %f %f\n", out[0], out[1], out[2]);
    cum_alpha *= (1 - alpha_ * val);
  }
}

template <uint32_t C>
__global__ void tile_based_vol_rendering_sh_entry(
    uint32_t N, uint32_t N_with_dub, float *__restrict__ mean,
    float *__restrict__ cov, float *__restrict__ sh_coeffs,
    float *__restrict__ alpha, int *__restrict__ start, int *__restrict__ end,
    int *__restrict__ gaussian_ids, float *__restrict__ out_rgb,
    float *__restrict__ topleft, float *__restrict__ c2w, uint32_t tile_size,
    uint32_t n_tiles_h, uint32_t n_tiles_w, float pixel_size_x,
    float pixel_size_y, uint32_t H, uint32_t W, float thresh) {
  // C stands for sh degree
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
  int global_y = blockIdx.y * tile_size + local_y;
  int global_x = blockIdx.x * tile_size + local_x;

  float sh_consts[C * C];
  float direction[3];
  float out[3] = {0.0f, 0.0f, 0.0f};
  float cum_alpha = 1.0f;

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
  out_rgb[3 * (global_y * W + global_x) + 0] = out[0];
  out_rgb[3 * (global_y * W + global_x) + 1] = out[1];
  out_rgb[3 * (global_y * W + global_x) + 2] = out[2];

  /* test kernel 2d */
}

template <uint32_t C>
void tile_based_vol_rendering_sh_cuda(
    uint32_t N, uint32_t N_with_dub, float *mean, float *cov, float *sh_coeffs,
    float *alpha, int *start, int *end, int *gaussian_ids, float *out_rgb,
    float *topleft, float *c2w, uint32_t tile_size, uint32_t n_tiles_h,
    uint32_t n_tiles_w, float pixel_size_x, float pixel_size_y, uint32_t H,
    uint32_t W, float thresh, cudaStream_t stream) {
  const dim3 block(n_tiles_w, n_tiles_h, 1);
  const int n_pixel_per_tile = tile_size * tile_size;
  tile_based_vol_rendering_sh_entry<C><<<block, n_pixel_per_tile, 0, stream>>>(
      N, N_with_dub, mean, cov, sh_coeffs, alpha, start, end, gaussian_ids,
      out_rgb, topleft, c2w, tile_size, n_tiles_h, n_tiles_w, pixel_size_x,
      pixel_size_y, H, W, thresh);

  cudaError_t last_error;
  checkLastCudaError(last_error);
}

template <uint32_t C>
__device__ void vol_render_one_batch_sh_backward(
    uint32_t N_gaussians_this_time, float *__restrict__ mean,
    float *__restrict__ cov, float *__restrict__ sh_coeffs,
    float *__restrict__ alpha, float *__restrict__ grad_mean,
    float *__restrict__ grad_cov, float *__restrict__ grad_sh_coeffs,
    float *__restrict__ grad_alpha, float *__restrict__ grad_out,
    float *__restrict__ final, float *__restrict__ out, float &cum_alpha,
    float *__restrict__ sh_consts, float *__restrict__ topleft,
    uint32_t tile_size, uint32_t n_tiles_h, uint32_t n_tiles_w,
    float pixel_size_x, float pixel_size_y, uint32_t H, uint32_t W,
    float thresh) {
  int local_id = threadIdx.x;
  // check row major here
  int local_y = local_id / tile_size;
  int local_x = local_id % tile_size;
  int global_y = blockIdx.y * tile_size + local_y;
  int global_x = blockIdx.x * tile_size + local_x;
  if (global_y >= H || global_x >= W) {
    return;
  }
  uint32_t CC = C * C;

  // position in camera coordinate
  float pos[2] = {topleft[0] + global_x * pixel_size_x,
                  topleft[1] + global_y * pixel_size_y};

  for (int i = 0; i < N_gaussians_this_time; ++i) {
    if (cum_alpha < thresh) {
      break;
    }
    float alpha_ = fminf(alpha[i], 0.99f);
    float G = kernel_gaussian_2d_float(mean + 2 * i, cov + 4 * i, pos); // a * G
    // assert(G >= 0.0f && G <= 1.0f);
    if (alpha_ * G < MIN_RENDER_ALPHA) {
      continue;
    }
    float coeff = alpha_ * cum_alpha * G; // a * T * G
    if (isnan(coeff)) {
      coeff = 0.0f;
    }
    // checkValue(coeff);
    float y0 = SIGMOID(sum_C<C>(sh_coeffs + 3 * i * CC, sh_consts));
    float y1 = SIGMOID(sum_C<C>(sh_coeffs + (3 * i + 1) * CC, sh_consts));
    float y2 = SIGMOID(sum_C<C>(sh_coeffs + (3 * i + 2) * CC, sh_consts));
    // if (isnan(coeff * y0) || isnan(coeff * y1) || isnan(coeff * y2)) {
    //   continue;
    // }
    if (isnan(y0 * coeff)) {
      y0 = 0.0f;
    }
    if (isnan(y1 * coeff)) {
      y1 = 0.0f;
    }
    if (isnan(y2 * coeff)) {
      y2 = 0.0f;
    }
    out[0] += coeff * y0;
    out[1] += coeff * y1;
    out[2] += coeff * y2;
    backward_C<C>(grad_sh_coeffs + (3 * i + 0) * CC, sh_consts,
                  coeff * SIGMOID_DSIGMOID(y0) * grad_out[0]);
    backward_C<C>(grad_sh_coeffs + (3 * i + 1) * CC, sh_consts,
                  coeff * SIGMOID_DSIGMOID(y1) * grad_out[1]);
    backward_C<C>(grad_sh_coeffs + (3 * i + 2) * CC, sh_consts,
                  coeff * SIGMOID_DSIGMOID(y2) * grad_out[2]);

    // double partial_aG = 0.0;
    float partial_aG = 0.0f;
    partial_aG +=
        grad_out[0] * (y0 * cum_alpha - (final[0] - out[0]) / (1 - alpha_ * G));
    partial_aG +=
        grad_out[1] * (y1 * cum_alpha - (final[1] - out[1]) / (1 - alpha_ * G));
    partial_aG +=
        grad_out[2] * (y2 * cum_alpha - (final[2] - out[2]) / (1 - alpha_ * G));
    kernel_gaussian_2d_backward(mean + 2 * i, cov + 4 * i, pos,
                                grad_mean + 2 * i, grad_cov + 4 * i,
                                partial_aG * alpha_ * G);
    atomicAdd(grad_alpha + i, partial_aG * G);
    cum_alpha *= (1 - alpha_ * G);
    // sh_coeffs += 3 * C * C;
    // grad_sh_coeffs += 3 * C * C;
  }
}

template <uint32_t C>
__global__ void tile_based_vol_rendering_backward_sh_entry(
    uint32_t N, uint32_t N_with_dub, float *__restrict__ mean,
    float *__restrict__ cov, float *__restrict__ sh_coeffs,
    float *__restrict__ alpha, int *__restrict__ start, int *__restrict__ end,
    int *__restrict__ gaussian_ids, float *__restrict__ out_rgb,
    float *__restrict__ grad_mean, float *__restrict__ grad_cov,
    float *__restrict__ grad_sh_coeffs, float *__restrict__ grad_alpha,
    float *__restrict__ grad_out_rgb, float *__restrict__ topleft,
    float *__restrict__ c2w, uint32_t tile_size, uint32_t n_tiles_h,
    uint32_t n_tiles_w, float pixel_size_x, float pixel_size_y, uint32_t H,
    uint32_t W, float thresh) {
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
  if (out_rgb[3 * (global_y * W + global_x) + 0] != out[0]) {
    printf("out_rgb[3 * (global_y * W + global_x) + 0] = %f, out[0] = %f\n",
           out_rgb[3 * (global_y * W + global_x) + 0], out[0]);
  }
  assert(out_rgb[3 * (global_y * W + global_x) + 0] == out[0]);
  assert(out_rgb[3 * (global_y * W + global_x) + 1] == out[1]);
  assert(out_rgb[3 * (global_y * W + global_x) + 2] == out[2]);
}

template <uint32_t C>
void tile_based_vol_rendering_backward_sh_cuda(
    uint32_t N, uint32_t N_with_dub, float *__restrict__ mean,
    float *__restrict__ cov, float *__restrict__ sh_coeffs,
    float *__restrict__ alpha, int *__restrict__ start, int *__restrict__ end,
    int *__restrict__ gaussian_ids, float *__restrict__ out_rgb,
    float *__restrict__ grad_mean, float *__restrict__ grad_cov,
    float *__restrict__ grad_sh_coeffs, float *__restrict__ grad_alpha,
    float *__restrict__ grad_out_rgb, float *__restrict__ topleft,
    float *__restrict__ c2w, uint32_t tile_size, uint32_t n_tiles_h,
    uint32_t n_tiles_w, float pixel_size_x, float pixel_size_y, uint32_t H,
    uint32_t W, float thresh, cudaStream_t stream) {
  const dim3 block(n_tiles_w, n_tiles_h, 1);
  const int n_pixel_per_tile = tile_size * tile_size;
  tile_based_vol_rendering_backward_sh_entry<C>
      <<<block, n_pixel_per_tile, 0, stream>>>(
          N, N_with_dub, mean, cov, sh_coeffs, alpha, start, end, gaussian_ids,
          out_rgb, grad_mean, grad_cov, grad_sh_coeffs, grad_alpha,
          grad_out_rgb, topleft, c2w, tile_size, n_tiles_h, n_tiles_w,
          pixel_size_x, pixel_size_y, H, W, thresh);

  cudaError_t last_error;
  checkLastCudaError(last_error);
}

template <uint32_t C>
__global__ void tile_based_vol_rendering_backward_sh_entry_v1(
    uint32_t N, uint32_t N_with_dub, float *__restrict__ mean,
    float *__restrict__ cov, float *__restrict__ sh_coeffs,
    float *__restrict__ alpha, int *__restrict__ start, int *__restrict__ end,
    int *__restrict__ gaussian_ids, float *__restrict__ out_rgb,
    float *__restrict__ grad_mean, float *__restrict__ grad_cov,
    float *__restrict__ grad_sh_coeffs, float *__restrict__ grad_alpha,
    float *__restrict__ grad_out_rgb, float *__restrict__ topleft,
    float *__restrict__ c2w, uint32_t tile_size, uint32_t n_tiles_h,
    uint32_t n_tiles_w, float pixel_size_x, float pixel_size_y, uint32_t H,
    uint32_t W, float thresh) {
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

  int n_float_per_gaussian = 2 + 4 + 3 * C * C + 1;
  n_float_per_gaussian *= 2; // for backward
  n_float_per_gaussian += 1;
  // mean + cov + color + alpha + gaussian_id
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

  int *sm_gaussian_ids = (int *)(sm_grad_alpha + 1 * max_gaussian_sm);

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
    direct_carry(num_gaussian_sm, 1, sm_gaussian_ids, gaussian_ids);
    carry(num_gaussian_sm, 2, sm_mean, mean, sm_gaussian_ids);
    carry(num_gaussian_sm, 4, sm_cov, cov, sm_gaussian_ids);
    carry(num_gaussian_sm, 3 * C * C, sm_sh_coeffs, sh_coeffs, sm_gaussian_ids);
    carry(num_gaussian_sm, 1, sm_alpha, alpha, sm_gaussian_ids);
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
    carry_back(num_gaussian_sm, 2, sm_grad_mean, grad_mean, NULL,
               sm_gaussian_ids);
    carry_back(num_gaussian_sm, 4, sm_grad_cov, grad_cov, NULL,
               sm_gaussian_ids);
    carry_back(num_gaussian_sm, 3 * C * C, sm_grad_sh_coeffs, grad_sh_coeffs,
               NULL, sm_gaussian_ids);
    carry_back(num_gaussian_sm, 1, sm_grad_alpha, grad_alpha, NULL,
               sm_gaussian_ids);
    __syncthreads();
    gaussian_ids += num_gaussian_sm;
  }
  if (global_x >= W || global_y >= H) {
    return;
  }
  // if (out_rgb[3 * (global_y * W + global_x) + 0] != out[0]) {
  //   printf("out_rgb[3 * (global_y * W + global_x) + 0] = %f, out[0] = %f\n",
  //          out_rgb[3 * (global_y * W + global_x) + 0], out[0]);
  // }
  // assert(out_rgb[3 * (global_y * W + global_x) + 0] == out[0]);
  // assert(out_rgb[3 * (global_y * W + global_x) + 1] == out[1]);
  // assert(out_rgb[3 * (global_y * W + global_x) + 2] == out[2]);
}

template <uint32_t C>
void tile_based_vol_rendering_backward_sh_cuda_v1(
    uint32_t N, uint32_t N_with_dub, float *__restrict__ mean,
    float *__restrict__ cov, float *__restrict__ sh_coeffs,
    float *__restrict__ alpha, int *__restrict__ start, int *__restrict__ end,
    int *__restrict__ gaussian_ids, float *__restrict__ out_rgb,
    float *__restrict__ grad_mean, float *__restrict__ grad_cov,
    float *__restrict__ grad_sh_coeffs, float *__restrict__ grad_alpha,
    float *__restrict__ grad_out_rgb, float *__restrict__ topleft,
    float *__restrict__ c2w, uint32_t tile_size, uint32_t n_tiles_h,
    uint32_t n_tiles_w, float pixel_size_x, float pixel_size_y, uint32_t H,
    uint32_t W, float thresh) {
  const dim3 block(n_tiles_w, n_tiles_h, 1);
  const int n_pixel_per_tile = tile_size * tile_size;
  tile_based_vol_rendering_backward_sh_entry_v1<C><<<block, n_pixel_per_tile>>>(
      N, N_with_dub, mean, cov, sh_coeffs, alpha, start, end, gaussian_ids,
      out_rgb, grad_mean, grad_cov, grad_sh_coeffs, grad_alpha, grad_out_rgb,
      topleft, c2w, tile_size, n_tiles_h, n_tiles_w, pixel_size_x, pixel_size_y,
      H, W, thresh);

  cudaError_t last_error;
  checkLastCudaError(last_error);
}

// with cooperative groups
template <uint32_t C>
__device__ void vol_render_one_batch_sh_backward_warp_reduce(
    uint32_t N_gaussians_this_time, float *__restrict__ mean,
    float *__restrict__ cov, float *__restrict__ sh_coeffs,
    float *__restrict__ alpha, float *__restrict__ grad_mean,
    float *__restrict__ grad_cov, float *__restrict__ grad_sh_coeffs,
    float *__restrict__ grad_alpha, float *__restrict__ grad_out,
    float *__restrict__ final, float *__restrict__ out, float &cum_alpha,
    float *__restrict__ sh_consts, float *__restrict__ topleft,
    uint32_t tile_size, uint32_t n_tiles_h, uint32_t n_tiles_w,
    float pixel_size_x, float pixel_size_y, uint32_t H, uint32_t W,
    float thresh) {
  int local_id = threadIdx.x;
  // check row major here
  int local_y = local_id / tile_size;
  int local_x = local_id % tile_size;
  int global_y = blockIdx.y * tile_size + local_y;
  int global_x = blockIdx.x * tile_size + local_x;
  if (global_y >= H || global_x >= W) {
    return;
  }
  uint32_t CC = C * C;

  // position in camera coordinate
  float pos[2] = {topleft[0] + global_x * pixel_size_x,
                  topleft[1] + global_y * pixel_size_y};

  float local_grad_mean[2] = {0.0f, 0.0f};
  float local_grad_cov[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float local_grad_sh_coeffs[3 * C * C] = {0.0f};
  float local_grad_alpha = 0.0f;
  // float test_local_grad_alpha = warpSum(local_grad_alpha);
  // #pragma unroll
  // for (int i = 1; i < 32; i *= 2)
  //   local_grad_alpha += __shfl_xor_sync(-1, local_grad_alpha, i);
  // auto active = cg::coalesced_threads();
  // float val = 1.0f;
  // for (uint32_t gap = 32 >> 1; gap > 0; gap >>= 1) {
  //   // val += __shfl_down_sync(mask, val, gap);
  //   val += active.shfl_down(val, gap);
  // }
  // printf("test_local_grad_alpha = %f\n", val);

  for (int i = 0; i < N_gaussians_this_time; ++i) {
    if (cum_alpha < thresh) {
      break;
    }
    float alpha_ = fminf(alpha[i], 0.99f);
    float G = kernel_gaussian_2d_float(mean + 2 * i, cov + 4 * i, pos); // a * G
    // assert(G >= 0.0f && G <= 1.0f);
    if (alpha_ * G < MIN_RENDER_ALPHA) {
      continue;
    }
    float coeff = alpha_ * cum_alpha * G; // a * T * G
    if (isnan(coeff)) {
      coeff = 0.0f;
    }
    // checkValue(coeff);
    float y0 = SIGMOID(sum_C<C>(sh_coeffs + 3 * i * CC, sh_consts));
    float y1 = SIGMOID(sum_C<C>(sh_coeffs + (3 * i + 1) * CC, sh_consts));
    float y2 = SIGMOID(sum_C<C>(sh_coeffs + (3 * i + 2) * CC, sh_consts));
    if (isnan(y0 * coeff)) {
      y0 = 0.0f;
    }
    if (isnan(y1 * coeff)) {
      y1 = 0.0f;
    }
    if (isnan(y2 * coeff)) {
      y2 = 0.0f;
    }
    out[0] += coeff * y0;
    out[1] += coeff * y1;
    out[2] += coeff * y2;
    backward_C_nonatomic<C>(local_grad_sh_coeffs, sh_consts,
                            coeff * SIGMOID_DSIGMOID(y0) * grad_out[0]);
    backward_C_nonatomic<C>(local_grad_sh_coeffs + 1 * CC, sh_consts,
                            coeff * SIGMOID_DSIGMOID(y1) * grad_out[1]);
    backward_C_nonatomic<C>(local_grad_sh_coeffs + 2 * CC, sh_consts,
                            coeff * SIGMOID_DSIGMOID(y2) * grad_out[2]);

    float partial_aG = 0.0;
    partial_aG +=
        grad_out[0] * (y0 * cum_alpha - (final[0] - out[0]) / (1 - alpha_ * G));
    partial_aG +=
        grad_out[1] * (y1 * cum_alpha - (final[1] - out[1]) / (1 - alpha_ * G));
    partial_aG +=
        grad_out[2] * (y2 * cum_alpha - (final[2] - out[2]) / (1 - alpha_ * G));
    kernel_gaussian_2d_backward_nonatomic(mean + 2 * i, cov + 4 * i, pos,
                                          local_grad_mean, local_grad_cov,
                                          partial_aG * alpha_ * G);
    local_grad_alpha = partial_aG * alpha_ * G;
    cum_alpha *= (1 - alpha_ * G);
    // sh_coeffs += 3 * C * C;
    // grad_sh_coeffs += 3 * C * C;
    // float test_local_grad_alpha = warpSum(local_grad_alpha);
    // printf("shit\n");
    unsigned int mask = __activemask();
    unsigned int leader_id = __ffs(mask) - 1;
    local_grad_mean[0] = warpSum(local_grad_mean[0], mask);
    local_grad_mean[1] = warpSum(local_grad_mean[1], mask);
    local_grad_cov[0] = warpSum(local_grad_cov[0], mask);
    local_grad_cov[1] = warpSum(local_grad_cov[1], mask);
    local_grad_cov[2] = warpSum(local_grad_cov[2], mask);
    local_grad_cov[3] = warpSum(local_grad_cov[3], mask);
#pragma unroll
    for (int i = 0; i < 3 * C * C; ++i) {
      local_grad_sh_coeffs[i] = warpSum(local_grad_sh_coeffs[i], mask);
    }
    if (local_id % 32 == leader_id) {
      // perform once inside a warp
      atomicAdd(grad_alpha + i, local_grad_alpha);
      atomicAdd(grad_mean + 2 * i, local_grad_mean[0]);
      atomicAdd(grad_mean + 2 * i + 1, local_grad_mean[1]);
      atomicAdd(grad_cov + 4 * i, local_grad_cov[0]);
      atomicAdd(grad_cov + 4 * i + 1, local_grad_cov[1]);
      atomicAdd(grad_cov + 4 * i + 2, local_grad_cov[2]);
      atomicAdd(grad_cov + 4 * i + 3, local_grad_cov[3]);
#pragma unroll
      for (int j = 0; j < 3 * C * C; ++j) {
        atomicAdd(grad_sh_coeffs + 3 * C * C * i + j, local_grad_sh_coeffs[j]);
      }
    }
  }
}

template <uint32_t C>
__global__ void tile_based_vol_rendering_backward_sh_entry_warp_reduce(
    uint32_t N, uint32_t N_with_dub, float *__restrict__ mean,
    float *__restrict__ cov, float *__restrict__ sh_coeffs,
    float *__restrict__ alpha, int *__restrict__ start, int *__restrict__ end,
    int *__restrict__ gaussian_ids, float *__restrict__ out_rgb,
    float *__restrict__ grad_mean, float *__restrict__ grad_cov,
    float *__restrict__ grad_sh_coeffs, float *__restrict__ grad_alpha,
    float *__restrict__ grad_out_rgb, float *__restrict__ topleft,
    float *__restrict__ c2w, uint32_t tile_size, uint32_t n_tiles_h,
    uint32_t n_tiles_w, float pixel_size_x, float pixel_size_y, uint32_t H,
    uint32_t W, float thresh) {
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
    vol_render_one_batch_sh_backward_warp_reduce<C>(
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
  if (out_rgb[3 * (global_y * W + global_x) + 0] != out[0]) {
    printf("out_rgb[3 * (global_y * W + global_x) + 0] = %f, out[0] = %f\n",
           out_rgb[3 * (global_y * W + global_x) + 0], out[0]);
  }
  assert(out_rgb[3 * (global_y * W + global_x) + 0] == out[0]);
  assert(out_rgb[3 * (global_y * W + global_x) + 1] == out[1]);
  assert(out_rgb[3 * (global_y * W + global_x) + 2] == out[2]);
}

template <uint32_t C>
void tile_based_vol_rendering_backward_sh_cuda_warp_reduce(
    uint32_t N, uint32_t N_with_dub, float *__restrict__ mean,
    float *__restrict__ cov, float *__restrict__ sh_coeffs,
    float *__restrict__ alpha, int *__restrict__ start, int *__restrict__ end,
    int *__restrict__ gaussian_ids, float *__restrict__ out_rgb,
    float *__restrict__ grad_mean, float *__restrict__ grad_cov,
    float *__restrict__ grad_sh_coeffs, float *__restrict__ grad_alpha,
    float *__restrict__ grad_out_rgb, float *__restrict__ topleft,
    float *__restrict__ c2w, uint32_t tile_size, uint32_t n_tiles_h,
    uint32_t n_tiles_w, float pixel_size_x, float pixel_size_y, uint32_t H,
    uint32_t W, float thresh) {
  const dim3 block(n_tiles_w, n_tiles_h, 1);
  const int n_pixel_per_tile = tile_size * tile_size;
  tile_based_vol_rendering_backward_sh_entry_warp_reduce<C>
      <<<block, n_pixel_per_tile>>>(
          N, N_with_dub, mean, cov, sh_coeffs, alpha, start, end, gaussian_ids,
          out_rgb, grad_mean, grad_cov, grad_sh_coeffs, grad_alpha,
          grad_out_rgb, topleft, c2w, tile_size, n_tiles_h, n_tiles_w,
          pixel_size_x, pixel_size_y, H, W, thresh);

  cudaError_t last_error;
  checkLastCudaError(last_error);
}