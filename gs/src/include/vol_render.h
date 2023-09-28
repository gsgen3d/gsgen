#pragma once
#include "common.h"
#include "data_spec.h"
#include "kernels.h"
#include <cub/cub.cuh>
#include <device_atomic_functions.h>
#include <device_launch_parameters.h>
#include <stdint.h>

__device__ inline void carry(uint32_t N, uint32_t dsize, float *sm, float *gm,
                             int *offset, int *gaussian_ids) {
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

__device__ __forceinline__ void carry_back(uint32_t N, uint32_t dsize,
                                           float *sm, float *gm, int *offset,
                                           int *gaussian_ids) {
  /*carry from scratchpad memory to global memory with (N * dsize) float32s */
  int local_id = threadIdx.x;
  int n_turns = (dsize * N) / blockDim.x;
  int n_left = (dsize * N) % blockDim.x;
  for (int i = 0; i < n_turns; i++) {
    atomicAdd(gm + (dsize * gaussian_ids[(local_id + i * blockDim.x) / dsize] +
                    ((local_id + i * blockDim.x) % dsize)),
              sm[local_id + i * blockDim.x]);
  }
  if (local_id < n_left) {
    atomicAdd(
        gm + (dsize * gaussian_ids[(local_id + n_turns * blockDim.x) / dsize] +
              ((local_id + n_turns * blockDim.x) % dsize)),
        sm[local_id + n_turns * blockDim.x]);
  }
}

__device__ __forceinline__ void direct_carry(uint32_t N, uint32_t dsize,
                                             int *sm, int *gm) {
  int local_id = threadIdx.x;
  int n_turns = (dsize * N) / blockDim.x;
  int n_left = (dsize * N) % blockDim.x;
  for (int i = 0; i < n_turns; i++) {
    sm[local_id + i * blockDim.x] = gm[local_id + i * blockDim.x];
  }
  if (local_id < n_left) {
    sm[local_id + n_turns * blockDim.x] = gm[local_id + n_turns * blockDim.x];
  }
}

__device__ __forceinline__ void carry_tile3(uint32_t N, float *sm, float *gm,
                                            const uint32_t tile_size,
                                            const uint32_t W) {
  int local_id = threadIdx.x;
  int local_y = local_id / tile_size;
  int local_x = local_id % tile_size;
  int global_x = blockIdx.x * blockDim.x + local_x;
  int global_y = blockIdx.y * blockDim.y + local_y;
#pragma unroll
  for (int i = 0; i < 3; ++i) {
    sm[3 * local_id + i] = gm[3 * (global_y * W + global_x) + i];
  }
}

__device__ __forceinline__ void set_zero(uint32_t N, float *sm) {
  int local_id = threadIdx.x;
  int n_turns = N / blockDim.x;
  int n_left = N % blockDim.x;
  for (int i = 0; i < n_turns; i++) {
    sm[local_id + i * blockDim.x] = 0;
  }
  if (local_id < n_left) {
    sm[local_id + n_turns * blockDim.x] = 0;
  }
}

__device__ void
vol_render_one_batch(uint32_t N_gaussians_this_time, float *mean, float *cov,
                     float *color, float *alpha, float *out, float *cum_alpha,
                     float *topleft, uint32_t tile_size, uint32_t n_tiles_h,
                     uint32_t n_tiles_w, float pixel_size_x, float pixel_size_y,
                     uint32_t H, uint32_t W, float thresh, bool first) {
  int local_id = threadIdx.x;
  // check row major here
  int local_y = local_id / tile_size;
  int local_x = local_id % tile_size;
  int global_y = blockIdx.y * tile_size + local_y;
  int global_x = blockIdx.x * tile_size + local_x;
  if (global_y >= H || global_x >= W) {
    return;
  }
  float color_this_time[3];
  float cum_alpha_this_time;

  float pos[2] = {topleft[0] + global_x * pixel_size_x,
                  topleft[1] + global_y * pixel_size_y};

  if (first) {
#pragma unroll
    for (int i = 0; i < 3; ++i) {
      color_this_time[i] = 0.0f;
    }
    cum_alpha_this_time = 1.0f;
  } else {
#pragma unroll
    for (int i = 0; i < 3; ++i) {
      // check bank conflict here
      color_this_time[i] = out[3 * local_id + i];
    }
    cum_alpha_this_time = cum_alpha[local_id];
  }

  // for (int i = 0; i < 1; ++i) {
  for (int i = 0; i < N_gaussians_this_time; ++i) {
    if (cum_alpha_this_time < thresh) {
      break;
    }
    float alpha_ = fminf(alpha[i], 0.99f);
    float coeff = alpha_ * cum_alpha_this_time;
    float val = kernel_gaussian_2d(mean + 2 * i, cov + 4 * i, pos);
    coeff *= val;
    if (alpha_ * val < MIN_RENDER_ALPHA) {
      continue;
    }
    // if (N_gaussians_this_time > 1 && threadIdx.x == 0) {
    //   if (color[3 * i + 2] > 0.0f) {
    //     printf("mean %f %f %f %f\n", mean[0], mean[1], mean[2], mean[3]);
    //     printf("cov %f %f %f %f %f %f %f %f\n", cov[0], cov[1], cov[2],
    //     cov[3],
    //            cov[4], cov[5], cov[6], cov[7]);
    //   }
    // }
    // color_this_time[0] += val * alpha[i];
    // color_this_time[1] += val * alpha[i];
    // color_this_time[2] += val * alpha[i];
    // if (color_this_time[0] > 1.0f) {
    //   printf("N gaussians: %d\n", N_gaussians_this_time);
    //   printf("before: %f after: %f\n", color_this_time[0] - val * alpha[i],
    //          color_this_time[0]);
    //   printf("what the fuck %f\n", color_this_time[0]);
    // }
    // no bank conflicts here
    color_this_time[0] += color[3 * i + 0] * coeff;
    color_this_time[1] += color[3 * i + 1] * coeff;
    color_this_time[2] += color[3 * i + 2] * coeff;
    // checkValue(color[3 * i + 0]);
    // checkValue(color[3 * i + 1]);
    // checkValue(color[3 * i + 2]);
    cum_alpha_this_time *= (1 - alpha_ * val);
  }

#pragma unroll
  for (int i = 0; i < 3; ++i) {
    out[3 * local_id + i] = color_this_time[i];
  }
  cum_alpha[local_id] = cum_alpha_this_time;
}

__device__ void vol_render_one_batch_v1(uint32_t N_gaussians_this_time,
                                        float *mean, float *cov, float *color,
                                        float *alpha, float *out,
                                        float &cum_alpha, float *topleft,
                                        uint32_t tile_size, uint32_t n_tiles_h,
                                        uint32_t n_tiles_w, float pixel_size_x,
                                        float pixel_size_y, uint32_t H,
                                        uint32_t W, float thresh, bool first) {
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

  float pos[2] = {topleft[0] + global_x * pixel_size_x,
                  topleft[1] + global_y * pixel_size_y};

  //   if (first) {
  // #pragma unroll
  //     for (int i = 0; i < 3; ++i) {
  //       color_this_time[i] = 0.0f;
  //     }
  //     cum_alpha_this_time = 1.0f;
  //   } else {
  // #pragma unroll
  //     for (int i = 0; i < 3; ++i) {
  //       // check bank conflict here
  //       color_this_time[i] = out[3 * local_id + i];
  //     }
  //     cum_alpha_this_time = cum_alpha[local_id];
  //   }

  // for (int i = 0; i < 1; ++i) {
  for (int i = 0; i < N_gaussians_this_time; ++i) {
    if (cum_alpha < thresh) {
      break;
    }
    float alpha_ = fminf(alpha[i], 0.99f);
    float coeff = alpha_ * cum_alpha;
    float val = kernel_gaussian_2d(mean + 2 * i, cov + 4 * i, pos);
    coeff *= val;
    if (isnan(coeff)) {
      coeff = 0.0f;
    }
    if (alpha_ * val < MIN_RENDER_ALPHA) {
      continue;
    }
    // if (N_gaussians_this_time > 1 && threadIdx.x == 0) {
    //   if (color[3 * i + 2] > 0.0f) {
    //     printf("mean %f %f %f %f\n", mean[0], mean[1], mean[2], mean[3]);
    //     printf("cov %f %f %f %f %f %f %f %f\n", cov[0], cov[1], cov[2],
    //     cov[3],
    //            cov[4], cov[5], cov[6], cov[7]);
    //   }
    // }
    // color_this_time[0] += val * alpha[i];
    // color_this_time[1] += val * alpha[i];
    // color_this_time[2] += val * alpha[i];
    // if (color_this_time[0] > 1.0f) {
    //   printf("N gaussians: %d\n", N_gaussians_this_time);
    //   printf("before: %f after: %f\n", color_this_time[0] - val * alpha[i],
    //          color_this_time[0]);
    //   printf("what the fuck %f\n", color_this_time[0]);
    // }
    out[0] += color[3 * i + 0] * coeff;
    out[1] += color[3 * i + 1] * coeff;
    out[2] += color[3 * i + 2] * coeff;
    checkValue(color[3 * i + 0]);
    checkValue(color[3 * i + 1]);
    checkValue(color[3 * i + 2]);
    if (isnan(out[0])) {
      out[0] = 0.0f;
    }
    if (isnan(out[1])) {
      out[1] = 0.0f;
    }
    if (isnan(out[2])) {
      out[2] = 0.0f;
    }
    cum_alpha *= (1 - alpha_ * val);
    if (isnan(cum_alpha) || cum_alpha < 0.0 || cum_alpha > 1.0) {
      cum_alpha = 0.0;
    }
  }

  // #pragma unroll
  //   for (int i = 0; i < 3; ++i) {
  //     out[3 * local_id + i] = color_this_time[i];
  //   }
  //   cum_alpha[local_id] = cum_alpha_this_time;
}

__device__ void vol_render_one_batch_with_T(
    uint32_t N_gaussians_this_time, float *mean, float *cov, float *color,
    float *alpha, float *out, float &cum_alpha, float *topleft,
    uint32_t tile_size, uint32_t n_tiles_h, uint32_t n_tiles_w,
    float pixel_size_x, float pixel_size_y, uint32_t H, uint32_t W,
    float thresh, float *T) {
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
  // for (int i = 0; i < 1; ++i) {
  for (int i = 0; i < N_gaussians_this_time; ++i) {
    if (cum_alpha < thresh) {
      break;
    }
    float alpha_ = fminf(alpha[i], 0.99f);
    float coeff = alpha_ * cum_alpha;
    float val = kernel_gaussian_2d(mean + 2 * i, cov + 4 * i, pos);
    coeff *= val;
    if (alpha_ * val < MIN_RENDER_ALPHA) {
      continue;
    }
    if (isnan(coeff)) {
      coeff = 0.0f;
    }
    out[0] += color[3 * i + 0] * coeff;
    out[1] += color[3 * i + 1] * coeff;
    out[2] += color[3 * i + 2] * coeff;
    checkValue(color[3 * i + 0]);
    checkValue(color[3 * i + 1]);
    checkValue(color[3 * i + 2]);
    if (isnan(out[0])) {
      out[0] = 0.0f;
    }
    if (isnan(out[1])) {
      out[1] = 0.0f;
    }
    if (isnan(out[2])) {
      out[2] = 0.0f;
    }
    cum_alpha *= (1 - alpha_ * val);
  }
}

__device__ void vol_render_one_batch_backward(
    uint32_t N_gaussians_this_time, float *mean, float *cov, float *color,
    float *alpha, float *grad_mean, float *grad_cov, float *grad_color,
    float *grad_alpha, float *grad_out, float *final, float *out,
    float *cum_alpha, float *topleft, uint32_t tile_size, uint32_t n_tiles_h,
    uint32_t n_tiles_w, float pixel_size_x, float pixel_size_y, uint32_t H,
    uint32_t W, float thresh, bool first) {
  int local_id = threadIdx.x;
  // check row major here
  int local_y = local_id / tile_size;
  int local_x = local_id % tile_size;
  int global_y = blockIdx.y * tile_size + local_y;
  int global_x = blockIdx.x * tile_size + local_x;
  if (global_y >= H || global_x >= W) {
    return;
  }
  float color_this_time[3];
  float cum_alpha_this_time;

  float pos[2] = {topleft[0] + global_x * pixel_size_x,
                  topleft[1] + global_y * pixel_size_y};

  if (first) {
#pragma unroll
    for (int i = 0; i < 3; ++i) {
      color_this_time[i] = 0.0f;
    }
    cum_alpha_this_time = 1.0f;
  } else {
#pragma unroll
    for (int i = 0; i < 3; ++i) {
      // check bank conflict here
      color_this_time[i] = out[3 * local_id + i];
    }
    cum_alpha_this_time = cum_alpha[local_id];
  }

  float grad_out_this_pixel[3];
#pragma unroll
  for (int i = 0; i < 3; ++i) {
    grad_out_this_pixel[i] = grad_out[3 * local_id + i];
  }

  for (int i = 0; i < N_gaussians_this_time; ++i) {
    if (cum_alpha_this_time < thresh) {
      break;
    }
    float alpha_ = fminf(alpha[i], 0.99f);
    float G = kernel_gaussian_2d(mean + 2 * i, cov + 4 * i, pos); // a * G
    assert(G >= 0.0f && G <= 1.0f);
    if (alpha_ * G < MIN_RENDER_ALPHA) {
      continue;
    }
    float coeff = alpha_ * cum_alpha_this_time * G; // a * T * G
    float color_this_gaussian[3];
    color_this_gaussian[0] = color[3 * i + 0] * coeff; // C * alpha * T * G
    color_this_gaussian[1] = color[3 * i + 1] * coeff;
    color_this_gaussian[2] = color[3 * i + 2] * coeff;
    color_this_time[0] += color_this_gaussian[0]; // \sum alpha * T * G * C
    color_this_time[1] += color_this_gaussian[1];
    color_this_time[2] += color_this_gaussian[2];
    atomicAdd(grad_color + 3 * i, coeff * grad_out_this_pixel[0]);
    atomicAdd(grad_color + 3 * i + 1, coeff * grad_out_this_pixel[1]);
    atomicAdd(grad_color + 3 * i + 2, coeff * grad_out_this_pixel[2]);

    double partial_aG = 0.0;
    // assert(1 - alpha_ * G >= 0.01f);
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      checkValue(color[3 * i + j] * cum_alpha_this_time);
      if (isnan(color[3 * i + j] * cum_alpha_this_time) ||
          isinf(color[3 * i + j] * cum_alpha_this_time)) {
        printf("color: %f, cum_alpha: %f\n", color[3 * i + j],
               cum_alpha_this_time);
      }
      // checkValue((final[3 * local_id + j] - color_this_time[j]) /
      //            (1 - alpha_ * G));
      partial_aG +=
          (color[3 * i + j] * cum_alpha_this_time -
           (final[3 * local_id + j] - color_this_time[j]) / (1 - alpha_ * G)) *
          grad_out_this_pixel[j];
    }
    // checkValue(partial_aG);
    // checkValue(partial_aG * alpha_ * G);
    // checkValue(partial_aG * alpha_);
    // kernel_gaussian_2d_backward(mean + 2 * i, cov + 4 * i, pos,
    //                             grad_mean + 2 * i, grad_cov + 4 * i,
    //                             partial_aG * alpha_ * G);
    kernel_gaussian_2d_backward(mean + 2 * i, cov + 4 * i, pos,
                                grad_mean + 2 * i, grad_cov + 4 * i,
                                partial_aG * alpha_ * G);
    atomicAdd(grad_alpha + i, partial_aG * G);
    cum_alpha_this_time *= (1 - alpha_ * G);
  }

#pragma unroll
  for (int i = 0; i < 3; ++i) {
    out[3 * local_id + i] = color_this_time[i];
  }
  cum_alpha[local_id] = cum_alpha_this_time;
}

__global__ void tile_based_vol_rendering_entry(
    uint32_t N, uint32_t N_with_dub, float *mean, float *cov, float *color,
    float *alpha, int *offset, int *gaussian_ids, float *out_rgb,
    float *topleft, uint32_t tile_size, uint32_t n_tiles_h, uint32_t n_tiles_w,
    float pixel_size_x, float pixel_size_y, uint32_t H, uint32_t W,
    float thresh) {
  int local_id = threadIdx.x;
  int local_y = local_id / tile_size;
  int local_x = local_id % tile_size;
  int tile_id = blockIdx.y * gridDim.x + blockIdx.x;
  int n_gaussians_this_tile = offset[tile_id + 1] - offset[tile_id];
  if (n_gaussians_this_tile == 0) {
    return;
  }

  // compute memory need for this tile
  int n_float_per_gaussian = 2 + 4 + 3 + 1;
  // mean + cov + color + alpha
  int n_pixel_per_tile = tile_size * tile_size;
  int n_float_per_pixel = 3 + 1;
  // output rgb + cum_alpha
  int max_gaussian_sm =
      (MAX_N_FLOAT_SM - n_float_per_pixel * n_pixel_per_tile) /
      n_float_per_gaussian;
  __shared__ float sm[MAX_N_FLOAT_SM];
  float *sm_mean = sm;
  float *sm_cov = sm_mean + 2 * max_gaussian_sm;
  float *sm_color = sm_cov + 4 * max_gaussian_sm;
  float *sm_alpha = sm_color + 3 * max_gaussian_sm;
  float *sm_cum_alpha = sm_alpha + 1 * max_gaussian_sm;
  float *sm_out = sm_cum_alpha + 1 * n_pixel_per_tile;

  gaussian_ids += offset[tile_id];

  for (int n = 0; n < n_gaussians_this_tile; n += max_gaussian_sm) {
    int num_gaussian_sm = min(max_gaussian_sm, n_gaussians_this_tile - n);
    carry(num_gaussian_sm, 2, sm_mean, mean, offset, gaussian_ids);
    carry(num_gaussian_sm, 4, sm_cov, cov, offset, gaussian_ids);
    carry(num_gaussian_sm, 3, sm_color, color, offset, gaussian_ids);
    carry(num_gaussian_sm, 1, sm_alpha, alpha, offset, gaussian_ids);
    __syncthreads();
    vol_render_one_batch(num_gaussian_sm, sm_mean, sm_cov, sm_color, sm_alpha,
                         sm_out, sm_cum_alpha, topleft, tile_size, n_tiles_h,
                         n_tiles_w, pixel_size_x, pixel_size_y, H, W, thresh,
                         n == 0);
    __syncthreads();
    gaussian_ids += num_gaussian_sm;
  }
  int global_y = blockIdx.y * tile_size + local_y;
  int global_x = blockIdx.x * tile_size + local_x;
  if (global_y >= H || global_x >= W) {
    return;
  }
  out_rgb[3 * (global_y * W + global_x) + 0] = sm_out[3 * local_id + 0];
  out_rgb[3 * (global_y * W + global_x) + 1] = sm_out[3 * local_id + 1];
  out_rgb[3 * (global_y * W + global_x) + 2] = sm_out[3 * local_id + 2];

  /* test kernel 2d */
}

__global__ void tile_based_vol_rendering_entry_v1(
    uint32_t N, uint32_t N_with_dub, float *mean, float *cov, float *color,
    float *alpha, int *offset, int *gaussian_ids, float *out_rgb,
    float *topleft, uint32_t tile_size, uint32_t n_tiles_h, uint32_t n_tiles_w,
    float pixel_size_x, float pixel_size_y, uint32_t H, uint32_t W,
    float thresh) {
  int local_id = threadIdx.x;
  int local_y = local_id / tile_size;
  int local_x = local_id % tile_size;
  int tile_id = blockIdx.y * gridDim.x + blockIdx.x;
  int n_gaussians_this_tile = offset[tile_id + 1] - offset[tile_id];
  if (n_gaussians_this_tile == 0) {
    return;
  }

  // compute memory need for this tile
  int n_float_per_gaussian = 2 + 4 + 3 + 1;
  // mean + cov + color + alpha
  int n_pixel_per_tile = tile_size * tile_size;
  // int n_float_per_pixel = 3 + 1;
  // output rgb + cum_alpha
  int max_gaussian_sm = (MAX_N_FLOAT_SM) / n_float_per_gaussian;
  __shared__ float sm[MAX_N_FLOAT_SM];
  float *sm_mean = sm;
  float *sm_cov = sm_mean + 2 * max_gaussian_sm;
  float *sm_color = sm_cov + 4 * max_gaussian_sm;
  float *sm_alpha = sm_color + 3 * max_gaussian_sm;
  // float *sm_cum_alpha = sm_alpha + 1 * max_gaussian_sm;
  // float *sm_out = sm_cum_alpha + 1 * n_pixel_per_tile;

  float out[3] = {0.0f, 0.0f, 0.0f};
  float cum_alpha = 1.0f;

  gaussian_ids += offset[tile_id];

  for (int n = 0; n < n_gaussians_this_tile; n += max_gaussian_sm) {
    int num_gaussian_sm = min(max_gaussian_sm, n_gaussians_this_tile - n);
    carry(num_gaussian_sm, 2, sm_mean, mean, offset, gaussian_ids);
    carry(num_gaussian_sm, 4, sm_cov, cov, offset, gaussian_ids);
    carry(num_gaussian_sm, 3, sm_color, color, offset, gaussian_ids);
    carry(num_gaussian_sm, 1, sm_alpha, alpha, offset, gaussian_ids);
    __syncthreads();
    vol_render_one_batch_v1(num_gaussian_sm, sm_mean, sm_cov, sm_color,
                            sm_alpha, out, cum_alpha, topleft, tile_size,
                            n_tiles_h, n_tiles_w, pixel_size_x, pixel_size_y, H,
                            W, thresh, n == 0);
    __syncthreads();
    gaussian_ids += num_gaussian_sm;
  }
  int global_y = blockIdx.y * tile_size + local_y;
  int global_x = blockIdx.x * tile_size + local_x;
  if (global_y >= H || global_x >= W) {
    return;
  }
  out_rgb[3 * (global_y * W + global_x) + 0] = out[0];
  out_rgb[3 * (global_y * W + global_x) + 1] = out[1];
  out_rgb[3 * (global_y * W + global_x) + 2] = out[2];

  /* test kernel 2d */
}

__global__ void tile_based_vol_rendering_entry_v2(
    uint32_t N, uint32_t N_with_dub, float *mean, float *cov, float *color,
    float *alpha, int *offset, int *gaussian_ids, float *out_rgb,
    float *topleft, uint32_t tile_size, uint32_t n_tiles_h, uint32_t n_tiles_w,
    float pixel_size_x, float pixel_size_y, uint32_t H, uint32_t W,
    float thresh) {
  int local_id = threadIdx.x;
  int local_y = local_id / tile_size;
  int local_x = local_id % tile_size;
  int tile_id = blockIdx.y * gridDim.x + blockIdx.x;
  int n_gaussians_this_tile = offset[tile_id + 1] - offset[tile_id];
  if (n_gaussians_this_tile == 0) {
    return;
  }

  // compute memory need for this tile
  int n_float_per_gaussian = 2 + 4 + 3 + 1 + 1;
  // mean + cov + color + alpha + gaussian_ids
  int n_pixel_per_tile = tile_size * tile_size;
  // int n_float_per_pixel = 3 + 1;
  // output rgb + cum_alpha
  int max_gaussian_sm = (MAX_N_FLOAT_SM) / n_float_per_gaussian;
  __shared__ float sm[MAX_N_FLOAT_SM];
  float *sm_mean = sm;
  float *sm_cov = sm_mean + 2 * max_gaussian_sm;
  float *sm_color = sm_cov + 4 * max_gaussian_sm;
  float *sm_alpha = sm_color + 3 * max_gaussian_sm;
  int *sm_gaussian_ids =
      reinterpret_cast<int *>(sm_alpha + 1 * max_gaussian_sm);
  // float *sm_cum_alpha = sm_alpha + 1 * max_gaussian_sm;
  // float *sm_out = sm_cum_alpha + 1 * n_pixel_per_tile;

  float out[3] = {0.0f, 0.0f, 0.0f};
  float cum_alpha = 1.0f;

  gaussian_ids += offset[tile_id];

  for (int n = 0; n < n_gaussians_this_tile; n += max_gaussian_sm) {
    int num_gaussian_sm = min(max_gaussian_sm, n_gaussians_this_tile - n);
    direct_carry(num_gaussian_sm, 1, sm_gaussian_ids, gaussian_ids);
    carry(num_gaussian_sm, 2, sm_mean, mean, offset, sm_gaussian_ids);
    carry(num_gaussian_sm, 4, sm_cov, cov, offset, sm_gaussian_ids);
    carry(num_gaussian_sm, 3, sm_color, color, offset, sm_gaussian_ids);
    carry(num_gaussian_sm, 1, sm_alpha, alpha, offset, sm_gaussian_ids);
    __syncthreads();
    vol_render_one_batch_v1(num_gaussian_sm, sm_mean, sm_cov, sm_color,
                            sm_alpha, out, cum_alpha, topleft, tile_size,
                            n_tiles_h, n_tiles_w, pixel_size_x, pixel_size_y, H,
                            W, thresh, n == 0);
    __syncthreads();
    gaussian_ids += num_gaussian_sm;
  }
  int global_y = blockIdx.y * tile_size + local_y;
  int global_x = blockIdx.x * tile_size + local_x;
  if (global_y >= H || global_x >= W) {
    return;
  }
  out_rgb[3 * (global_y * W + global_x) + 0] = out[0];
  out_rgb[3 * (global_y * W + global_x) + 1] = out[1];
  out_rgb[3 * (global_y * W + global_x) + 2] = out[2];

  /* test kernel 2d */
}

__global__ void tile_based_vol_rendering_backward_entry(
    uint32_t N, uint32_t N_with_dub, float *mean, float *cov, float *color,
    float *alpha, int *offset, int *gaussian_ids, float *out_rgb,
    float *grad_mean, float *grad_cov, float *grad_color, float *grad_alpha,
    float *grad_out, float *topleft, uint32_t tile_size, uint32_t n_tiles_h,
    uint32_t n_tiles_w, float pixel_size_x, float pixel_size_y, uint32_t H,
    uint32_t W, float thresh) {
  int local_id = threadIdx.x;
  int local_y = local_id / tile_size;
  int local_x = local_id % tile_size;
  int tile_id = blockIdx.y * gridDim.x + blockIdx.x;
  int global_y = blockIdx.y * tile_size + local_y;
  int global_x = blockIdx.x * tile_size + local_x;
  int n_gaussians_this_tile = offset[tile_id + 1] - offset[tile_id];
  if (n_gaussians_this_tile == 0) {
    return;
  }

  // compute memory need for this tile
  int n_float_per_gaussian = 2 + 4 + 3 + 1;
  n_float_per_gaussian *= 2; // for backward
  // mean + cov + color + alpha
  int n_pixel_per_tile = tile_size * tile_size;
  int n_float_per_pixel = 3 + 1;
  n_float_per_pixel +=
      (3 + 3); // for backward: 3 for out_rgb, 3 for grad_out_rgb
  // output rgb + cum_alpha
  int max_gaussian_sm =
      (MAX_N_FLOAT_SM - n_float_per_pixel * n_pixel_per_tile) /
      n_float_per_gaussian;
  __shared__ float sm[MAX_N_FLOAT_SM];
  float *sm_mean = sm;
  float *sm_cov = sm_mean + 2 * max_gaussian_sm;
  float *sm_color = sm_cov + 4 * max_gaussian_sm;
  float *sm_alpha = sm_color + 3 * max_gaussian_sm;
  float *sm_cum_alpha = sm_alpha + 1 * max_gaussian_sm;
  float *sm_out = sm_cum_alpha + 1 * n_pixel_per_tile;

  float *sm_grad_mean = sm_out + 3 * n_pixel_per_tile;
  float *sm_grad_cov = sm_grad_mean + 2 * max_gaussian_sm;
  float *sm_grad_color = sm_grad_cov + 4 * max_gaussian_sm;
  float *sm_grad_alpha = sm_grad_color + 3 * max_gaussian_sm;

  float *sm_grad_out = sm_grad_alpha + 1 * max_gaussian_sm;
  float *sm_final = sm_grad_out + 3 * n_pixel_per_tile;

  gaussian_ids += offset[tile_id];

  for (int n = 0; n < n_gaussians_this_tile; n += max_gaussian_sm) {
    int num_gaussian_sm = min(max_gaussian_sm, n_gaussians_this_tile - n);
    carry(num_gaussian_sm, 2, sm_mean, mean, offset, gaussian_ids);
    carry(num_gaussian_sm, 4, sm_cov, cov, offset, gaussian_ids);
    carry(num_gaussian_sm, 3, sm_color, color, offset, gaussian_ids);
    carry(num_gaussian_sm, 1, sm_alpha, alpha, offset, gaussian_ids);
    set_zero(2 * num_gaussian_sm, sm_grad_mean);
    set_zero(4 * num_gaussian_sm, sm_grad_cov);
    set_zero(3 * num_gaussian_sm, sm_grad_color);
    set_zero(1 * num_gaussian_sm, sm_grad_alpha);
    if (global_x < W && global_y < H) {
#pragma unroll
      for (size_t i = 0; i < 3; ++i) {
        sm_grad_out[3 * local_id + i] =
            grad_out[3 * (global_y * W + global_x) + i];
        sm_final[3 * local_id + i] = out_rgb[3 * (global_y * W + global_x) + i];
      }
    }
    __syncthreads();
    vol_render_one_batch_backward(
        num_gaussian_sm, sm_mean, sm_cov, sm_color, sm_alpha, sm_grad_mean,
        sm_grad_cov, sm_grad_color, sm_grad_alpha, sm_grad_out, sm_final,
        sm_out, sm_cum_alpha, topleft, tile_size, n_tiles_h, n_tiles_w,
        pixel_size_x, pixel_size_y, H, W, thresh, n == 0);
    __syncthreads();
    carry_back(num_gaussian_sm, 2, sm_grad_mean, grad_mean, offset,
               gaussian_ids);
    carry_back(num_gaussian_sm, 4, sm_grad_cov, grad_cov, offset, gaussian_ids);
    carry_back(num_gaussian_sm, 3, sm_grad_color, grad_color, offset,
               gaussian_ids);
    carry_back(num_gaussian_sm, 1, sm_grad_alpha, grad_alpha, offset,
               gaussian_ids);
    __syncthreads();
    gaussian_ids += num_gaussian_sm;
  }
  if (global_x >= W || global_y >= H) {
    return;
  }
  if (out_rgb[3 * (global_y * W + global_x) + 0] != sm_out[3 * local_id + 0]) {
    printf("out_rgb[3 * (global_y * W + global_x) + 0] = %f, sm_out[3 * "
           "local_id + 0] = %f\n",
           out_rgb[3 * (global_y * W + global_x) + 0],
           sm_out[3 * local_id + 0]);
  }
  assert(out_rgb[3 * (global_y * W + global_x) + 0] ==
         sm_out[3 * local_id + 0]);
  assert(out_rgb[3 * (global_y * W + global_x) + 1] ==
         sm_out[3 * local_id + 1]);
  assert(out_rgb[3 * (global_y * W + global_x) + 2] ==
         sm_out[3 * local_id + 2]);
}

// TODO: add backward compatibility
// TODO: add spherical harmonic support
void tile_based_vol_rendering_cuda(uint32_t N, uint32_t N_with_dub, float *mean,
                                   float *cov, float *color, float *alpha,
                                   int *offset, int *gaussian_ids,
                                   float *out_rgb, float *topleft,
                                   uint32_t tile_size, uint32_t n_tiles_h,
                                   uint32_t n_tiles_w, float pixel_size_x,
                                   float pixel_size_y, uint32_t H, uint32_t W,
                                   float thresh) {
  const dim3 block(n_tiles_w, n_tiles_h, 1);
  const int n_pixel_per_tile = tile_size * tile_size;
  tile_based_vol_rendering_entry<<<block, n_pixel_per_tile>>>(
      N, N_with_dub, mean, cov, color, alpha, offset, gaussian_ids, out_rgb,
      topleft, tile_size, n_tiles_h, n_tiles_w, pixel_size_x, pixel_size_y, H,
      W, thresh);

  cudaError_t last_error;
  checkLastCudaError(last_error);
}

void tile_based_vol_rendering_cuda_v1(uint32_t N, uint32_t N_with_dub,
                                      float *mean, float *cov, float *color,
                                      float *alpha, int *offset,
                                      int *gaussian_ids, float *out_rgb,
                                      float *topleft, uint32_t tile_size,
                                      uint32_t n_tiles_h, uint32_t n_tiles_w,
                                      float pixel_size_x, float pixel_size_y,
                                      uint32_t H, uint32_t W, float thresh) {
  const dim3 block(n_tiles_w, n_tiles_h, 1);
  const int n_pixel_per_tile = tile_size * tile_size;
  tile_based_vol_rendering_entry_v1<<<block, n_pixel_per_tile>>>(
      N, N_with_dub, mean, cov, color, alpha, offset, gaussian_ids, out_rgb,
      topleft, tile_size, n_tiles_h, n_tiles_w, pixel_size_x, pixel_size_y, H,
      W, thresh);

  cudaError_t last_error;
  checkLastCudaError(last_error);
}

void tile_based_vol_rendering_cuda_v2(uint32_t N, uint32_t N_with_dub,
                                      float *mean, float *cov, float *color,
                                      float *alpha, int *offset,
                                      int *gaussian_ids, float *out_rgb,
                                      float *topleft, uint32_t tile_size,
                                      uint32_t n_tiles_h, uint32_t n_tiles_w,
                                      float pixel_size_x, float pixel_size_y,
                                      uint32_t H, uint32_t W, float thresh) {
  const dim3 block(n_tiles_w, n_tiles_h, 1);
  const int n_pixel_per_tile = tile_size * tile_size;
  tile_based_vol_rendering_entry_v2<<<block, n_pixel_per_tile>>>(
      N, N_with_dub, mean, cov, color, alpha, offset, gaussian_ids, out_rgb,
      topleft, tile_size, n_tiles_h, n_tiles_w, pixel_size_x, pixel_size_y, H,
      W, thresh);

  cudaError_t last_error;
  checkLastCudaError(last_error);
}

void tile_based_vol_rendering_backward_cuda(
    uint32_t N, uint32_t N_with_dub, float *mean, float *cov, float *color,
    float *alpha, int *offset, int *gaussian_ids, float *out_rgb,
    float *grad_mean, float *grad_cov, float *grad_color, float *grad_alpha,
    float *grad_out, float *topleft, uint32_t tile_size, uint32_t n_tiles_h,
    uint32_t n_tiles_w, float pixel_size_x, float pixel_size_y, uint32_t H,
    uint32_t W, float thresh) {
  // TODO: delete final and out
  const dim3 block(n_tiles_w, n_tiles_h, 1);
  const int n_pixel_per_tile = tile_size * tile_size;
  tile_based_vol_rendering_backward_entry<<<block, n_pixel_per_tile>>>(
      N, N_with_dub, mean, cov, color, alpha, offset, gaussian_ids, out_rgb,
      grad_mean, grad_cov, grad_color, grad_alpha, grad_out, topleft, tile_size,
      n_tiles_h, n_tiles_w, pixel_size_x, pixel_size_y, H, W, thresh);
  cudaError_t last_error;
  checkLastCudaError(last_error);
}

__global__ void tile_based_vol_rendering_entry_start_end(
    uint32_t N, uint32_t N_with_dub, float *__restrict__ mean,
    float *__restrict__ cov, float *__restrict__ color,
    float *__restrict__ alpha, int *__restrict__ start, int *__restrict__ end,
    int *__restrict__ gaussian_ids, float *__restrict__ out_rgb,
    float *__restrict__ topleft, uint32_t tile_size, uint32_t n_tiles_h,
    uint32_t n_tiles_w, float pixel_size_x, float pixel_size_y, uint32_t H,
    uint32_t W, float thresh) {
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
  int n_float_per_gaussian = 2 + 4 + 3 + 1;
  // mean + cov + color + alpha
  int n_pixel_per_tile = tile_size * tile_size;
  // int n_float_per_pixel = 3 + 1;
  // output rgb + cum_alpha
  int max_gaussian_sm = (MAX_N_FLOAT_SM) / n_float_per_gaussian;
  __shared__ float sm[MAX_N_FLOAT_SM];
  float *sm_mean = sm;
  float *sm_cov = sm_mean + 2 * max_gaussian_sm;
  float *sm_color = sm_cov + 4 * max_gaussian_sm;
  float *sm_alpha = sm_color + 3 * max_gaussian_sm;
  // float *sm_cum_alpha = sm_alpha + 1 * max_gaussian_sm;
  // float *sm_out = sm_cum_alpha + 1 * n_pixel_per_tile;

  float out[3] = {0.0f, 0.0f, 0.0f};
  float cum_alpha = 1.0f;

  gaussian_ids += start[tile_id];

  for (int n = 0; n < n_gaussians_this_tile; n += max_gaussian_sm) {
    int num_gaussian_sm = min(max_gaussian_sm, n_gaussians_this_tile - n);
    carry(num_gaussian_sm, 2, sm_mean, mean, NULL, gaussian_ids);
    carry(num_gaussian_sm, 4, sm_cov, cov, NULL, gaussian_ids);
    carry(num_gaussian_sm, 3, sm_color, color, NULL, gaussian_ids);
    carry(num_gaussian_sm, 1, sm_alpha, alpha, NULL, gaussian_ids);
    __syncthreads();
    vol_render_one_batch_v1(num_gaussian_sm, sm_mean, sm_cov, sm_color,
                            sm_alpha, out, cum_alpha, topleft, tile_size,
                            n_tiles_h, n_tiles_w, pixel_size_x, pixel_size_y, H,
                            W, thresh, n == 0);
    __syncthreads();
    gaussian_ids += num_gaussian_sm;
  }
  int global_y = blockIdx.y * tile_size + local_y;
  int global_x = blockIdx.x * tile_size + local_x;
  if (global_y >= H || global_x >= W) {
    return;
  }
  out_rgb[3 * (global_y * W + global_x) + 0] = out[0];
  out_rgb[3 * (global_y * W + global_x) + 1] = out[1];
  out_rgb[3 * (global_y * W + global_x) + 2] = out[2];

  /* test kernel 2d */
}

void tile_based_vol_rendering_start_end_cuda(
    uint32_t N, uint32_t N_with_dub, float *mean, float *cov, float *color,
    float *alpha, int *start, int *end, int *gaussian_ids, float *out_rgb,
    float *topleft, uint32_t tile_size, uint32_t n_tiles_h, uint32_t n_tiles_w,
    float pixel_size_x, float pixel_size_y, uint32_t H, uint32_t W,
    float thresh) {
  const dim3 block(n_tiles_w, n_tiles_h, 1);
  const int n_pixel_per_tile = tile_size * tile_size;
  tile_based_vol_rendering_entry_start_end<<<block, n_pixel_per_tile>>>(
      N, N_with_dub, mean, cov, color, alpha, start, end, gaussian_ids, out_rgb,
      topleft, tile_size, n_tiles_h, n_tiles_w, pixel_size_x, pixel_size_y, H,
      W, thresh);

  cudaError_t last_error;
  checkLastCudaError(last_error);
}

__global__ void tile_based_vol_rendering_backward_entry_start_end(
    uint32_t N, uint32_t N_with_dub, float *__restrict__ mean,
    float *__restrict__ cov, float *__restrict__ color,
    float *__restrict__ alpha, int *__restrict__ start, int *__restrict__ end,
    int *__restrict__ gaussian_ids, float *__restrict__ out_rgb,
    float *__restrict__ grad_mean, float *__restrict__ grad_cov,
    float *__restrict__ grad_color, float *__restrict__ grad_alpha,
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
  int n_float_per_gaussian = 2 + 4 + 3 + 1;
  n_float_per_gaussian *= 2; // for backward
  // mean + cov + color + alpha
  int n_pixel_per_tile = tile_size * tile_size;
  int n_float_per_pixel = 3 + 1;
  n_float_per_pixel +=
      (3 + 3); // for backward: 3 for out_rgb, 3 for grad_out_rgb
  // output rgb + cum_alpha
  int max_gaussian_sm =
      (MAX_N_FLOAT_SM - n_float_per_pixel * n_pixel_per_tile) /
      n_float_per_gaussian;
  __shared__ float sm[MAX_N_FLOAT_SM];
  float *sm_mean = sm;
  float *sm_cov = sm_mean + 2 * max_gaussian_sm;
  float *sm_color = sm_cov + 4 * max_gaussian_sm;
  float *sm_alpha = sm_color + 3 * max_gaussian_sm;
  float *sm_cum_alpha = sm_alpha + 1 * max_gaussian_sm;
  float *sm_out = sm_cum_alpha + 1 * n_pixel_per_tile;

  float *sm_grad_mean = sm_out + 3 * n_pixel_per_tile;
  float *sm_grad_cov = sm_grad_mean + 2 * max_gaussian_sm;
  float *sm_grad_color = sm_grad_cov + 4 * max_gaussian_sm;
  float *sm_grad_alpha = sm_grad_color + 3 * max_gaussian_sm;

  float *sm_grad_out = sm_grad_alpha + 1 * max_gaussian_sm;
  float *sm_final = sm_grad_out + 3 * n_pixel_per_tile;

  gaussian_ids += start[tile_id];

  for (int n = 0; n < n_gaussians_this_tile; n += max_gaussian_sm) {
    int num_gaussian_sm = min(max_gaussian_sm, n_gaussians_this_tile - n);
    carry(num_gaussian_sm, 2, sm_mean, mean, NULL, gaussian_ids);
    carry(num_gaussian_sm, 4, sm_cov, cov, NULL, gaussian_ids);
    carry(num_gaussian_sm, 3, sm_color, color, NULL, gaussian_ids);
    carry(num_gaussian_sm, 1, sm_alpha, alpha, NULL, gaussian_ids);
    set_zero(2 * num_gaussian_sm, sm_grad_mean);
    set_zero(4 * num_gaussian_sm, sm_grad_cov);
    set_zero(3 * num_gaussian_sm, sm_grad_color);
    set_zero(1 * num_gaussian_sm, sm_grad_alpha);
    if (global_x < W && global_y < H) {
#pragma unroll
      for (size_t i = 0; i < 3; ++i) {
        sm_grad_out[3 * local_id + i] =
            grad_out[3 * (global_y * W + global_x) + i];
        sm_final[3 * local_id + i] = out_rgb[3 * (global_y * W + global_x) + i];
      }
    }
    __syncthreads();
    vol_render_one_batch_backward(
        num_gaussian_sm, sm_mean, sm_cov, sm_color, sm_alpha, sm_grad_mean,
        sm_grad_cov, sm_grad_color, sm_grad_alpha, sm_grad_out, sm_final,
        sm_out, sm_cum_alpha, topleft, tile_size, n_tiles_h, n_tiles_w,
        pixel_size_x, pixel_size_y, H, W, thresh, n == 0);
    __syncthreads();
    carry_back(num_gaussian_sm, 2, sm_grad_mean, grad_mean, NULL, gaussian_ids);
    carry_back(num_gaussian_sm, 4, sm_grad_cov, grad_cov, NULL, gaussian_ids);
    carry_back(num_gaussian_sm, 3, sm_grad_color, grad_color, NULL,
               gaussian_ids);
    carry_back(num_gaussian_sm, 1, sm_grad_alpha, grad_alpha, NULL,
               gaussian_ids);
    __syncthreads();
    gaussian_ids += num_gaussian_sm;
  }
  if (global_x >= W || global_y >= H) {
    return;
  }
  // temporary disable this check due to we need to render the background
  // outside the cpp code. if (out_rgb[3 * (global_y * W + global_x) + 0] !=
  // sm_out[3 * local_id + 0]) {
  //   printf("out_rgb[3 * (global_y * W + global_x) + 0] = %f, sm_out[3 * "
  //          "local_id + 0] = %f\n",
  //          out_rgb[3 * (global_y * W + global_x) + 0],
  //          sm_out[3 * local_id + 0]);
  // }
  // assert(out_rgb[3 * (global_y * W + global_x) + 0] ==
  //        sm_out[3 * local_id + 0]);
  // assert(out_rgb[3 * (global_y * W + global_x) + 1] ==
  //        sm_out[3 * local_id + 1]);
  // assert(out_rgb[3 * (global_y * W + global_x) + 2] ==
  //        sm_out[3 * local_id + 2]);
}

void tile_based_vol_rendering_backward_start_end_cuda(
    uint32_t N, uint32_t N_with_dub, float *mean, float *cov, float *color,
    float *alpha, int *start, int *end, int *gaussian_ids, float *out_rgb,
    float *grad_mean, float *grad_cov, float *grad_color, float *grad_alpha,
    float *grad_out, float *topleft, uint32_t tile_size, uint32_t n_tiles_h,
    uint32_t n_tiles_w, float pixel_size_x, float pixel_size_y, uint32_t H,
    uint32_t W, float thresh) {
  // TODO: delete final and out
  const dim3 block(n_tiles_w, n_tiles_h, 1);
  const int n_pixel_per_tile = tile_size * tile_size;
  tile_based_vol_rendering_backward_entry_start_end<<<block,
                                                      n_pixel_per_tile>>>(
      N, N_with_dub, mean, cov, color, alpha, start, end, gaussian_ids, out_rgb,
      grad_mean, grad_cov, grad_color, grad_alpha, grad_out, topleft, tile_size,
      n_tiles_h, n_tiles_w, pixel_size_x, pixel_size_y, H, W, thresh);
  cudaError_t last_error;
  checkLastCudaError(last_error);
}

__global__ void tile_based_vol_rendering_with_T(
    uint32_t N, uint32_t N_with_dub, float *__restrict__ mean,
    float *__restrict__ cov, float *__restrict__ color,
    float *__restrict__ alpha, int *__restrict__ start, int *__restrict__ end,
    int *__restrict__ gaussian_ids, float *__restrict__ out_rgb,
    float *__restrict__ topleft, uint32_t tile_size, uint32_t n_tiles_h,
    uint32_t n_tiles_w, float pixel_size_x, float pixel_size_y, uint32_t H,
    uint32_t W, float thresh, float *T) {
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
  int n_float_per_gaussian = 2 + 4 + 3 + 1;
  // mean + cov + color + alpha
  int n_pixel_per_tile = tile_size * tile_size;
  // int n_float_per_pixel = 3 + 1;
  // output rgb + cum_alpha
  int max_gaussian_sm = (MAX_N_FLOAT_SM) / n_float_per_gaussian;
  __shared__ float sm[MAX_N_FLOAT_SM];
  float *sm_mean = sm;
  float *sm_cov = sm_mean + 2 * max_gaussian_sm;
  float *sm_color = sm_cov + 4 * max_gaussian_sm;
  float *sm_alpha = sm_color + 3 * max_gaussian_sm;
  // float *sm_cum_alpha = sm_alpha + 1 * max_gaussian_sm;
  // float *sm_out = sm_cum_alpha + 1 * n_pixel_per_tile;

  float out[3] = {0.0f, 0.0f, 0.0f};
  float cum_alpha = 1.0f;

  gaussian_ids += start[tile_id];

  for (int n = 0; n < n_gaussians_this_tile; n += max_gaussian_sm) {
    int num_gaussian_sm = min(max_gaussian_sm, n_gaussians_this_tile - n);
    carry(num_gaussian_sm, 2, sm_mean, mean, NULL, gaussian_ids);
    carry(num_gaussian_sm, 4, sm_cov, cov, NULL, gaussian_ids);
    carry(num_gaussian_sm, 3, sm_color, color, NULL, gaussian_ids);
    carry(num_gaussian_sm, 1, sm_alpha, alpha, NULL, gaussian_ids);
    __syncthreads();
    vol_render_one_batch_v1(num_gaussian_sm, sm_mean, sm_cov, sm_color,
                            sm_alpha, out, cum_alpha, topleft, tile_size,
                            n_tiles_h, n_tiles_w, pixel_size_x, pixel_size_y, H,
                            W, thresh, n == 0);
    __syncthreads();
    gaussian_ids += num_gaussian_sm;
  }
  int global_y = blockIdx.y * tile_size + local_y;
  int global_x = blockIdx.x * tile_size + local_x;
  if (global_y >= H || global_x >= W) {
    return;
  }
  out_rgb[3 * (global_y * W + global_x) + 0] = out[0];
  out_rgb[3 * (global_y * W + global_x) + 1] = out[1];
  out_rgb[3 * (global_y * W + global_x) + 2] = out[2];

  if (isnan(cum_alpha)) {
    printf("cum_alpha is nan\n");
  }
  T[global_y * W + global_x] = cum_alpha;
}

void tile_based_vol_rendering_start_end_cuda_with_T(
    uint32_t N, uint32_t N_with_dub, float *mean, float *cov, float *color,
    float *alpha, int *start, int *end, int *gaussian_ids, float *out_rgb,
    float *topleft, uint32_t tile_size, uint32_t n_tiles_h, uint32_t n_tiles_w,
    float pixel_size_x, float pixel_size_y, uint32_t H, uint32_t W,
    float thresh, float *T) {
  const dim3 block(n_tiles_w, n_tiles_h, 1);
  const int n_pixel_per_tile = tile_size * tile_size;
  tile_based_vol_rendering_with_T<<<block, n_pixel_per_tile>>>(
      N, N_with_dub, mean, cov, color, alpha, start, end, gaussian_ids, out_rgb,
      topleft, tile_size, n_tiles_h, n_tiles_w, pixel_size_x, pixel_size_y, H,
      W, thresh, T);

  cudaError_t last_error;
  checkLastCudaError(last_error);
}