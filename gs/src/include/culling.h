#pragma once
#include "common.h"
#include "data_spec.h"
#include "device_launch_parameters.h"
#include "kernels.h"
#include <cub/cub.cuh>
#include <device_atomic_functions.h>
#include <device_launch_parameters.h>
#include <stdint.h>

__global__ void culling_gaussian_bsphere_kernel(uint32_t N, float3 *mean,
                                                float4 *qvec, float3 *svec,
                                                float3 *normal, float3 *pts,
                                                bool *mask, float thresh) {
  tid_1d(tid);
  if (tid >= N)
    return;
  float r = fmaxf(fmaxf(svec[tid].x, svec[tid].y), svec[tid].z) * thresh;
  mask[tid] = intersect_sphere_frustum(mean[tid], r, normal, pts);
}

void culling_gaussian_bsphere_cuda(uint32_t N, float *mean, float *qvec,
                                   float *svec, float *normal, float *pts,
                                   bool *mask, float thresh) {
  float3 *mean_ = reinterpret_cast<float3 *>(mean);
  float4 *qvec_ = reinterpret_cast<float4 *>(qvec);
  float3 *svec_ = reinterpret_cast<float3 *>(svec);
  float3 *normal_ = reinterpret_cast<float3 *>(normal);
  float3 *pts_ = reinterpret_cast<float3 *>(pts);

  uint32_t n_blocks = div_round_up(N, (uint32_t)N_THREADS);
  culling_gaussian_bsphere_kernel<<<n_blocks, N_THREADS>>>(
      N, mean_, qvec_, svec_, normal_, pts_, mask, thresh);
}

__device__ void direct_carry(uint32_t N, uint32_t dsize, float *sm, float *gm) {
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

__device__ void fill_tiledepth_bcircle_cuda_batch(
    uint32_t N, int *gaussian_ids, double *tiledepth, float *depth,
    int *tile_n_gaussians, int *offset, float *mean, float *radius,
    float *topleft, uint32_t tile_size, uint32_t n_tiles_h, uint32_t n_tiles_w,
    float pixel_size_x, float pixel_size_y, int &n_gaussians_this_tile,
    int &off, int N_base) {
  tid_1d(tid);
  if (tid >= n_tiles_h * n_tiles_w)
    return;
  assert(off >= 0);
  int tile_x = tid % n_tiles_w;
  int tile_y = tid / n_tiles_w;
  float2 *mean_ = reinterpret_cast<float2 *>(mean);
  float2 tile_topleft =
      make_float2(topleft[0] + pixel_size_x * tile_x * tile_size,
                  topleft[1] + pixel_size_y * tile_y * tile_size);
  int *tile_ids = reinterpret_cast<int *>(tiledepth);
  float *tile_depths = reinterpret_cast<float *>(tiledepth);
  for (int i = 0; i < N; i++) {
    if (intersect_tile_gaussian2d_bcircle(tile_topleft, tile_size, pixel_size_x,
                                          pixel_size_y, mean_ + i, radius[i])) {
      tile_ids[2 * off + 1] = tid;
      tile_depths[2 * off] = depth[i];
      assert(gaussian_ids[off] == 0);
      gaussian_ids[off] = i + N_base;
      off += 1;
      n_gaussians_this_tile += 1;
    }
  }
}

__global__ void fill_tiledepth_bcircle_cuda_entry(
    uint32_t N, int *gaussian_ids, double *tiledepth, float *depth,
    int *tile_n_gaussians, int *offset, float *mean, float *radius,
    float *topleft, uint32_t tile_size, uint32_t n_tiles_h, uint32_t n_tiles_w,
    float pixel_size_x, float pixel_size_y) {
  tid_1d(tid);
  const int max_num_gaussians_sm = MAX_N_FLOAT_SM / 3;
  __shared__ float sm[MAX_N_FLOAT_SM];
  float *sm_mean = sm;
  float *sm_radius = sm_mean + 2 * max_num_gaussians_sm;
  int off;
  if (tid >= n_tiles_h * n_tiles_w) {
    off = -1;
  } else {
    off = offset[tid];
  }
  int n_gaussians_this_tile = 0;
  for (int i = 0; i < N; i += max_num_gaussians_sm) {
    int n_gaussians = min(max_num_gaussians_sm, N - i);
    direct_carry(n_gaussians, 2, sm_mean, mean + i * 2);
    direct_carry(n_gaussians, 1, sm_radius, radius + i);
    __syncthreads();
    fill_tiledepth_bcircle_cuda_batch(
        n_gaussians, gaussian_ids, tiledepth, depth + i, tile_n_gaussians,
        offset, sm_mean, sm_radius, topleft, tile_size, n_tiles_h, n_tiles_w,
        pixel_size_x, pixel_size_y, n_gaussians_this_tile, off, i);
    __syncthreads();
  }
  if (tid >= n_tiles_h * n_tiles_w)
    return;
  // tile_n_gaussians[tid] = n_gaussians_this_tile;
  if (tid + 1 != n_tiles_w * n_tiles_h) {
    if (off != offset[tid + 1]) {
      printf("tid: %d; off %d offset[tid+1] %d\n", tid, off, offset[tid + 1]);
    }
    assert(off == offset[tid + 1]);
  }
  assert(tile_n_gaussians[tid] == n_gaussians_this_tile);
}

void fill_tiledepth_bcircle_cuda(uint32_t N, int *gaussian_ids,
                                 double *tiledepth, float *depth,
                                 int *tile_n_gaussians, int *offset,
                                 float *mean, float *radius, float *topleft,
                                 uint32_t tile_size, uint32_t n_tiles_h,
                                 uint32_t n_tiles_w, float pixel_size_x,
                                 float pixel_size_y) {
  int n_tiles = n_tiles_h * n_tiles_w;
  uint32_t n_blocks = div_round_up(n_tiles_h * n_tiles_w, (uint32_t)N_THREADS);
  fill_tiledepth_bcircle_cuda_entry<<<n_blocks, N_THREADS>>>(
      N, gaussian_ids, tiledepth, depth, tile_n_gaussians, offset, mean, radius,
      topleft, tile_size, n_tiles_h, n_tiles_w, pixel_size_x, pixel_size_y);
}