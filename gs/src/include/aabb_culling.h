#pragma once
#include "common.h"
#include "data_spec.h"
#include "device_launch_parameters.h"
#include "kernels.h"
#include <cub/cub.cuh>
#include <device_atomic_functions.h>
#include <device_launch_parameters.h>
#include <stdint.h>

__device__ __forceinline__ int xy2tile_id(int x, int y, int n_tiles_w) {
  return y * n_tiles_w + x;
}

__global__ void fill_tiledepth_aabb(uint32_t N, uint32_t N_with_dub,
                                    uint32_t n_tiles_h, uint32_t n_tiles_w,
                                    int *tile_ids, float *tile_depth,
                                    int *gaussian_ids, int *aabb_topleft,
                                    int *aabb_bottomright, float *depth,
                                    int *size) {
  int global_id = blockIdx.x * blockDim.x + threadIdx.x; // gaussian id
  if (global_id >= N) {
    return;
  }
  int start_x = aabb_topleft[global_id * 2];
  int start_y = aabb_topleft[global_id * 2 + 1];
  int end_x = aabb_bottomright[global_id * 2];
  int end_y = aabb_bottomright[global_id * 2 + 1];
  for (int i = start_x; i <= end_x; i++) {
    for (int j = start_y; j <= end_y; j++) {
      int tileid = xy2tile_id(i, j, n_tiles_w);
      int pos = atomicAdd(size, 1);
      // DEBUG
      assert(tile_ids[2 * pos] == 0);
      assert(tile_depth[2 * pos + 1] == 0);
      tile_ids[2 * pos + 1] = tileid;
      tile_depth[2 * pos] = depth[global_id];
      gaussian_ids[pos] = global_id;
    }
  }
}

__global__ void fill_offset_aabb(uint32_t N_with_dub, int *sorted_tile_ids,
                                 int *offset) {
  int global_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (global_id >= N_with_dub) {
    return;
  }
  int tile_id = sorted_tile_ids[global_id * 2 + 1];
  if (global_id == 0) {
    offset[tile_id] = 0;
    return;
  }
  int prev_tile_id = sorted_tile_ids[(global_id - 1) * 2 + 1];
  if (prev_tile_id > tile_id) {
    printf("prev_tile_id: %d, tile_id: %d, global_id: %d\n", prev_tile_id,
           tile_id, global_id);
  }
  assert(prev_tile_id <= tile_id);
  if (prev_tile_id != tile_id) {
    offset[tile_id] = global_id;
  }
  // DEBUG
  if (prev_tile_id == tile_id) {
    float *depth = reinterpret_cast<float *>(sorted_tile_ids);
    assert(depth[global_id * 2] >= depth[(global_id - 1) * 2]);
  }
}

__global__ void fill_start_aabb(uint32_t N_with_dub, int *sorted_tile_ids,
                                int *start) {
  int global_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (global_id >= N_with_dub) {
    return;
  }
  int tile_id = sorted_tile_ids[global_id * 2 + 1];
  if (global_id == 0) {
    start[tile_id] = 0;
    return;
  }
  int prev_tile_id = sorted_tile_ids[(global_id - 1) * 2 + 1];
  assert(prev_tile_id <= tile_id);
  if (prev_tile_id != tile_id) {
    start[tile_id] = global_id;
  }
}

__global__ void fill_end_aabb(uint32_t N_with_dub, int *sorted_tile_ids,
                              int *end) {
  int global_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (global_id >= N_with_dub) {
    return;
  }
  int tile_id = sorted_tile_ids[global_id * 2 + 1];
  if (global_id == N_with_dub - 1) {
    end[tile_id] = N_with_dub;
    return;
  }
  int next_tile_id = sorted_tile_ids[(global_id + 1) * 2 + 1];
  if (next_tile_id != tile_id) {
    end[tile_id] = global_id + 1;
  }
}

__global__ void fill_offset_for_blank_tiles_aabb(uint32_t N_tiles,
                                                 int *offset) {
  int global_id = blockIdx.x * blockDim.x + threadIdx.x; // stands for tile_id
  if (global_id >= N_tiles) {
    return;
  }
  if (offset[global_id] == -1) {
    offset[global_id] = offset[global_id + 1];
  }
}

void tile_culling_aabb_cuda(uint32_t N, uint32_t N_with_dub, uint32_t n_tiles_h,
                            uint32_t n_tiles_w, int *gaussian_ids, int *offset,
                            int *aabb_topleft, int *aabb_bottomright,
                            float *depth) {
  // offset should have length n_tiles + 1
  GpuTimer timer;

  int n_tiles = n_tiles_h * n_tiles_w;
  cudaCheck(cudaMemset(offset, -1, sizeof(int) * (n_tiles + 1)));
  cudaCheck(cudaMemcpy(offset + n_tiles, &N_with_dub, sizeof(int),
                       cudaMemcpyHostToDevice)); // stupid way but I idk a
                                                 // better one to do this

  int64_t *tiledepth, *sorted_tiledepth;
  int *unsorted_gaussian_ids;
  cudaCheck(cudaMalloc((void **)&tiledepth, sizeof(int64_t) * N_with_dub));
  int *size;
  cudaCheck(cudaMalloc((void **)&size, sizeof(int)));
  cudaCheck(cudaMemset(size, 0, sizeof(int)));
  int *tile_ids = reinterpret_cast<int *>(tiledepth);
  float *tile_depth = reinterpret_cast<float *>(tiledepth);

  cudaCheck(
      cudaMalloc((void **)&sorted_tiledepth, sizeof(int64_t) * N_with_dub));
  cudaCheck(
      cudaMalloc((void **)&unsorted_gaussian_ids, sizeof(int) * N_with_dub));

  uint32_t n_blocks = div_round_up(N, (uint32_t)N_THREADS);
  // DEBUG
  // cudaCheck(cudaMemset(tiledepth, 0, sizeof(int64_t) * N_with_dub));
  timer.Start();
  fill_tiledepth_aabb<<<n_blocks, N_THREADS>>>(
      N, N_with_dub, n_tiles_h, n_tiles_w, tile_ids, tile_depth,
      unsorted_gaussian_ids, aabb_topleft, aabb_bottomright, depth, size);
  timer.Stop();
  timer.Elapsed("fill_tiledepth_aabb");

  int size_h;
  cudaCheck(cudaMemcpy(&size_h, size, sizeof(int), cudaMemcpyDeviceToHost));
  assert(size_h == N_with_dub);
  cudaCheck(cudaFree(size));

  // device radix sort
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  timer.Start();
  cudaCheck(cub::DeviceRadixSort::SortPairs(
      d_temp_storage, temp_storage_bytes, tiledepth, sorted_tiledepth,
      unsorted_gaussian_ids, gaussian_ids, N_with_dub));
  cudaCheck(cudaMalloc((void **)&d_temp_storage, temp_storage_bytes));
  cudaCheck(cub::DeviceRadixSort::SortPairs(
      d_temp_storage, temp_storage_bytes, tiledepth, sorted_tiledepth,
      unsorted_gaussian_ids, gaussian_ids, N_with_dub));
  timer.Stop();
  timer.Elapsed("radix sort");

  n_blocks = div_round_up(N_with_dub, (uint32_t)N_THREADS);
  timer.Start();
  int *sorted_tile_ids = reinterpret_cast<int *>(sorted_tiledepth);
  fill_offset_aabb<<<n_blocks, N_THREADS>>>(N_with_dub, sorted_tile_ids,
                                            offset);
  timer.Stop();
  timer.Elapsed("fill offset");

  // n_blocks = div_round_up(n_tiles, N_THREADS);
  // timer.Start();
  // fill_offset_for_blank_tiles_aabb<<<n_blocks, N_THREADS>>>(n_tiles, offset);
  // timer.Stop();
  // printf("fill_offset_for_blank_tiles_aabb: %f ms\n", timer.Elapsed());

  cudaCheck(cudaFree(d_temp_storage));
  cudaCheck(cudaFree(tiledepth));
  cudaCheck(cudaFree(sorted_tiledepth));
  cudaCheck(cudaFree(unsorted_gaussian_ids));
}

void tile_culling_aabb_start_end_cuda(uint32_t N, uint32_t N_with_dub,
                                      uint32_t n_tiles_h, uint32_t n_tiles_w,
                                      int *gaussian_ids, int *start, int *end,
                                      int *aabb_topleft, int *aabb_bottomright,
                                      float *depth) {
  // offset should have length n_tiles + 1
  GpuTimer timer;

  int n_tiles = n_tiles_h * n_tiles_w;

  int64_t *tiledepth, *sorted_tiledepth;
  int *unsorted_gaussian_ids;
  cudaCheck(cudaMalloc((void **)&tiledepth, sizeof(int64_t) * N_with_dub));
  int *size;
  cudaCheck(cudaMalloc((void **)&size, sizeof(int)));
  cudaCheck(cudaMemset(size, 0, sizeof(int)));
  int *tile_ids = reinterpret_cast<int *>(tiledepth);
  float *tile_depth = reinterpret_cast<float *>(tiledepth);

  cudaCheck(
      cudaMalloc((void **)&sorted_tiledepth, sizeof(int64_t) * N_with_dub));
  cudaCheck(
      cudaMalloc((void **)&unsorted_gaussian_ids, sizeof(int) * N_with_dub));

  uint32_t n_blocks = div_round_up(N, (uint32_t)N_THREADS);
  // DEBUG
  cudaCheck(cudaMemset(tiledepth, 0, sizeof(int64_t) * N_with_dub));
  timer.Start();
  fill_tiledepth_aabb<<<n_blocks, N_THREADS>>>(
      N, N_with_dub, n_tiles_h, n_tiles_w, tile_ids, tile_depth,
      unsorted_gaussian_ids, aabb_topleft, aabb_bottomright, depth, size);
  timer.Stop();
  timer.Elapsed("fill tiledepth");

  int size_h;
  cudaCheck(cudaMemcpy(&size_h, size, sizeof(int), cudaMemcpyDeviceToHost));
  assert(size_h == N_with_dub);
  cudaCheck(cudaFree(size));

  // device radix sort
  timer.Start();
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cudaCheck(cub::DeviceRadixSort::SortPairs(
      d_temp_storage, temp_storage_bytes, tiledepth, sorted_tiledepth,
      unsorted_gaussian_ids, gaussian_ids, N_with_dub));
  cudaCheck(cudaMalloc((void **)&d_temp_storage, temp_storage_bytes));
  cudaCheck(cub::DeviceRadixSort::SortPairs(
      d_temp_storage, temp_storage_bytes, tiledepth, sorted_tiledepth,
      unsorted_gaussian_ids, gaussian_ids, N_with_dub));
  timer.Stop();
  timer.Elapsed("radix sort");

  n_blocks = div_round_up(N_with_dub, (uint32_t)N_THREADS);

  timer.Start();
  cudaCheck(cudaMemset(start, -1, sizeof(int) * n_tiles));
  cudaCheck(cudaMemset(end, -1, sizeof(int) * n_tiles));
  int *sorted_tile_ids = reinterpret_cast<int *>(sorted_tiledepth);
  fill_start_aabb<<<n_blocks, N_THREADS>>>(N_with_dub, sorted_tile_ids, start);
  fill_end_aabb<<<n_blocks, N_THREADS>>>(N_with_dub, sorted_tile_ids, end);
  timer.Stop();
  timer.Elapsed("fill start end");

  cudaCheck(cudaFree(d_temp_storage));
  cudaCheck(cudaFree(tiledepth));
  cudaCheck(cudaFree(sorted_tiledepth));
  cudaCheck(cudaFree(unsorted_gaussian_ids));
}