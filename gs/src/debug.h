#include "common.h"

void check_tiledepth_host(size_t N, int *offset, double *tiledepth) {
  int *tile_ids = reinterpret_cast<int *>(tiledepth);
  float *tile_depth = reinterpret_cast<float *>(tiledepth);
  printf("N : %d\n", N);
  for (size_t i = 0; i < N - 1; i++) {
    int start = offset[i];
    int end = offset[i + 1];
    for (size_t j = start; j < end - 1; j++) {
      if (end - start > 1) {
        printf("[DEBUG] [CUDA] tile %d==%d n = %d depth = %f\n", i,
               tile_ids[2 * j + 1], end - start, tile_depth[2 * j]);
      }
      assert(tile_ids[2 * j + 1] == i);
      assert(tile_ids[2 * (j + 1) + 1] == i);
      assert(tile_depth[2 * j] <= tile_depth[2 * (j + 1)]);
    }
    if (end - start > 1) {
      printf("[DEBUG] [CUDA] tile %d==%d n = %d depth = %f\n", i,
             tile_ids[2 * (end - 1) + 1], end - start,
             tile_depth[2 * (end - 1)]);
    }
  }
  printf("[DEBUG] [CUDA] Passed tiledepth check\n");
}

void debug_check_tiledepth(Tensor offset, Tensor tiledepth) {
  // make sure tensors are on host device
  size_t N = offset.size(0);
  check_tiledepth_host(N, offset.data_ptr<int>(), tiledepth.data_ptr<double>());
}
