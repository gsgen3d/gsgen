#ifndef _DATA_SPEC_H
#define _DATA_SPEC_H
#include "common.h"
#include <iostream>
#include <stdint.h>

struct bbox3d {
  float3 vertices[8];
};

struct bbox2d {
  float3 vertices[4];
};

struct Sphere {
  float3 center;
  float radius;
};

struct Camera {
  float3 eye;
  float3 pixel_dir_x, pixel_dir_y;
};

struct Renderer {
  float3 *mean_d;
  float4 *qvec_d;
  float3 *svec_d;
  uint32_t *ids_d;
  uint32_t *ids_expanded;
};

struct Tile {
  float2 topleft;
  uint32_t xe, ye;
};

#endif