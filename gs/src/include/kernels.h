#pragma once
#include "common.h"
#include "data_spec.h"
#include "helper_math.h"
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ void swap(float &a, float &b) {
  float temp = a;
  a = b;
  b = temp;
}

__device__ void swap(float3 &a, float3 &b) {
  float3 temp = a;
  a = b;
  b = temp;
}

__device__ void transpose(float3 *m, float3 *mt) {
  mt[0] = make_float3(m[0].x, m[1].x, m[2].x);
  mt[1] = make_float3(m[0].y, m[1].y, m[2].y);
  mt[2] = make_float3(m[0].z, m[1].z, m[2].z);
}

__device__ void qvec2rotmat(float4 *qvec, float3 *svec, float3 *m) {
  float w = qvec->x, x = qvec->y, y = qvec->z, z = qvec->w;
  float xx = x * x, yy = y * y, zz = z * z;
  float xy = x * y, xz = x * z, yz = y * z;
  float wx = w * x, wy = w * y, wz = w * z;

  m[0] = make_float3(1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)) * svec[0];
  m[1] = make_float3(2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)) * svec[0];
  m[2] = make_float3(2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)) * svec[0];
}

__device__ void qvec2rotmat(float4 &qvec, float3 &svec, float3 *m) {
  float w = qvec.x, x = qvec.y, y = qvec.z, z = qvec.w;
  float xx = x * x, yy = y * y, zz = z * z;
  float xy = x * y, xz = x * z, yz = y * z;
  float wx = w * x, wy = w * y, wz = w * z;

  m[0] = make_float3(1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)) * svec;
  m[1] = make_float3(2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)) * svec;
  m[2] = make_float3(2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)) * svec;
}

__device__ void qvec2rotmat(float4 &qvec, float3 *m) {
  float w = qvec.x, x = qvec.y, y = qvec.z, z = qvec.w;
  float xx = x * x, yy = y * y, zz = z * z;
  float xy = x * y, xz = x * z, yz = y * z;
  float wx = w * x, wy = w * y, wz = w * z;

  m[0] = make_float3(1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy));
  m[1] = make_float3(2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx));
  m[2] = make_float3(2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy));
}

__device__ void qvec2rotmat_col_major(float4 *qvec, float3 *svec, float3 *m) {
  // get rot matrix from qvec and svec, but store into m with column major
  float w = qvec->x, x = qvec->y, y = qvec->z, z = qvec->w;
  float xx = x * x, yy = y * y, zz = z * z;
  float xy = x * y, xz = x * z, yz = y * z;
  float wx = w * x, wy = w * y, wz = w * z;
  m[0] = make_float3(1 - 2 * (yy + zz), 2 * (xy + wz), 2 * (xz - wy));
  m[1] = make_float3(2 * (xy - wz), 1 - 2 * (xx + zz), 2 * (yz + wx));
  m[2] = make_float3(2 * (xz + wy), 2 * (yz - wx), 1 - 2 * (xx + yy));
}

__device__ void qvec2rotmat_col_major(float4 &qvec, float3 &svec, float3 *m) {
  // get rot matrix from qvec and svec, but store into m with column major
  float w = qvec.x, x = qvec.y, y = qvec.z, z = qvec.w;
  float xx = x * x, yy = y * y, zz = z * z;
  float xy = x * y, xz = x * z, yz = y * z;
  float wx = w * x, wy = w * y, wz = w * z;
  m[0] = make_float3(1 - 2 * (yy + zz), 2 * (xy + wz), 2 * (xz - wy));
  m[1] = make_float3(2 * (xy - wz), 1 - 2 * (xx + zz), 2 * (yz + wx));
  m[2] = make_float3(2 * (xz + wy), 2 * (yz - wx), 1 - 2 * (xx + yy));
}

__device__ float gaussian_kernel_3d_with_inv(float3 *mean, float3 *sigma_inv,
                                             float3 *query) {
  // float3 *sigam_inv_row = reinterpret_cast<float3 *>(sigma_inv);
  float3 x = query[0] - mean[0];
  float3 tmp = make_float3(dot(x, sigma_inv[0]), dot(x, sigma_inv[1]),
                           dot(x, sigma_inv[2]));

  return __expf(-0.5f * dot(x, tmp)) * gs_coeff_3d;
}

__device__ float gaussian_kernel_3d(float3 *mean, float4 *qvec, float3 *svec,
                                    float3 *query) {
  float3 sigma_inv[3], sigma[3];
  qvec2rotmat(qvec, svec, sigma);
  transpose(sigma, sigma_inv);
  float det_sqr = sqrt(svec[0].x * svec[0].y * svec[0].z);

  return gaussian_kernel_3d_with_inv(mean, sigma_inv, query) / det_sqr;
}

__device__ inline float det2x2(float2 *m) {
  return m[0].x * m[1].y - m[0].y * m[1].x;
}

__device__ float gaussian_kernel_2d(float2 &mean, float2 *cov_inv,
                                    float2 &query) {
  float2 mu = query - mean;
  float2 tmp = make_float2(dot(query, cov_inv[0]), dot(query, cov_inv[1]));
  float e_tmp = dot(tmp, query) * 0.5f;

  return gs_coeff_2d * __expf(-e_tmp) * sqrtf(det2x2(cov_inv));
}

// __device__ void covariance2bbox(float3 &mean, float4 &qvec, float3 &svec,
//                                 bbox3d &bbox, float thresh = 3.0f) {
//   float3 rotmat[3];
//   qvec2rotmat(qvec, rotmat);
// #pragma unroll
//   for (int i = 0; i < 8; ++i) {
//     int x = -1 + 2 * (i & 1);
//     int y = -1 + 2 * (i & (1 << 1));
//     int z = -1 + 2 * (i & (1 << 2));
//     bbox.vertices[i] =
//         mean + (x * rotmat[0] + y * rotmat[1] + z * rotmat[2]) * svec *
//         thresh;
//   }
//   return;
// }

__device__ bool point_in_frustum(float3 &query, float3 *normal, float3 *pts) {
#pragma unroll
  for (int i = 0; i < 6; ++i) {
    if (dot(query - pts[i], normal[i]) < -EPS)
      return false;
  }
  return true;
}

// __device__ bool onorforward_bbox3d_plane(bbox3d &bbox, float3 *normal,
//                                          float3 *pts) {
//   assert(true);
//   return false;
// }

// __device__ bool intersect_bbox3d_frustum(bbox3d &bbox, float3 *normal,
//                                          float3 *pts) {
// #pragma unroll
//   for (int i = 0; i < 8; ++i) {
//     if (point_in_frustum(bbox.vertices[i], normal, pts))
//       return true;
//   }
//   return false;
// }

__device__ bool onorforward_sphere_plane(float3 &query, float radius,
                                         float3 &normal, float3 &pts) {
  return dot(query - pts, normal) > -radius;
}

__device__ bool intersect_sphere_frustum(float3 &query, float radius,
                                         float3 *normal, float3 *pts) {
#pragma unroll
  for (int i = 0; i < 6; ++i) {
    if (!onorforward_sphere_plane(query, radius, normal[i], pts[i])) {
      return false;
    }
  }
  return true;
}

__host__ __device__ __forceinline__ float
kernel_gaussian_2d_float(float *mean, float *cov, float *query) {
  float c0 = cov[0];
  float c1 = cov[1];
  float c2 = cov[2];
  float c3 = cov[3];
  float det = c0 * c3 - c1 * c2;
  float x = query[0] - mean[0];
  float y = query[1] - mean[1];
  float tmpx = x * c3 - y * c2;
  float tmpy = -x * c1 + y * c0;
  float radial = tmpx * x + tmpy * y;
  radial /= det;
  if (radial < 0.0) {
    radial = 1000.0;
  }
  float val = (float)expf(-0.5 * radial);
  // checkValue(val);
  // assert(val <= 1.0f && val >= 0.0f);

  return val;
}

__host__ __device__ __forceinline__ float
kernel_gaussian_2d(float *mean, float *cov, float *query) {
  double c0 = (double)cov[0];
  double c1 = (double)cov[1];
  double c2 = (double)cov[2];
  double c3 = (double)cov[3];
  double det = c0 * c3 - c1 * c2;
  double x = query[0] - mean[0];
  double y = query[1] - mean[1];
  double tmpx = x * c3 - y * c2;
  double tmpy = -x * c1 + y * c0;
  double radial = tmpx * x + tmpy * y;
  radial /= det;
  if (radial < 0.0) {
    radial = 1000.0;
  }
  float val = (float)exp(-0.5 * radial);
  // checkValue(val);
  // assert(val <= 1.0f && val >= 0.0f);

  return val;
  // float2 x = make_float2(query[0] - mean[0], query[1] - mean[1]);
  // float2 tmp =
  //     make_float2(x.x * cov[3] - x.y * cov[2], -x.x * cov[1] + x.y * cov[0]);
  // float radial = dot(tmp, x) / det;
  // if (radial > 0)
  //   return expf(-0.5f * radial);
  // else
  //   return 0.0f;
}

__host__ __device__ inline float
kernel_gaussian_2d(float *mean, float *cov, float query_x, float query_y) {
  double c0 = (double)cov[0];
  double c1 = (double)cov[1];
  double c2 = (double)cov[2];
  double c3 = (double)cov[3];
  double det = c0 * c3 - c1 * c2;
  double x = query_x - mean[0];
  double y = query_y - mean[1];
  double tmpx = x * c3 - y * c2;
  double tmpy = -x * c1 + y * c0;
  double radial = tmpx * x + tmpy * y;
  radial /= det;

  if (radial < 0.0) {
    radial = 1000.0;
  }
  float val = (float)exp(-0.5 * radial);
  checkValue(val);
  // if (!(val <= 1.0f && val >= 0.0f)) {
  //   printf("val: %f\n", val);
  // }
  // assert(val <= 1.0f && val >= 0.0f);

  return val;
}

__device__ bool intersect_tile_gaussian2d(float2 &topleft, uint32_t tile_size,
                                          float pixel_size_x,
                                          float pixel_size_y, float *mean,
                                          float *cov, float thresh) {

  float max_val = 0.0f;
  max_val = fmaxf(max_val, kernel_gaussian_2d(mean, cov, topleft.x, topleft.y));
  max_val = fmaxf(
      max_val, kernel_gaussian_2d(
                   mean, cov, topleft.x + tile_size * pixel_size_x, topleft.y));
  max_val =
      fmaxf(max_val, kernel_gaussian_2d(mean, cov, topleft.x,
                                        topleft.y + tile_size * pixel_size_y));
  max_val =
      fmaxf(max_val,
            kernel_gaussian_2d(mean, cov, topleft.x + tile_size * pixel_size_x,
                               topleft.y + tile_size * pixel_size_y));

  return max_val > thresh;
}

__device__ float dist_point_lineseg(float x, float y, float x1, float x2,
                                    float y1, float y2) {
  float A = x - x1;
  float B = y - y1;
  float C = x2 - x1;
  float D = y2 - y1;

  float dot = A * C + B * D;
  float len_sq = C * C + D * D;
  float param = -1;

  if (len_sq != 0) // in case of 0 length line
    param = dot / len_sq;

  float xx, yy;

  if (param < 0) {
    xx = x1;
    yy = y1;
  } else if (param > 1) {
    xx = x2;
    yy = y2;
  } else {
    xx = x1 + param * C;
    yy = y1 + param * D;
  }

  float dx = x - xx;
  float dy = y - yy;
  return sqrtf(dx * dx + dy * dy);
}

__device__ bool intersect_tile_gaussian2d_bcircle(float2 &topleft,
                                                  uint32_t tile_size,
                                                  float pixel_size_x,
                                                  float pixel_size_y,
                                                  float2 *mean, float radius) {
  float2 rela = mean[0] - topleft;
  float px = pixel_size_x * tile_size;
  float py = pixel_size_y * tile_size;
  bool gaussian_inside_tile =
      (rela.x >= 0) && (rela.x <= px) && (rela.y >= 0) && (rela.y <= py);
  if (gaussian_inside_tile) {
    return true;
  }

  float2 pxy = make_float2(px, py);
  float2 nearest;
  // if (rela.x * (rela.x + px) < 0) {
  //   nearest.x = rela.x;
  // } else {
  //   nearest.x = rela.x > 0 ? px : 0;
  // }
  // if (rela.y * (rela.y + py) < 0) {
  //   nearest.y = rela.y;
  // } else {
  //   nearest.y = rela.y > 0 ? py : 0;
  // }
  // if (rela.x * (px - rela.x) < 0.0f) {
  //   nearest.x = rela.x;
  // } else {
  //   nearest.x = rela.x > 0.0f ? px : 0.0f;
  // }
  // if (rela.y * (py - rela.y) < 0.0f) {
  //   nearest.y = rela.y;
  // } else {
  //   nearest.y = rela.y > 0.0f ? py : 0.0f;
  // }
  float d1, d2, d3, d4;
  d1 = dist_point_lineseg(rela.x, rela.y, 0.0f, px, 0.0f, 0.0f);
  d2 = dist_point_lineseg(rela.x, rela.y, 0.0f, px, py, py);
  d3 = dist_point_lineseg(rela.x, rela.y, 0.0f, 0.0f, 0.0f, py);
  d4 = dist_point_lineseg(rela.x, rela.y, px, px, 0.0f, py);
  float d = fminf(fminf(d1, d2), fminf(d3, d4));
  // printf("gaussian value: %.2f\n", gaussian_kernel_2d(rela, cov_inv,
  // nearest));
  // float R = sqrtf(dot(rela - nearest, rela - nearest));
  // printf("R: %.2f; radius: %.2f\n", R, radius);
  // return dot(rela - nearest, rela - nearest) < radius * radius;
  return d < radius;
}

// __host__ __device__ inline float kernel_gaussian_2d(float2 &mean, float4
// &cov,
//                                                     float2 &query) {
//   float2 xy = query - mean;
//   float2 tmp =
//       make_float2(xy.x * cov.x + xy.y * cov.y, xy.x * cov.z + xy.y * cov.w);
//   float radial = dot(tmp, xy);
//   return expf(-0.5f * radial);
//   // return 1.0f;
// }

__device__ inline void
kernel_gaussian_2d_backward_float(float *mean, float *cov, float *query,
                                  float *grad_mean, float *grad_cov,
                                  float grad) {
  // nan
  float d_grad = grad;
  float c0 = cov[0];
  float c1 = cov[1];
  float c2 = cov[2];
  float c3 = cov[3];
  float det = c0 * c3 - c1 * c2;
  float x = query[0] - mean[0];
  float y = query[1] - mean[1];
  float tmpx = (x * c3 - y * c2) / det;
  float tmpy = (-x * c1 + y * c0) / det;
  // note: val has been multiplied in the grad
  // atomic ops here
  // checkValue((float)(d_grad * tmpx));
  // checkValue((float)(d_grad * tmpy * tmpy));
  atomicAdd(grad_mean, (float)(d_grad * tmpx));
  atomicAdd(grad_mean + 1, (float)(d_grad * tmpy));
  atomicAdd(grad_cov, 0.5 * (float)(d_grad * tmpx * tmpx));
  atomicAdd(grad_cov + 1, 0.5 * (float)(d_grad * tmpx * tmpy));
  atomicAdd(grad_cov + 2, 0.5 * (float)(d_grad * tmpy * tmpx));
  atomicAdd(grad_cov + 3, 0.5 * (float)(d_grad * tmpy * tmpy));
}

__device__ inline void
kernel_gaussian_2d_backward(float *mean, float *cov, float *query,
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
  // checkValue((float)(d_grad * tmpx));
  // checkValue((float)(d_grad * tmpy * tmpy));
  atomicAdd(grad_mean, (float)(d_grad * tmpx));
  atomicAdd(grad_mean + 1, (float)(d_grad * tmpy));
  atomicAdd(grad_cov, 0.5 * (float)(d_grad * tmpx * tmpx));
  atomicAdd(grad_cov + 1, 0.5 * (float)(d_grad * tmpx * tmpy));
  atomicAdd(grad_cov + 2, 0.5 * (float)(d_grad * tmpy * tmpx));
  atomicAdd(grad_cov + 3, 0.5 * (float)(d_grad * tmpy * tmpy));
}

__device__ inline void
kernel_gaussian_2d_backward_nonatomic(float *mean, float *cov, float *query,
                                      float *grad_mean, float *grad_cov,
                                      float grad) {
  // non-atomic version
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
  // checkValue((float)(d_grad * tmpx));
  // checkValue((float)(d_grad * tmpy * tmpy));
  grad_mean[0] = (float)(d_grad * tmpx);
  grad_mean[1] = (float)(d_grad * tmpy);
  grad_cov[0] = 0.5 * (float)(d_grad * tmpx * tmpx);
  grad_cov[1] = 0.5 * (float)(d_grad * tmpx * tmpy);
  grad_cov[2] = 0.5 * (float)(d_grad * tmpy * tmpx);
  grad_cov[3] = 0.5 * (float)(d_grad * tmpy * tmpy);
}