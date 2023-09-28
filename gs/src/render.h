#include "common.h"

void culling_gaussian_bsphere(Tensor mean, Tensor qvec, Tensor svec,
                              Tensor normal, Tensor pts, Tensor mask,
                              float thresh);

void count_num_gaussians_each_tile(Tensor mean, Tensor cov_inv, Tensor topleft,
                                   uint32_t tile_size, uint32_t n_tiles_h,
                                   uint32_t n_tiles_w, float pixel_size_x,
                                   float pixel_size_y, Tensor num_gaussians,
                                   float thresh);

void count_num_gaussians_each_tile_bcircle(
    Tensor mean, Tensor radius, Tensor topleft, uint32_t tile_size,
    uint32_t n_tiles_h, uint32_t n_tiles_w, float pixel_size_x,
    float pixel_size_y, Tensor num_gaussians);

void prepare_image_sort(Tensor gaussian_ids, Tensor tiledepth, Tensor depth,
                        Tensor tile_n_gaussians, Tensor offset, Tensor mean,
                        Tensor radius, Tensor topleft, uint32_t tile_size,
                        uint32_t n_tiles_h, uint32_t n_tiles_w,
                        float pixel_size_x, float pixel_size_y);

void image_sort(Tensor gaussian_ids, Tensor tiledepth, Tensor depth,
                Tensor tile_n_gaussians, Tensor offset, Tensor mean, Tensor cov,
                Tensor topleft, uint32_t tile_size, uint32_t n_tiles_h,
                uint32_t n_tiles_w, float pixel_size_x, float pixel_size_y,
                float thresh);

void tile_based_vol_rendering(Tensor mean, Tensor cov, Tensor color,
                              Tensor alpha, Tensor offset, Tensor gaussian_ids,
                              Tensor out, Tensor topleft, uint32_t tile_size,
                              uint32_t n_tiles_h, uint32_t n_tiles_w,
                              float pixel_size_x, float pixel_size_y,
                              uint32_t H, uint32_t W, float thresh);

void tile_based_vol_rendering_v1(Tensor mean, Tensor cov, Tensor color,
                                 Tensor alpha, Tensor offset,
                                 Tensor gaussian_ids, Tensor out,
                                 Tensor topleft, uint32_t tile_size,
                                 uint32_t n_tiles_h, uint32_t n_tiles_w,
                                 float pixel_size_x, float pixel_size_y,
                                 uint32_t H, uint32_t W, float thresh);

void tile_based_vol_rendering_v2(Tensor mean, Tensor cov, Tensor color,
                                 Tensor alpha, Tensor offset,
                                 Tensor gaussian_ids, Tensor out,
                                 Tensor topleft, uint32_t tile_size,
                                 uint32_t n_tiles_h, uint32_t n_tiles_w,
                                 float pixel_size_x, float pixel_size_y,
                                 uint32_t H, uint32_t W, float thresh);

void tile_based_vol_rendering_backward(
    Tensor mean, Tensor cov, Tensor color, Tensor alpha, Tensor offset,
    Tensor gaussian_ids, Tensor out, Tensor grad_mean, Tensor grad_cov,
    Tensor grad_color, Tensor grad_alpha, Tensor grad_out, Tensor topleft,
    uint32_t tile_size, uint32_t n_tiles_h, uint32_t n_tiles_w,
    float pixel_size_x, float pixel_size_y, uint32_t H, uint32_t W,
    float thresh);

void tile_culling_aabb(Tensor aabb_topleft, Tensor aabb_bottomright,
                       Tensor gaussian_ids, Tensor offset, Tensor depth,
                       uint32_t n_tiles_h, uint32_t n_tiles_w);

void tile_culling_aabb_start_end(Tensor aabb_topleft, Tensor aabb_bottomright,
                                 Tensor gaussian_ids, Tensor start, Tensor end,
                                 Tensor depth, uint32_t n_tiles_h,
                                 uint32_t n_tiles_w);

void tile_based_vol_rendering_start_end(Tensor mean, Tensor cov, Tensor color,
                                        Tensor alpha, Tensor start, Tensor end,
                                        Tensor gaussian_ids, Tensor out,
                                        Tensor topleft, uint32_t tile_size,
                                        uint32_t n_tiles_h, uint32_t n_tiles_w,
                                        float pixel_size_x, float pixel_size_y,
                                        uint32_t H, uint32_t W, float thresh);

void tile_based_vol_rendering_backward_start_end(
    Tensor mean, Tensor cov, Tensor color, Tensor alpha, Tensor start,
    Tensor end, Tensor gaussian_ids, Tensor out, Tensor grad_mean,
    Tensor grad_cov, Tensor grad_color, Tensor grad_alpha, Tensor grad_out,
    Tensor topleft, uint32_t tile_size, uint32_t n_tiles_h, uint32_t n_tiles_w,
    float pixel_size_x, float pixel_size_y, uint32_t H, uint32_t W,
    float thresh);

void tile_based_vol_rendering_sh(Tensor mean, Tensor cov, Tensor sh_coeffs,
                                 Tensor alpha, Tensor start, Tensor end,
                                 Tensor gaussian_ids, Tensor out,
                                 Tensor topleft, Tensor c2w, uint32_t tile_size,
                                 uint32_t n_tiles_h, uint32_t n_tiles_w,
                                 float pixel_size_x, float pixel_size_y,
                                 uint32_t H, uint32_t W, uint32_t C,
                                 float thresh);

void tile_based_vol_rendering_backward_sh(
    Tensor mean, Tensor cov, Tensor sh_coeffs, Tensor alpha, Tensor start,
    Tensor end, Tensor gaussian_ids, Tensor out, Tensor grad_mean,
    Tensor grad_cov, Tensor grad_sh_coeffs, Tensor grad_alpha, Tensor grad_out,
    Tensor topleft, Tensor c2w, uint32_t tile_size, uint32_t n_tiles_h,
    uint32_t n_tiles_w, float pixel_size_x, float pixel_size_y, uint32_t H,
    uint32_t W, uint32_t C, float thresh);

void tile_based_vol_rendering_backward_sh_v1(
    Tensor mean, Tensor cov, Tensor sh_coeffs, Tensor alpha, Tensor start,
    Tensor end, Tensor gaussian_ids, Tensor out, Tensor grad_mean,
    Tensor grad_cov, Tensor grad_sh_coeffs, Tensor grad_alpha, Tensor grad_out,
    Tensor topleft, Tensor c2w, uint32_t tile_size, uint32_t n_tiles_h,
    uint32_t n_tiles_w, float pixel_size_x, float pixel_size_y, uint32_t H,
    uint32_t W, uint32_t C, float thresh);

void tile_based_vol_rendering_backward_sh_warp_reduce(
    Tensor mean, Tensor cov, Tensor sh_coeffs, Tensor alpha, Tensor start,
    Tensor end, Tensor gaussian_ids, Tensor out, Tensor grad_mean,
    Tensor grad_cov, Tensor grad_sh_coeffs, Tensor grad_alpha, Tensor grad_out,
    Tensor topleft, Tensor c2w, uint32_t tile_size, uint32_t n_tiles_h,
    uint32_t n_tiles_w, float pixel_size_x, float pixel_size_y, uint32_t H,
    uint32_t W, uint32_t C, float thresh);

void tile_based_vol_rendering_sh_with_bg(
    Tensor mean, Tensor cov, Tensor sh_coeffs, Tensor alpha, Tensor start,
    Tensor end, Tensor gaussian_ids, Tensor out, Tensor topleft, Tensor c2w,
    uint32_t tile_size, uint32_t n_tiles_h, uint32_t n_tiles_w,
    float pixel_size_x, float pixel_size_y, uint32_t H, uint32_t W, uint32_t C,
    float thresh, Tensor bg_rgb);

void tile_based_vol_rendering_backward_sh_with_bg(
    Tensor mean, Tensor cov, Tensor sh_coeffs, Tensor alpha, Tensor start,
    Tensor end, Tensor gaussian_ids, Tensor out, Tensor grad_mean,
    Tensor grad_cov, Tensor grad_sh_coeffs, Tensor grad_alpha, Tensor grad_out,
    Tensor topleft, Tensor c2w, uint32_t tile_size, uint32_t n_tiles_h,
    uint32_t n_tiles_w, float pixel_size_x, float pixel_size_y, uint32_t H,
    uint32_t W, uint32_t C, float thresh, Tensor bg_rgb);

void tile_based_vol_rendering_scalar(Tensor mean, Tensor cov, Tensor scalar,
                                     Tensor alpha, Tensor start, Tensor end,
                                     Tensor gaussian_ids, Tensor out,
                                     Tensor topleft, uint32_t tile_size,
                                     uint32_t n_tiles_h, uint32_t n_tiles_w,
                                     float pixel_size_x, float pixel_size_y,
                                     uint32_t H, uint32_t W, float thresh,
                                     Tensor T);

void tile_based_vol_rendering_scalar_backward(
    Tensor mean, Tensor cov, Tensor scalar, Tensor alpha, Tensor start,
    Tensor end, Tensor gaussian_ids, Tensor out, Tensor grad_mean,
    Tensor grad_cov, Tensor grad_scalar, Tensor grad_alpha, Tensor grad_out,
    Tensor topleft, uint32_t tile_size, uint32_t n_tiles_h, uint32_t n_tiles_w,
    float pixel_size_x, float pixel_size_y, uint32_t H, uint32_t W,
    float thresh);

void tile_based_vol_rendering_start_end_with_T(
    Tensor mean, Tensor cov, Tensor color, Tensor alpha, Tensor start,
    Tensor end, Tensor gaussian_ids, Tensor out, Tensor topleft,
    uint32_t tile_size, uint32_t n_tiles_h, uint32_t n_tiles_w,
    float pixel_size_x, float pixel_size_y, uint32_t H, uint32_t W,
    float thresh, Tensor T);