#include "debug.h"
#include "render.h"
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("culling_gaussian_bsphere", &culling_gaussian_bsphere,
        "Cull Gaussian with Bounding Sphere");
  m.def("count_num_gaussians_each_tile", &count_num_gaussians_each_tile,
        "Count number of gaussians in each tile");
  m.def("count_num_gaussians_each_tile_bcircle",
        &count_num_gaussians_each_tile_bcircle,
        "Count number of gaussians in each tile with bounding circle");
  m.def("prepare_image_sort", &prepare_image_sort, "Prepare image for sorting");
  m.def("image_sort", &image_sort, "Image radix sort");
  m.def("tile_based_vol_rendering", &tile_based_vol_rendering,
        "Tile based volume rendering");
  m.def("tile_based_vol_rendering_backward", &tile_based_vol_rendering_backward,
        "Tile based volume rendering backward");
  m.def("debug_check_tiledepth", &debug_check_tiledepth,
        "(DEBUG) check tile and depth");
  m.def("tile_culling_aabb", &tile_culling_aabb, "Tile culling with AABB");
  m.def("tile_based_vol_rendering_v1", &tile_based_vol_rendering_v1,
        "Tile based volume rendering v1 which uses out[3] and cum_alpha in "
        "local var instead of scratchpad memory");
  m.def("tile_based_vol_rendering_v2", &tile_based_vol_rendering_v2,
        "Tile based volume rendering v2 which uses out[3] and cum_alpha in "
        "local var instead of scratchpad memory and further carry gaussian_ids "
        "into scatchpad memory");
  m.def("tile_culling_aabb_start_end", &tile_culling_aabb_start_end,
        "Tile culling with aabb and fill blanks in start and end");
  m.def("tile_based_vol_rendering_start_end",
        &tile_based_vol_rendering_start_end,
        "Tile based volume rendering with start and end array, designed for "
        "aabb culling");
  m.def("tile_based_vol_rendering_backward_start_end",
        &tile_based_vol_rendering_backward_start_end,
        "Tile based volume "
        "rendering backward "
        "with start and end "
        "array, designed for "
        "aabb culling");
  m.def("tile_based_vol_rendering_sh", &tile_based_vol_rendering_sh,
        "Tile based volume rendering with spherical harmonics");
  m.def("tile_based_vol_rendering_backward_sh",
        &tile_based_vol_rendering_backward_sh,
        "Tile based volume rendering "
        "backward with spherical "
        "harmonics");
  m.def("tile_based_vol_rendering_backward_sh_v1",
        &tile_based_vol_rendering_backward_sh_v1,
        "Tile based volume rendering "
        "backward with spherical "
        "harmonics, the gaussian_ids are store in scratchpad memory");
  m.def("tile_based_vol_rendering_backward_sh_warp_reduce",
        &tile_based_vol_rendering_backward_sh_warp_reduce,
        "backward while using warp reduce");
  m.def("tile_based_vol_rendering_sh_with_bg",
        &tile_based_vol_rendering_sh_with_bg,
        "Tile based volume rendering with spherical harmonics and background");
  m.def("tile_based_vol_rendering_backward_sh_with_bg",
        &tile_based_vol_rendering_backward_sh_with_bg,
        "Tile based volume "
        "rendering backward "
        "with spherical "
        "harmonics and "
        "background");
  m.def("tile_based_vol_rendering_scalar", &tile_based_vol_rendering_scalar,
        "Tile based volume rendering with scalar, used for rendering depth or "
        "opacity");
  m.def("tile_based_vol_rendering_scalar_backward",
        &tile_based_vol_rendering_scalar_backward,
        "Tile based volume rendering "
        "backward with scalar, "
        "used for rendering depth "
        "or opacity");
  m.def("tile_based_vol_rendering_start_end_with_T",
        &tile_based_vol_rendering_start_end_with_T,
        "Tile based volume "
        "rendering with start "
        "and end array, designed "
        "for aabb culling, with "
        "transformation matrix");
}