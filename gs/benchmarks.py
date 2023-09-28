import torch
import _gs as _backend


class _render_v0(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        mean,
        cov,
        color,
        alpha,
        offset,
        gaussian_ids,
        topleft,
        tile_size,
        n_tiles_h,
        n_tiles_w,
        pixel_size_x,
        pixel_size_y,
        H,
        W,
        thresh,
    ):
        out = torch.zeros([H * W * 3], dtype=torch.float32, device=mean.device)
        _backend.tile_based_vol_rendering(
            mean,
            cov,
            color,
            alpha,
            offset,
            gaussian_ids,
            out,
            topleft,
            tile_size,
            n_tiles_h,
            n_tiles_w,
            pixel_size_x,
            pixel_size_y,
            H,
            W,
            thresh,
        )
        ctx.save_for_backward(
            mean, cov, color, alpha, offset, gaussian_ids, out, topleft
        )
        ctx.const = [
            tile_size,
            n_tiles_h,
            n_tiles_w,
            pixel_size_x,
            pixel_size_y,
            H,
            W,
            thresh,
        ]

        return out

    @staticmethod
    def backward(ctx, grad):
        mean, cov, color, alpha, offset, gaussian_ids, out, topleft = ctx.saved_tensors
        grad_mean = torch.zeros_like(mean)
        grad_cov = torch.zeros_like(cov)
        grad_color = torch.zeros_like(color)
        grad_alpha = torch.zeros_like(alpha)
        (
            tile_size,
            n_tiles_h,
            n_tiles_w,
            pixel_size_x,
            pixel_size_y,
            H,
            W,
            thresh,
        ) = ctx.const

        _backend.tile_based_vol_rendering_backward(
            mean,
            cov,
            color,
            alpha,
            offset,
            gaussian_ids,
            out,
            grad_mean,
            grad_cov,
            grad_color,
            grad_alpha,
            grad,
            topleft,
            tile_size,
            n_tiles_h,
            n_tiles_w,
            pixel_size_x,
            pixel_size_y,
            H,
            W,
            thresh,
        )

        return (
            grad_mean,
            grad_cov,
            grad_color,
            grad_alpha,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _render_v1(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        mean,
        cov,
        color,
        alpha,
        offset,
        gaussian_ids,
        topleft,
        tile_size,
        n_tiles_h,
        n_tiles_w,
        pixel_size_x,
        pixel_size_y,
        H,
        W,
        thresh,
    ):
        out = torch.zeros([H * W * 3], dtype=torch.float32, device=mean.device)
        _backend.tile_based_vol_rendering_v1(
            mean,
            cov,
            color,
            alpha,
            offset,
            gaussian_ids,
            out,
            topleft,
            tile_size,
            n_tiles_h,
            n_tiles_w,
            pixel_size_x,
            pixel_size_y,
            H,
            W,
            thresh,
        )
        ctx.save_for_backward(
            mean, cov, color, alpha, offset, gaussian_ids, out, topleft
        )
        ctx.const = [
            tile_size,
            n_tiles_h,
            n_tiles_w,
            pixel_size_x,
            pixel_size_y,
            H,
            W,
            thresh,
        ]

        return out

    @staticmethod
    def backward(ctx, grad):
        mean, cov, color, alpha, offset, gaussian_ids, out, topleft = ctx.saved_tensors
        grad_mean = torch.zeros_like(mean)
        grad_cov = torch.zeros_like(cov)
        grad_color = torch.zeros_like(color)
        grad_alpha = torch.zeros_like(alpha)
        (
            tile_size,
            n_tiles_h,
            n_tiles_w,
            pixel_size_x,
            pixel_size_y,
            H,
            W,
            thresh,
        ) = ctx.const

        _backend.tile_based_vol_rendering_backward(
            mean,
            cov,
            color,
            alpha,
            offset,
            gaussian_ids,
            out,
            grad_mean,
            grad_cov,
            grad_color,
            grad_alpha,
            grad,
            topleft,
            tile_size,
            n_tiles_h,
            n_tiles_w,
            pixel_size_x,
            pixel_size_y,
            H,
            W,
            thresh,
        )

        return (
            grad_mean,
            grad_cov,
            grad_color,
            grad_alpha,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _render_v2(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        mean,
        cov,
        color,
        alpha,
        offset,
        gaussian_ids,
        topleft,
        tile_size,
        n_tiles_h,
        n_tiles_w,
        pixel_size_x,
        pixel_size_y,
        H,
        W,
        thresh,
    ):
        out = torch.zeros([H * W * 3], dtype=torch.float32, device=mean.device)
        _backend.tile_based_vol_rendering_v2(
            mean,
            cov,
            color,
            alpha,
            offset,
            gaussian_ids,
            out,
            topleft,
            tile_size,
            n_tiles_h,
            n_tiles_w,
            pixel_size_x,
            pixel_size_y,
            H,
            W,
            thresh,
        )
        ctx.save_for_backward(
            mean, cov, color, alpha, offset, gaussian_ids, out, topleft
        )
        ctx.const = [
            tile_size,
            n_tiles_h,
            n_tiles_w,
            pixel_size_x,
            pixel_size_y,
            H,
            W,
            thresh,
        ]

        return out

    @staticmethod
    def backward(ctx, grad):
        mean, cov, color, alpha, offset, gaussian_ids, out, topleft = ctx.saved_tensors
        grad_mean = torch.zeros_like(mean)
        grad_cov = torch.zeros_like(cov)
        grad_color = torch.zeros_like(color)
        grad_alpha = torch.zeros_like(alpha)
        (
            tile_size,
            n_tiles_h,
            n_tiles_w,
            pixel_size_x,
            pixel_size_y,
            H,
            W,
            thresh,
        ) = ctx.const

        _backend.tile_based_vol_rendering_backward(
            mean,
            cov,
            color,
            alpha,
            offset,
            gaussian_ids,
            out,
            grad_mean,
            grad_cov,
            grad_color,
            grad_alpha,
            grad,
            topleft,
            tile_size,
            n_tiles_h,
            n_tiles_w,
            pixel_size_x,
            pixel_size_y,
            H,
            W,
            thresh,
        )

        return (
            grad_mean,
            grad_cov,
            grad_color,
            grad_alpha,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
