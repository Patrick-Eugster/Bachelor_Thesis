#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
import gsplat
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from gaussians.scene.gaussian_model import GaussianModel
from gaussians.utils.sh_utils import eval_sh

from flashsplat_rasterization import GaussianRasterizationSettings as FlashSplat_GaussianRasterizationSettings
from flashsplat_rasterization import GaussianRasterizer as FlashSplat_GaussianRasterizer
import pdb

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene with gsplat instead of diff-gaussian-rasterization.

    gsplat takes the full camera intrinsic matrix K (with the real principal point cx/cy),
    so the pixel-shift fix is handled natively here — no asymmetric-frustum projmatrix needed.
    The old diff-gaussian path is kept below as render_diffgs() for A/B comparison.
    Background tensor (bg_color) must be on GPU!
    """
    device = pc.get_xyz.device
    W = int(viewpoint_camera.image_width)
    H = int(viewpoint_camera.image_height)

    # Build the intrinsic matrix K. fx/fy are recovered from the stored FoV (= the COLMAP focal),
    # cx/cy are the real principal point when use_principal_point set the Camera's cx/cy,
    # else default to the image center (= vanilla symmetric behaviour).
    fx = W / (2.0 * math.tan(viewpoint_camera.FoVx * 0.5))
    fy = H / (2.0 * math.tan(viewpoint_camera.FoVy * 0.5))
    # MiniCam (used by render_360 / viewer) has no cx/cy attribute → fall back to image center,
    # which is the symmetric behaviour those paths always had.
    cx = getattr(viewpoint_camera, "cx", None)
    cy = getattr(viewpoint_camera, "cy", None)
    cx = cx if cx is not None else W * 0.5
    cy = cy if cy is not None else H * 0.5
    K = torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], device=device, dtype=torch.float32)[None]

    # world_view_transform is stored TRANSPOSED for the glm/CUDA path; gsplat wants the
    # plain world-to-camera matrix, so undo the transpose.
    viewmat = viewpoint_camera.world_view_transform.transpose(0, 1).contiguous()[None]

    means = pc.get_xyz
    quats = pc.get_rotation                      # [N,4] wxyz — same convention as gsplat
    scales = pc.get_scaling * scaling_modifier
    opacities = pc.get_opacity.squeeze(-1)       # [N,1] -> [N]

    # Colors: pass SH coefficients and let gsplat do SH->RGB, unless an override colour is given.
    if override_color is None:
        colors = pc.get_features                 # [N, K, 3] SH coeffs
        sh_degree = pc.active_sh_degree
    else:
        colors = override_color
        sh_degree = None

    render_colors, render_alphas, meta = gsplat.rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmat,
        Ks=K,
        width=W,
        height=H,
        sh_degree=sh_degree,
        render_mode="RGB+ED",                    # also return expected depth, like the old renderer
        packed=False,                            # unpacked -> per-Gaussian [C,N,...] maps cleanly to [N]
        backgrounds=bg_color[None],
        near_plane=0.01,
        eps2d=0.3,                               # matches the old rasterizer's 0.3 screen-space blur
    )

    # render_colors: [1, H, W, 4] = RGB + expected-depth
    image = render_colors[0, ..., :3].permute(2, 0, 1)        # [3,H,W]
    depth = render_colors[0, ..., 3:4].permute(2, 0, 1)       # [1,H,W]
    alpha = render_alphas[0].permute(2, 0, 1)                 # [1,H,W]

    # means2d carries the screen-space gradient used for densification. gsplat's grad is in
    # PIXELS, but the INRIA densify_grad_threshold is calibrated for NDC, so rescale the incoming
    # grad by (W/2, H/2) to convert pixels -> NDC half-extent. Keeps densification identical.
    means2d = meta["means2d"]                    # [1, N, 2]
    # Only needed for densification during training; under torch.no_grad() (render/metrics/eval)
    # means2d has requires_grad=False, so retain_grad/hook would error — skip them.
    if means2d.requires_grad:
        means2d.retain_grad()
        def _to_ndc_grad(g):
            g = g.clone()
            g[..., 0] *= W * 0.5
            g[..., 1] *= H * 0.5
            return g
        means2d.register_hook(_to_ndc_grad)

    # radii: gsplat gives per-axis [1,N,2]; collapse to one per-Gaussian radius like INRIA.
    radii = meta["radii"][0].amax(dim=-1)        # [N]
    visibility_filter = radii > 0

    return {"render": image,
            "viewspace_points": means2d,
            "visibility_filter": visibility_filter,
            "radii": radii,
            "depth": depth,
            "alpha": alpha}


def render_diffgs(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Original diff-gaussian-rasterization render path (pre-gsplat). Kept for A/B comparison
    against the gsplat render() above — uses the asymmetric-frustum projmatrix (pixel-shift fix).

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "depth": rendered_depth,
            "alpha": rendered_alpha}


def flashsplat_render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, 
                  override_color = None, gt_mask = None, used_mask = None, unique_label = None, setpdb=False,
                  obj_num = 2,):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # if unique_label is not None:
    #     pdb.set_trace()

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = FlashSplat_GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        mask_grad=False,
        num_obj=obj_num,
    )

    rasterizer = FlashSplat_GaussianRasterizer(raster_settings=raster_settings)

    means2D = screenspace_points
    means3D = pc.get_xyz
    opacity = pc.get_opacity
    if used_mask is not None:
        means3D = means3D[used_mask]
        opacity = opacity[used_mask]

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
        if used_mask is not None:
            scales = scales[used_mask]
            rotations = rotations[used_mask]

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            # pdb.set_trace()
            shs = pc.get_features
            if used_mask is not None:
                shs = shs[used_mask]
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    if setpdb:
        pdb.set_trace()
    rendered_image, radii, depth, alpha, contrib_num, used_count, proj_xy, gs_depth = rasterizer(
        gt_mask = gt_mask,
        unique_label = unique_label,
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)


    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "alpha": alpha, 
            "depth": depth, 
            "contrib_num": contrib_num,
            "used_count": used_count,
            "proj_xy": proj_xy,
            "gs_depth": gs_depth}
