# latent_sbs.py  (Latent Stereo Camera)
#
# ComfyUI node: LatentStereoCamera
#
# Simulates a stereo camera pair (left/right) from a single generated image,
# using a depth map to create per-pixel horizontal disparity.

from typing import Dict, Any
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


class _DepthEstimatorStereo:
    """
    Helper: get depth via MiDaS_small if available, otherwise a luminance fallback.
    Depth convention inside this node: 1.0 = NEAR, 0.0 = FAR.
    """

    _device = None
    _model = None
    _transforms = None

    @classmethod
    def ensure_loaded(cls, device: str = "cpu"):
        if cls._model is not None:
            return
        if not TORCH_AVAILABLE:
            return
        try:
            import torch as _t

            cls._device = _t.device(device)
            cls._model = _t.hub.load("intel-isl/MiDaS", "MiDaS_small")
            cls._model.to(cls._device)
            cls._model.eval()
            transforms = _t.hub.load("intel-isl/MiDaS", "transforms")
            cls._transforms = transforms.small_transform
        except Exception:
            cls._model = None
            cls._transforms = None

    @classmethod
    def infer_depth(cls, image_bhwc):
        """
        image_bhwc: [B,H,W,C] float32 [0,1]
        Returns depth [B,1,H,W] in [0,1], where 1 = near, 0 = far.
        """
        if not TORCH_AVAILABLE:
            return cls._fallback_depth(image_bhwc)

        import torch as _t

        cls.ensure_loaded(device="cuda" if _t.cuda.is_available() else "cpu")
        if cls._model is None or cls._transforms is None:
            return cls._fallback_depth(image_bhwc)

        B, H, W, C = image_bhwc.shape
        outs = []
        with _t.no_grad():
            for b in range(B):
                img = image_bhwc[b].detach().cpu().numpy()
                img_rgb = (img * 255.0).clip(0, 255).astype(np.uint8)
                inp = cls._transforms(img_rgb).to(cls._device)
                pred = cls._model(inp)
                pred = _t.nn.functional.interpolate(
                    pred.unsqueeze(1),
                    size=(H, W),
                    mode="bicubic",
                    align_corners=False,
                ).squeeze(1)
                outs.append(pred)
        depth = _t.stack(outs, dim=0)  # [B,H,W]
        dmin = depth.amin(dim=(1, 2), keepdim=True)
        dmax = depth.amax(dim=(1, 2), keepdim=True)
        depth = (depth - dmin) / (dmax - dmin + 1e-8)
        depth = depth.unsqueeze(1)  # [B,1,H,W]
        # MiDaS is inverse-ish, so invert: 1 = near, 0 = far
        depth = 1.0 - depth
        return depth

    @classmethod
    def _fallback_depth(cls, image_bhwc):
        """
        Fallback: approximate depth from inverted luminance.
        Brighter = far, darker = near (so we invert to get 1=near).
        """
        if TORCH_AVAILABLE and isinstance(image_bhwc, torch.Tensor):
            r = image_bhwc[..., 0]
            g = image_bhwc[..., 1]
            b = image_bhwc[..., 2]
            gray = 0.2126 * r + 0.7152 * g + 0.0722 * b
            depth = 1.0 - gray
            depth = depth.unsqueeze(1)  # [B,1,H,W]
            return depth
        else:
            img = np.asarray(image_bhwc)
            gray = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
            depth = 1.0 - gray
            depth = depth[None, None, ...]
            return depth


def _build_uv_identity(h: int, w: int, device):
    """
    Build a base UV identity grid for rectified images:
    uv in [0,1], shape [1,h,w,2].
    """
    ys = torch.linspace(0.0, 1.0, steps=h, device=device)
    xs = torch.linspace(0.0, 1.0, steps=w, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")  # [H,W]
    uv = torch.stack([xx, yy], dim=-1)  # [H,W,2] -> (u,v)
    return uv.unsqueeze(0)  # [1,H,W,2]


def _uv_to_grid(uv):
    """
    Convert UV in [0,1] to grid_sample coords in [-1,1].
    uv: [B,H,W,2]
    """
    u = uv[..., 0]
    v = uv[..., 1]
    xg = u * 2.0 - 1.0
    yg = v * 2.0 - 1.0
    return torch.stack([xg, yg], dim=-1)


class LatentStereoCamera:
    """
    LatentStereoCamera

    Simulates a stereo camera pair from a single view using depth.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                # Image is always required
                "image": ("IMAGE", {}),

                # stereo controls
                "ipd_norm": (
                    "FLOAT",
                    {
                        "default": 0.06,
                        "min": 0.0,
                        "max": 0.3,
                        "step": 0.001,
                    },
                ),
                "depth_strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 3.0,
                        "step": 0.01,
                    },
                ),
                "max_disp_norm": (
                    "FLOAT",
                    {
                        "default": 0.08,
                        "min": 0.0,
                        "max": 0.5,
                        "step": 0.001,
                    },
                ),
                "fg_boost": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 3.0,
                        "step": 0.01,
                    },
                ),
                "bg_boost": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 3.0,
                        "step": 0.01,
                    },
                ),

                # TRUE dropdown: first element is list-of-choices
                "stereo_preset": (
                    [
                        "Default",
                        "Strong_Pop",
                        "Cinematic",
                        "Background_Depth",
                        "Portrait",
                        "Focus_Lock",
                        "Reverse_Parallax",
                        "VR180_Prep",
                    ],
                    {"default": "Focus_Lock"},
                ),

                # TRUE dropdown for focus mode
                "focus_mode": (
                    ["Median", "Manual"],
                    {"default": "Median"},
                ),

                "manual_focus": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                    },
                ),
                "use_internal_depth": ("BOOLEAN", {"default": False}),
                "invert_internal_depth": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                # latent is now optional
                "latent": ("LATENT", {}),
                "opt_depth": ("IMAGE", {}),
                "invert_external_depth": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("LATENT", "LATENT", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = (
        "latent_left",
        "latent_right",
        "image_left",
        "image_right",
        "depth_preview",
        "sbs_image",
        "text_debug",
    )
    FUNCTION = "run"
    CATEGORY = "🟢 3DVidTools"

    def run(
        self,
        image,
        ipd_norm: float,
        depth_strength: float,
        max_disp_norm: float,
        fg_boost: float,
        bg_boost: float,
        stereo_preset: str,
        focus_mode: str,
        manual_focus: float,
        use_internal_depth: bool,
        invert_internal_depth: bool,
        latent=None,
        opt_depth=None,
        invert_external_depth: bool = False,
    ):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for LatentStereoCamera node.")

        # --- decide device based on latent or CUDA/CPU ---
        if latent is not None and isinstance(latent, dict) and "samples" in latent:
            lat_tensor = latent["samples"]
            if not isinstance(lat_tensor, torch.Tensor):
                raise ValueError("latent['samples'] must be a torch.Tensor.")
            if lat_tensor.is_cuda:
                device = lat_tensor.device
            else:
                device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        else:
            lat_tensor = None
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # --- image tensor ---
        if not isinstance(image, torch.Tensor):
            img = torch.from_numpy(np.asarray(image)).float()
        else:
            img = image.float()

        img = img.to(device)
        B_img, H_img, W_img, C_img = img.shape

        # --- latent (optional) ---
        if lat_tensor is not None:
            lat_tensor = lat_tensor.to(device)
            B_lat, C_lat, H_lat, W_lat = lat_tensor.shape

            if B_lat != B_img:
                raise ValueError(
                    f"Batch size mismatch: latent batch={B_lat}, image batch={B_img}"
                )
        else:
            # Image-only mode: create a dummy latent matching a typical VAE downscale
            C_lat = 4
            H_lat = max(1, H_img // 8)
            W_lat = max(1, W_img // 8)
            B_lat = B_img
            lat_tensor = torch.zeros((B_lat, C_lat, H_lat, W_lat), device=device)

        # ---- apply stereo preset (runtime overrides only) ----
        preset_key = (stereo_preset or "Default").lower()

        if preset_key == "strong_pop":
            ipd_norm = 0.12
            depth_strength = 1.4
            max_disp_norm = 0.12
            fg_boost = 1.8
            bg_boost = 0.9
            focus_mode = "Median"

        elif preset_key == "cinematic":
            ipd_norm = 0.05
            depth_strength = 0.9
            max_disp_norm = 0.06
            fg_boost = 1.1
            bg_boost = 1.1
            focus_mode = "Median"

        elif preset_key == "background_depth":
            ipd_norm = 0.06
            depth_strength = 1.0
            max_disp_norm = 0.10
            fg_boost = 0.2
            bg_boost = 2.2
            focus_mode = "Median"

        elif preset_key == "portrait":
            ipd_norm = 0.04
            depth_strength = 0.7
            max_disp_norm = 0.05
            fg_boost = 0.9
            bg_boost = 1.2
            focus_mode = "Manual"
            manual_focus = 0.70

        elif preset_key == "focus_lock":
            ipd_norm = 0.06
            depth_strength = 1.0
            max_disp_norm = 0.10
            fg_boost = 2.0
            bg_boost = 1.0
            focus_mode = "Manual"
            manual_focus = 0.35

        elif preset_key == "reverse_parallax":
            max_disp_norm = 0.10
            fg_boost = 0.8
            bg_boost = 1.6
            invert_internal_depth = True
            # NOTE: no longer overriding invert_external_depth here

        elif preset_key == "vr180_prep":
            ipd_norm = 0.09
            depth_strength = 1.3
            max_disp_norm = 0.15
            focus_mode = "Manual"
            manual_focus = 0.55
            invert_internal_depth = False

        # --- depth source (image resolution) ---
        if opt_depth is not None and not use_internal_depth:
            d_src = opt_depth
            if not isinstance(d_src, torch.Tensor):
                d_src = torch.from_numpy(np.asarray(d_src)).float()
            d_src = d_src.to(device)

            if d_src.dim() == 4 and d_src.shape[3] == 1:
                d_src = d_src.permute(0, 3, 1, 2)
            elif d_src.dim() == 4 and d_src.shape[3] == 3:
                d_src = (
                    0.2126 * d_src[..., 0:1]
                    + 0.7152 * d_src[..., 1:2]
                    + 0.0722 * d_src[..., 2:3]
                ).permute(0, 3, 1, 2)
            elif d_src.dim() == 3:
                d_src = d_src.unsqueeze(1)

            dmin = d_src.amin(dim=(2, 3), keepdim=True)
            dmax = d_src.amax(dim=(2, 3), keepdim=True)
            depth_src = (d_src - dmin) / (dmax - dmin + 1e-8)

            if invert_external_depth:
                depth_src = 1.0 - depth_src
        else:
            depth_src = _DepthEstimatorStereo.infer_depth(img)
            depth_src = depth_src.to(device)

        dmin_s = depth_src.amin(dim=(2, 3), keepdim=True)
        dmax_s = depth_src.amax(dim=(2, 3), keepdim=True)
        depth_src = (depth_src - dmin_s) / (dmax_s - dmin_s + 1e-8)

        if use_internal_depth and invert_internal_depth:
            depth_src = 1.0 - depth_src

        # --- choose focus depth ---
        if focus_mode.lower() == "median":
            focus = depth_src.median(dim=2, keepdim=True).values.median(
                dim=3, keepdim=True
            ).values
        else:
            focus = torch.full_like(depth_src[:, :, :1, :1], manual_focus)

        # --- disparity at image resolution ---
        delta_img = focus - depth_src
        is_fg_img = depth_src > focus
        disp_img = torch.where(is_fg_img, delta_img * fg_boost, delta_img * bg_boost)

        disp_img = disp_img * depth_strength
        disp_img = disp_img.clamp(-1.0, 1.0)
        disp_img = disp_img * (ipd_norm * max_disp_norm)

        uv_img_base = _build_uv_identity(H_img, W_img, device)
        uv_img_base = uv_img_base.expand(B_img, -1, -1, -1)

        u_base_img = uv_img_base[..., 0]
        v_base_img = uv_img_base[..., 1]

        disp_hw_img = disp_img.squeeze(1)

        u_L_img = u_base_img - disp_hw_img / 2.0
        u_R_img = u_base_img + disp_hw_img / 2.0

        uv_L_img = torch.stack([u_L_img, v_base_img], dim=-1)
        uv_R_img = torch.stack([u_R_img, v_base_img], dim=-1)

        grid_L_img = _uv_to_grid(uv_L_img)
        grid_R_img = _uv_to_grid(uv_R_img)

        img_nchw = img.permute(0, 3, 1, 2)

        img_L = F.grid_sample(
            img_nchw,
            grid_L_img,
            mode="bicubic",
            padding_mode="zeros",
            align_corners=True,
        )
        img_R = F.grid_sample(
            img_nchw,
            grid_R_img,
            mode="bicubic",
            padding_mode="zeros",
            align_corners=True,
        )

        image_left = img_L.permute(0, 2, 3, 1)
        image_right = img_R.permute(0, 2, 3, 1)

        # --- disparity at latent resolution ---
        depth_lat = F.interpolate(
            depth_src,
            size=(H_lat, W_lat),
            mode="bilinear",
            align_corners=False,
        )

        delta_lat = focus - depth_lat
        is_fg_lat = depth_lat > focus

        disp_lat = torch.where(is_fg_lat, delta_lat * fg_boost, delta_lat * bg_boost)
        disp_lat = disp_lat * depth_strength
        disp_lat = disp_lat.clamp(-1.0, 1.0)
        disp_lat = disp_lat * (ipd_norm * max_disp_norm)

        uv_lat_base = _build_uv_identity(H_lat, W_lat, device)
        uv_lat_base = uv_lat_base.expand(B_lat, -1, -1, -1)

        u_base_lat = uv_lat_base[..., 0]
        v_base_lat = uv_lat_base[..., 1]

        disp_hw_lat = disp_lat.squeeze(1)

        u_L_lat = u_base_lat - disp_hw_lat / 2.0
        u_R_lat = u_base_lat + disp_hw_lat / 2.0

        uv_L_lat = torch.stack([u_L_lat, v_base_lat], dim=-1)
        uv_R_lat = torch.stack([u_R_lat, v_base_lat], dim=-1)

        grid_L_lat = _uv_to_grid(uv_L_lat)
        grid_R_lat = _uv_to_grid(uv_R_lat)

        lat_L = F.grid_sample(
            lat_tensor,
            grid_L_lat,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        lat_R = F.grid_sample(
            lat_tensor,
            grid_R_lat,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

        # Build latent dicts: preserve original dict if provided, otherwise minimal dict
        if latent is not None and isinstance(latent, dict) and "samples" in latent:
            latent_left = dict(latent)
            latent_left["samples"] = lat_L

            latent_right = dict(latent)
            latent_right["samples"] = lat_R
        else:
            latent_left = {"samples": lat_L}
            latent_right = {"samples": lat_R}

        depth_preview = depth_src.repeat(1, 3, 1, 1).permute(0, 2, 3, 1)

        sbs_image = torch.cat([image_left, image_right], dim=2)

        # --- text_debug summary ---
        preset_is_default = (stereo_preset == "Default")

        if preset_is_default:
            ipd_str = f"{ipd_norm:.3f}"
            deps_str = f"{depth_strength:.3f}"
            maxd_str = f"{max_disp_norm:.3f}"
            mfoc_str = f"{manual_focus:.3f}"
            intd_str = "1" if use_internal_depth else "0"
            invext_str = "1" if invert_external_depth else "0"
            invint_str = "1" if invert_internal_depth else "0"
            fg_str = f"{fg_boost:.3f}"
            bg_str = f"{bg_boost:.3f}"

            text_debug = (
                f"_IPD_{ipd_str}"
                f"_DepS_{deps_str}"
                f"_MaxD_{maxd_str}"
                f"_Preset_{stereo_preset}"
                f"_FMode_{focus_mode}"
                f"_MFoc_{mfoc_str}"
                f"_IntD_{intd_str}"
                f"_InvExt_{invext_str}"
                f"_InvInt_{invint_str}"
                f"_FgB_{fg_str}"
                f"_BgB_{bg_str}"
            )
        else:
            # For any preset other than Default, just output the preset name.
            text_debug = stereo_preset

        return (
            latent_left,
            latent_right,
            image_left,
            image_right,
            depth_preview,
            sbs_image,
            text_debug,
        )
