import numpy as np
import torch

try:
    import cv2
except Exception:
    cv2 = None


def _ensure_bhwc(img_t: torch.Tensor) -> torch.Tensor:
    if img_t is None:
        raise ValueError("Image is None")
    if img_t.ndim != 4 or img_t.shape[-1] != 3:
        raise ValueError(f"Expected IMAGE tensor [B,H,W,3], got {tuple(img_t.shape)}")
    return img_t


def _to_numpy_bhwc(img_t: torch.Tensor) -> np.ndarray:
    # [B,H,W,3] float 0..1
    img_t = _ensure_bhwc(img_t)
    img = img_t.detach().cpu().numpy().astype(np.float32)
    return np.clip(img, 0.0, 1.0)


def _depth_to_hw(depth_t: torch.Tensor, h: int, w: int) -> np.ndarray:
    """
    depth_t accepted shapes:
      - IMAGE: [B,H,W,3]
      - MASK-like: [B,H,W] or [H,W]
    returns: [B,H,W] float32 0..1
    """
    if depth_t is None:
        raise ValueError("Depth input is required.")

    dt = depth_t.detach().cpu()

    # Normalize into [B,H,W]
    if dt.ndim == 4 and dt.shape[-1] == 3:
        d = dt.numpy().astype(np.float32)  # [B,H,W,3]
        d = 0.2126 * d[..., 0] + 0.7152 * d[..., 1] + 0.0722 * d[..., 2]
    elif dt.ndim == 3:
        d = dt.numpy().astype(np.float32)  # [B,H,W]
    elif dt.ndim == 2:
        d = dt.numpy().astype(np.float32)[None, ...]  # [1,H,W]
    else:
        raise ValueError(f"Unsupported depth shape: {tuple(depth_t.shape)}")

    d = np.nan_to_num(d, nan=0.0, posinf=1.0, neginf=0.0)
    d = np.clip(d, 0.0, 1.0)

    # Resize per-batch if needed
    if d.shape[1] != h or d.shape[2] != w:
        if cv2 is None:
            # nearest fallback
            out = np.zeros((d.shape[0], h, w), dtype=np.float32)
            for i in range(d.shape[0]):
                yy = (np.linspace(0, d.shape[1] - 1, h)).astype(np.int32)
                xx = (np.linspace(0, d.shape[2] - 1, w)).astype(np.int32)
                out[i] = d[i][yy[:, None], xx[None, :]]
            d = out
        else:
            out = np.zeros((d.shape[0], h, w), dtype=np.float32)
            for i in range(d.shape[0]):
                out[i] = cv2.resize(d[i], (w, h), interpolation=cv2.INTER_LINEAR)
            d = out

    return d.astype(np.float32)


def _gaussian_blur_depth_batch(d_bhw: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.0:
        return d_bhw
    if cv2 is None:
        # conservative box blur fallback
        k = max(3, int(round(sigma * 3)) | 1)
        pad = k // 2
        out = np.zeros_like(d_bhw)
        area = float(k * k)
        for i in range(d_bhw.shape[0]):
            d = d_bhw[i]
            dp = np.pad(d, ((pad, pad), (pad, pad)), mode="edge")
            # naive sliding sum (ok for small sigmas)
            for y in range(d.shape[0]):
                for x in range(d.shape[1]):
                    out[i, y, x] = dp[y:y + k, x:x + k].sum() / area
        return out
    else:
        out = np.zeros_like(d_bhw)
        for i in range(d_bhw.shape[0]):
            out[i] = cv2.GaussianBlur(d_bhw[i], (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)
        return out


def _auto_contrast_depth(d_hw: np.ndarray, low_p: float = 2.0, high_p: float = 98.0) -> np.ndarray:
    """
    Conservative percentile stretch to ensure effect is visible even if depth is flat.
    Operates per-frame (no temporal logic).
    """
    lo = np.percentile(d_hw, low_p)
    hi = np.percentile(d_hw, high_p)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1e-6:
        return d_hw
    out = (d_hw - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _build_center_pull_map(h: int, w: int):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx = (w - 1) * 0.5
    cy = (h - 1) * 0.5
    vx = cx - xx
    vy = cy - yy
    mag = np.sqrt(vx * vx + vy * vy) + 1e-6
    ux = vx / mag
    uy = vy / mag
    return ux, uy


def _warp_one(img_hw3: np.ndarray, depth_hw: np.ndarray, strength_px: float, max_px: float, ux: np.ndarray, uy: np.ndarray):
    h, w = depth_hw.shape

    # displacement magnitude: background (low depth) moves more
    disp = strength_px * (1.0 - depth_hw)
    disp = np.clip(disp, 0.0, max_px).astype(np.float32)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x = xx - (ux * disp)
    map_y = yy - (uy * disp)

    map_x = np.clip(map_x, 0.0, w - 1).astype(np.float32)
    map_y = np.clip(map_y, 0.0, h - 1).astype(np.float32)

    if cv2 is None:
        ix = np.round(map_x).astype(np.int32)
        iy = np.round(map_y).astype(np.int32)
        return img_hw3[iy, ix].astype(np.float32)

    return cv2.remap(
        img_hw3,
        map_x,
        map_y,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    ).astype(np.float32)


class Pop3DClass:
    """
    """

    PRESETS = {
        # base_strength_px, max_px, depth_blur_sigma, bar_ratio
        "FaceSafe": (8.0, 14.0, 1.1, 0.14),
        "Subtle":   (10.0, 16.0, 1.0, 0.14),
        "Medium":   (12.0, 18.0, 1.0, 0.16),
        "Strong":   (14.0, 22.0, 1.1, 0.18),
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "depth": ("IMAGE",),
                "preset": (list(cls.PRESETS.keys()), {
                    "default": "FaceSafe",
                    "tooltip": "Selects a tuned stability profile. FaceSafe prioritizes clean edges and stable faces; stronger presets increase depth separation."
                }),
"strength": ("FLOAT", {
                    "default": 0.80, "min": 0.0, "max": 2.0, "step": 0.01,
                    "tooltip": "Overall depth strength. Lower = subtler/flatter. Higher = stronger window depth, but extreme values can exaggerate edges."
                }),
"depth_cutoff": ("FLOAT", {
                    "default": -0.50, "min": -1.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Optional foreground cutoff. Below 0 disables. Increasing focuses the effect on nearer regions and can improve face stability."
                }),
                "bars": (["Overlay", "Breakout (Near Over Bars)"], {
                    "default": "Breakout (Near Over Bars)",
                    "tooltip": "Overlay = permanent breakout (bars always on, near elements can appear in the bars). Breakout = dynamic breakout with optional collapsing/smoothing."
                }),
                "breakout_cutoff": ("FLOAT", {
                    "default": 0.40, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Global breakout threshold (both bars). Lower = more breakout. Higher = cleaner, only closest parts."
                }),
                "halo_fix": ("FLOAT", {
                    "default": 0.85, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Reduces bright/dark outlines on breakout edges by stabilizing the edge color. 0 = off. Higher = cleaner edges."
                }),
                "Bar_Smooth": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "When enabled (video batches only), the top bar will smoothly appear/disappear over time based on upcoming breakout overlap (lookahead/fade via spinners)."
                }),
                "Bar_Smooth_Lookahead": ("INT", {
                    "default": 24, "min": 8, "max": 64, "step": 1,
                    "tooltip": "Frames to look ahead for upcoming breakout overlap when Bar_Smooth is enabled (video batches only)."
                }),
                "Bar_Smooth_Fade": ("INT", {
                    "default": 16, "min": 8, "max": 64, "step": 1,
                    "tooltip": "Frames to fade out after breakout overlap ends when Bar_Smooth is enabled (video batches only)."
                }),
                "separate_bars": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable to control Top and Bottom bars independently (Breakout Cutoff + Halo Fix)."
                }),
                "Top_Bar_Off": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Disable the TOP bar entirely (no black bar and no top breakout). Works in both Overlay and Breakout modes."
                }),
                "breakout_cutoff_top": ("FLOAT", {
                    "default": 0.40, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Top bar breakout threshold (used only if Separate Bars is enabled)."
                }),
                "halo_fix_top": ("FLOAT", {
                    "default": 0.85, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Top bar halo cleanup (used only if Separate Bars is enabled)."
                }),
                "Top_AutoCancel": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "When enabled (Breakout mode only), if the bottom edge of the TOP bar has too little 'popped' (near) coverage, the top bar is temporarily cancelled for that frame to avoid tiny hairline pops triggering a bar."
                }),
                "Top_AutoCancel_StripRows": ("INT", {
                    "default": 5, "min": 1, "max": 64, "step": 1,
                    "tooltip": "How many rows at the bottom edge of the TOP bar region to test for 'popped' coverage when Top_AutoCancel is enabled."
                }),
                "Top_AutoCancel_CoverageMin": ("FLOAT", {
                    "default": 0.10, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "If < this fraction of pixels in the TOP bar edge strip are 'popped' (near), the top bar is cancelled for that frame (Breakout mode only)."
                }),
                "Bottom_Bar_Off": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Disable the BOTTOM bar entirely (no black bar and no bottom breakout). Works in both Overlay and Breakout modes."
                }),
                "breakout_cutoff_bottom": ("FLOAT", {
                    "default": 0.40, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Bottom bar breakout threshold (used only if Separate Bars is enabled)."
                }),
                "halo_fix_bottom": ("FLOAT", {
                    "default": 0.85, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Bottom bar halo cleanup (used only if Separate Bars is enabled)."
                }),
                "Bottom_AutoCancel": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "When enabled (Breakout mode only), if the bottom strip is mostly 'popped' (near) the bottom bar is temporarily cancelled for that frame to avoid black ground gaps."
                }),
                "Bottom_AutoCancel_StripRows": ("INT", {
                    "default": 5, "min": 1, "max": 64, "step": 1,
                    "tooltip": "How many rows at the very bottom of the frame to test for 'popped' coverage when Bottom_AutoCancel is enabled."
                }),
                "Bottom_AutoCancel_Coverage": ("FLOAT", {
                    "default": 0.90, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "If >= this fraction of pixels in the bottom strip are 'popped' (near), the bottom bar is cancelled for that frame (Breakout mode only)."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = "🟢 3DVidTools"

    def run(self, image, depth, preset, strength, depth_cutoff, bars, breakout_cutoff, halo_fix, separate_bars, breakout_cutoff_top, breakout_cutoff_bottom, halo_fix_top, halo_fix_bottom, Bar_Smooth, Bar_Smooth_Lookahead, Bar_Smooth_Fade, Top_Bar_Off, Bottom_Bar_Off, Bottom_AutoCancel, Bottom_AutoCancel_StripRows, Bottom_AutoCancel_Coverage, Top_AutoCancel, Top_AutoCancel_StripRows, Top_AutoCancel_CoverageMin):
        img_bhwc = _to_numpy_bhwc(image)
        b, h, w, _ = img_bhwc.shape

        d_bhw = _depth_to_hw(depth, h, w)
        # Support depth batch of 1 or B
        if d_bhw.shape[0] not in (1, b):
            # If mismatch, fall back to first depth for all frames
            d_bhw = d_bhw[:1]

        base_strength_px, max_px, blur_sigma, bar_ratio = self.PRESETS[preset]
        strength_px = float(base_strength_px) * float(strength)

        # Precompute direction field once
        ux, uy = _build_center_pull_map(h, w)

        out = np.zeros_like(img_bhwc, dtype=np.float32)

        # Optional bar smoothing (video batches only): precompute per-frame TOP/BOTTOM bar height envelopes.
        # lookahead/fade are controlled by spinners. Top grows top-down; Bottom grows bottom-up.
        # IMPORTANT: uses already-preprocessed depth 'd_proc_cache' to avoid double-processing (which can make the bar always-on).
        top_h_override = None
        bot_h_override = None
        d_proc_cache = None

        if bool(Bar_Smooth) and (bars == "Breakout (Near Over Bars)") and (b > 1):
            try:
                LA = int(np.clip(int(Bar_Smooth_Lookahead), 8, 64))
                FD = int(np.clip(int(Bar_Smooth_Fade), 8, 64))
                bar_h0 = int(round(h * float(bar_ratio)))
                bar_h0 = max(0, min(h, bar_h0))

                if bar_h0 > 0:
                    # Determine top breakout cutoff once (matches per-frame logic)
                    cut_g0 = float(np.clip(breakout_cutoff, 0.0, 1.0))
                    if bool(separate_bars):
                        cut_t0 = float(np.clip(breakout_cutoff_top, 0.0, 1.0))
                        cut_b0 = float(np.clip(breakout_cutoff_bottom, 0.0, 1.0))
                    else:
                        cut_t0 = cut_g0
                        cut_b0 = cut_g0

                    pop_t = np.zeros((b,), dtype=np.bool_)
                    pop_b = np.zeros((b,), dtype=np.bool_)
                    d_proc_cache = [None] * b

                    for ii in range(b):
                        d0 = d_bhw[ii if d_bhw.shape[0] == b else 0]

                        # Optional cutoff anchoring (smooth ramp)
                        if depth_cutoff is not None and depth_cutoff >= 0.0:
                            t0 = np.clip((d0 - depth_cutoff) / max(1e-6, (1.0 - depth_cutoff)), 0.0, 1.0)
                            d0 = np.maximum(d0, depth_cutoff + t0 * (1.0 - depth_cutoff))

                        # Auto-contrast depth per frame (conservative)
                        d0 = _auto_contrast_depth(d0)

                        # Blur per-frame (cv2 path)
                        if blur_sigma > 0 and cv2 is not None:
                            d0 = cv2.GaussianBlur(d0, (0, 0), sigmaX=float(blur_sigma), sigmaY=float(blur_sigma), borderType=cv2.BORDER_REPLICATE)

                        d0 = np.clip(d0, 0.0, 1.0).astype(np.float32)

                        d_proc_cache[ii] = d0
                        pop_t[ii] = (d0[:bar_h0, :] >= cut_t0).any()
                        pop_b[ii] = (d0[h - bar_h0:, :] >= cut_b0).any()
                        # If smoothing is enabled, fold top/bottom AutoCancel rules into the pop presence signal so the bar fades instead of snapping.
                        if bool(Top_Bar_Off):
                            pop_t[ii] = False
                        if bool(Bottom_Bar_Off):
                            pop_b[ii] = False
                        if bool(Top_AutoCancel) and (not bool(Top_Bar_Off)):
                            sr_t = int(Top_AutoCancel_StripRows) if Top_AutoCancel_StripRows is not None else 5
                            sr_t = max(1, min(int(sr_t), bar_h0))
                            thr_t = float(Top_AutoCancel_CoverageMin) if Top_AutoCancel_CoverageMin is not None else 0.10
                            thr_t = float(np.clip(thr_t, 0.0, 1.0))
                            y0 = max(0, bar_h0 - sr_t)
                            top_edge_mask0 = (d0[y0:bar_h0, :] >= cut_t0)
                            cov_t0 = float(top_edge_mask0.mean()) if top_edge_mask0.size > 0 else 0.0
                            if cov_t0 < thr_t:
                                pop_t[ii] = False
                        if False and bool(Bottom_AutoCancel) and (not bool(Bottom_Bar_Off)):
                            sr = int(Bottom_AutoCancel_StripRows) if Bottom_AutoCancel_StripRows is not None else 5
                            sr = max(1, min(int(sr), h))
                            thr = float(Bottom_AutoCancel_Coverage) if Bottom_AutoCancel_Coverage is not None else 0.90
                            thr = float(np.clip(thr, 0.0, 1.0))
                            bottom_strip_mask0 = (d0[h - sr:, :] >= cut_b0)
                            cov0 = float(bottom_strip_mask0.mean()) if bottom_strip_mask0.size > 0 else 0.0
                            if cov0 >= thr:
                                pop_b[ii] = False
                    # Presence envelopes: max(future ramp-in, past fade-out) for TOP and BOTTOM
                    def _presence_from_pop(pop_arr: np.ndarray, LAf: int, FDf: int) -> np.ndarray:
                        pres = np.zeros((b,), dtype=np.float32)
                        for ii in range(b):
                            fut = None
                            for k in range(0, LAf + 1):
                                jj = ii + k
                                if jj < b and bool(pop_arr[jj]):
                                    fut = k
                                    break
                            v_f = 0.0 if fut is None else max(0.0, (LAf - float(fut)) / float(LAf))

                            prv = None
                            for k in range(0, FDf + 1):
                                jj = ii - k
                                if jj >= 0 and bool(pop_arr[jj]):
                                    prv = k
                                    break
                            v_p = 0.0 if prv is None else max(0.0, (FDf - float(prv)) / float(FDf))

                            pres[ii] = max(v_f, v_p)
                        return pres

                    presence_t = _presence_from_pop(pop_t, LA, FD)
                    presence_b = _presence_from_pop(pop_b, LA, FD)

                    top_h_override = np.clip(np.round(presence_t * float(bar_h0)), 0.0, float(bar_h0)).astype(np.int32)
                    bot_h_override = np.clip(np.round(presence_b * float(bar_h0)), 0.0, float(bar_h0)).astype(np.int32)

                else:
                    top_h_override = np.zeros((b,), dtype=np.int32)
                    bot_h_override = np.zeros((b,), dtype=np.int32)
            except Exception:
                top_h_override = None
                bot_h_override = None
                d_proc_cache = None

        # Per-frame processing (no temporal state)
        for i in range(b):
            img = img_bhwc[i]
            if d_proc_cache is not None and d_proc_cache[i] is not None:
                d = d_proc_cache[i]
            else:
                d = d_bhw[i if d_bhw.shape[0] == b else 0]

            if d_proc_cache is None or d_proc_cache[i] is None:
                # Optional cutoff anchoring (smooth ramp)
                if depth_cutoff is not None and depth_cutoff >= 0.0:
                    t = np.clip((d - depth_cutoff) / max(1e-6, (1.0 - depth_cutoff)), 0.0, 1.0)
                    d = np.maximum(d, depth_cutoff + t * (1.0 - depth_cutoff))

                # Auto-contrast depth per frame (conservative)
                d = _auto_contrast_depth(d)

                # Depth smoothing (noise suppression only)
                if blur_sigma > 0:
                    if cv2 is None:
                        # defer to batch blur later for fallback simplicity
                        pass

                # Blur per-frame (cv2 path)
                if blur_sigma > 0 and cv2 is not None:
                    d = cv2.GaussianBlur(d, (0, 0), sigmaX=float(blur_sigma), sigmaY=float(blur_sigma), borderType=cv2.BORDER_REPLICATE)

                d = np.clip(d, 0.0, 1.0).astype(np.float32)

            warped = _warp_one(img, d, strength_px=strength_px, max_px=float(max_px), ux=ux, uy=uy)

            bar_h = int(round(h * float(bar_ratio)))
            if bar_h > 0:
                # Breakout-style bars (Overlay = permanent breakout; Breakout = dynamic/collapsing with optional smoothing)
                    permanent_breakout = (bars == "Overlay")
                    # Breakout: keep near pixels visible in bar area; far stays black
                    cut_g = float(np.clip(breakout_cutoff, 0.0, 1.0))
                    if bool(separate_bars):
                        cut_t = float(np.clip(breakout_cutoff_top, 0.0, 1.0))
                        cut_b = float(np.clip(breakout_cutoff_bottom, 0.0, 1.0))
                    else:
                        cut_t = cut_g
                        cut_b = cut_g
                    top_mask = (d[:bar_h, :] >= cut_t)
                    bot_mask = (d[h - bar_h:, :] >= cut_b)
                    # Determine active bar heights (Overlay = permanent breakout; Breakout = collapse/smooth)
                    if permanent_breakout:
                        top_h = bar_h
                        bot_h = bar_h
                    else:
                        # If there's no overlap into the TOP/BOTTOM bar at all, collapse it (or use smoothed height if enabled).
                        if top_h_override is not None:
                            top_h = int(top_h_override[i])
                        else:
                            top_h = bar_h if top_mask.any() else 0

                        if bot_h_override is not None:
                            bot_h = int(bot_h_override[i])
                        else:
                            bot_h = bar_h if bot_mask.any() else 0


                    if bool(Top_Bar_Off):
                        top_h = 0
                    if bool(Bottom_Bar_Off):
                        bot_h = 0
                    # Bottom bar auto-cancel: if the very bottom strip is mostly "popped" (near), cancel the bottom bar for this frame.
                    # This avoids common ground-plane black gaps when the bottom region is nearly all near depth but not perfectly uniform.
                    if (not permanent_breakout) and (not bool(Bottom_Bar_Off)) and bool(Bottom_AutoCancel) and (bot_h > 0):
                        sr = int(Bottom_AutoCancel_StripRows) if Bottom_AutoCancel_StripRows is not None else 5
                        sr = max(1, min(int(sr), h))
                        thr = float(Bottom_AutoCancel_Coverage) if Bottom_AutoCancel_Coverage is not None else 0.90
                        thr = float(np.clip(thr, 0.0, 1.0))
                        bottom_strip_mask = (d[h - sr:, :] >= cut_b)
                        cov = float(bottom_strip_mask.mean()) if bottom_strip_mask.size > 0 else 0.0
                        if cov >= thr:
                            bot_h = 0
                    # Top bar auto-cancel: if the bottom edge strip of the TOP bar has too little "popped" (near), cancel the top bar for this frame.
                    # This avoids tiny hairline breakout (e.g., a few pixels of hair) causing the top bar to appear.
                    if (not permanent_breakout) and (not bool(Top_Bar_Off)) and bool(Top_AutoCancel) and (top_h > 0) and (top_h_override is None):
                        sr_t = int(Top_AutoCancel_StripRows) if Top_AutoCancel_StripRows is not None else 5
                        sr_t = max(1, min(int(sr_t), int(top_h)))
                        thr_t = float(Top_AutoCancel_CoverageMin) if Top_AutoCancel_CoverageMin is not None else 0.10
                        thr_t = float(np.clip(thr_t, 0.0, 1.0))
                        top_edge_mask = (d[top_h - sr_t:top_h, :] >= cut_t)
                        cov_t = float(top_edge_mask.mean()) if top_edge_mask.size > 0 else 0.0
                        if cov_t < thr_t:
                            top_h = 0



                    # start from black bars (apply only within current bar heights)
                    if top_h > 0:
                        warped[:top_h, :, :] *= top_mask[:top_h, ...][..., None].astype(np.float32)
                    if bot_h > 0:
                        warped[h - bot_h:, :, :] *= bot_mask[bar_h - bot_h:, ...][..., None].astype(np.float32)

                    # Halo Fix (optional): recolor only the 1px rim of the breakout mask by pulling RGB from just inside the window.
                    hf_g = float(np.clip(halo_fix, 0.0, 1.0))
                    if bool(separate_bars):
                        hf_t = float(np.clip(halo_fix_top, 0.0, 1.0))
                        hf_b = float(np.clip(halo_fix_bottom, 0.0, 1.0))
                    else:
                        hf_t = hf_g
                        hf_b = hf_g
                    if (hf_t > 0.0) or (hf_b > 0.0):
                        def _erode1(mask_bool: np.ndarray) -> np.ndarray:
                            if cv2 is not None:
                                k = np.ones((3, 3), np.uint8)
                                return (cv2.erode(mask_bool.astype(np.uint8), k, iterations=1) > 0)
                            # numpy fallback: pixel survives only if all 8 neighbors (and self) are true
                            m = mask_bool.astype(np.uint8)
                            p = np.pad(m, ((1, 1), (1, 1)), mode="edge")
                            nb = (
                                p[0:-2, 0:-2] & p[0:-2, 1:-1] & p[0:-2, 2:] &
                                p[1:-1, 0:-2] & p[1:-1, 1:-1] & p[1:-1, 2:] &
                                p[2:  , 0:-2] & p[2:  , 1:-1] & p[2:  , 2:]
                            )
                            return (nb > 0)

                        # Top bar rim
                        if (hf_t > 0.0) and (top_h >= 3):
                            er_t = _erode1(top_mask)
                            rim_t = (top_mask & (~er_t))[:top_h, :]
                            if rim_t.any():
                                ys = np.arange(top_h, dtype=np.int32)[:, None]
                                xs = np.arange(w, dtype=np.int32)[None, :]
                                # sample 3 pixels inward (toward window = +y)
                                s1 = np.clip(ys + 2, 0, h - 1)
                                s2 = np.clip(ys + 3, 0, h - 1)
                                s3 = np.clip(ys + 4, 0, h - 1)
                                samp1 = warped[s1, xs]
                                samp2 = warped[s2, xs]
                                samp3 = warped[s3, xs]
                                med = np.median(np.stack([samp1, samp2, samp3], axis=0), axis=0).astype(np.float32)
                                region = warped[:top_h, :, :]
                                region = np.where(rim_t[..., None], (1.0 - hf_t) * region + hf_t * med, region)
                                warped[:top_h, :, :] = region

                        # Bottom bar rim
                        if (hf_b > 0.0) and (bot_h >= 3):
                            bot_mask_h = bot_mask[bar_h - bot_h:, :]
                            er_b = _erode1(bot_mask_h)
                            rim_b = bot_mask_h & (~er_b)
                            if rim_b.any():
                                ys0 = np.arange(h - bot_h, h, dtype=np.int32)[:, None]
                                xs = np.arange(w, dtype=np.int32)[None, :]
                                # sample inward (toward window = -y)
                                s1 = np.clip(ys0 - 2, 0, h - 1)
                                s2 = np.clip(ys0 - 3, 0, h - 1)
                                s3 = np.clip(ys0 - 4, 0, h - 1)
                                samp1 = warped[s1, xs]
                                samp2 = warped[s2, xs]
                                samp3 = warped[s3, xs]
                                med = np.median(np.stack([samp1, samp2, samp3], axis=0), axis=0).astype(np.float32)
                                region = warped[h - bot_h:, :, :]
                                region = np.where(rim_b[..., None], (1.0 - hf_b) * region + hf_b * med, region)
                                warped[h - bot_h:, :, :] = region

            out[i] = np.clip(warped, 0.0, 1.0)

        out_t = torch.from_numpy(out).to(image.device)
        return (out_t,)
