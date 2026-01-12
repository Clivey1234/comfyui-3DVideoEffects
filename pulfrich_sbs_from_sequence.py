import torch

class PulfrichSBSFromSequence:
    '''
    Creates a Pulfrich-effect stereo frame by time-shifting one eye and packing as SBS.

    Input:  IMAGE batch (B,H,W,C) float [0..1]
    Output: IMAGE batch (B,H,2*W,C) SBS
    '''

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "delay_frames": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 8,
                    "step": 1,
                    "tooltip": "Temporal offset in frames. The delayed eye uses an earlier frame (t - delay). 0 disables delay."
                }),
                "pad_mode": (["Hold First", "Loop", "Mirror"], {
                    "default": "Hold First",
                    "tooltip": "How to resolve indices before frame 0 when delay pushes t-delay negative."
                }),
                "fade_in_frames": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 16,
                    "step": 1,
                    "tooltip": "Ramps the effective delay from 0 to delay_frames over this many frames to avoid a hard start."
                }),
                "swap_eyes": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Swap the left/right halves (reverses perceived depth direction)."
                }),
                "dim_delayed_eye": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 0.7,
                    "step": 0.01,
                    "tooltip": "Darken the delayed eye (like a neutral-density filter). Can strengthen the Pulfrich effect for some clips."
                }),
                "delayed_eye_side": (["Right", "Left"], {
                    "default": "Right",
                    "tooltip": "Which SBS side gets delayed/dimmed."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images_sbs",)
    FUNCTION = "run"
    CATEGORY = "🟢 3DVidTools"

    @staticmethod
    def _mirror_index(idx: int, n: int) -> int:
        if n <= 1:
            return 0
        if idx >= 0:
            return min(idx, n - 1)
        k = -idx
        return min(k - 1, n - 1)

    @staticmethod
    def _resolve_index(idx: int, n: int, pad_mode: str) -> int:
        if n <= 1:
            return 0
        if idx >= 0:
            return min(idx, n - 1)

        if pad_mode == "Hold First":
            return 0
        if pad_mode == "Loop":
            return idx % n
        return PulfrichSBSFromSequence._mirror_index(idx, n)

    @staticmethod
    def _effective_delay(t: int, delay_frames: int, fade_in_frames: int) -> int:
        if delay_frames <= 0:
            return 0
        if fade_in_frames <= 0:
            return delay_frames
        frac = min(1.0, max(0.0, float(t) / float(fade_in_frames)))
        eff = int(round(delay_frames * frac))
        return max(0, min(delay_frames, eff))

    def run(
        self,
        images: torch.Tensor,
        delay_frames: int = 1,
        pad_mode: str = "Hold First",
        fade_in_frames: int = 0,
        swap_eyes: bool = False,
        dim_delayed_eye: float = 0.0,
        delayed_eye_side: str = "Right",
    ):
        if not isinstance(images, torch.Tensor):
            raise ValueError("Pulfrich SBS From Sequence: expected images as a torch.Tensor (ComfyUI IMAGE).")

        if images.ndim != 4:
            raise ValueError(f"Pulfrich SBS From Sequence: expected images shape (B,H,W,C); got {tuple(images.shape)}")

        B, H, W, C = images.shape
        if B <= 0:
            return (images,)

        delay_frames = int(delay_frames)
        fade_in_frames = int(fade_in_frames)
        dim_delayed_eye = max(0.0, min(0.7, float(dim_delayed_eye)))

        x = images
        if x.dtype not in (torch.float16, torch.float32, torch.float64):
            x = x.float()

        out = []
        for t in range(B):
            eff_delay = self._effective_delay(t, delay_frames, fade_in_frames)
            td = t - eff_delay
            td = self._resolve_index(td, B, pad_mode)

            current = x[t:t+1]
            delayed = x[td:td+1]

            if dim_delayed_eye > 0.0:
                delayed = delayed * (1.0 - dim_delayed_eye)

            if delayed_eye_side == "Right":
                left = current
                right = delayed
            else:
                left = delayed
                right = current

            if swap_eyes:
                left, right = right, left

            sbs = torch.cat([left, right], dim=2)
            out.append(sbs)

        images_sbs = torch.cat(out, dim=0)
        return (images_sbs,)


NODE_CLASS_MAPPINGS = {
    "PulfrichSBSFromSequence": PulfrichSBSFromSequence,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PulfrichSBSFromSequence": "Pulfrich SBS From Sequence",
}
