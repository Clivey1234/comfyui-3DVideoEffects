import torch

def _shift_horizontal(img: torch.Tensor, shift_px: int) -> torch.Tensor:
    """Shift an IMAGE tensor horizontally with black padding (no wrap).
    img: [B,H,W,C] float 0-1
    """
    if shift_px == 0:
        return img
    s = int(shift_px)
    out = torch.roll(img, shifts=s, dims=2)  # W dimension
    if s > 0:
        out[:, :, :s, :] = 0.0
    else:
        out[:, :, s:, :] = 0.0  # s negative: last -s cols
    return out

class Anaglyph3D:
    """Combine Left/Right IMAGE tensors into a single anaglyph image."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "left_image": ("IMAGE",),
                "right_image": ("IMAGE",),
                "anaglyph_type": (["Red/Cyan", "Red/Green", "Amber/Blue"], {"default": "Red/Cyan"}),
                "channel_balance": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.5, "step": 0.01}),
                "depth_strength": ("INT", {"default": 0, "min": -30, "max": 30, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "🟢 3DVidTools"

    def run(self, left_image, right_image, anaglyph_type="Red/Cyan", channel_balance=1.0, depth_strength=0):
        if not isinstance(left_image, torch.Tensor) or not isinstance(right_image, torch.Tensor):
            raise TypeError("left_image and right_image must be IMAGE tensors.")
        if left_image.ndim != 4 or right_image.ndim != 4:
            raise ValueError("Expected IMAGE tensors of shape [B,H,W,C].")
        if left_image.shape != right_image.shape:
            raise ValueError(f"Left/Right images must match shape. Got {tuple(left_image.shape)} vs {tuple(right_image.shape)}")

        s = int(depth_strength)
        if s != 0:
            left_s  = _shift_horizontal(left_image, -s)
            right_s = _shift_horizontal(right_image,  s)
        else:
            left_s, right_s = left_image, right_image

        bal = float(channel_balance)

        Lr, Lg = left_s[..., 0], left_s[..., 1]
        Rg, Rb = right_s[..., 1], right_s[..., 2]

        if anaglyph_type == "Red/Cyan":
            out_r = Lr
            out_g = Rg * bal
            out_b = Rb * bal
        elif anaglyph_type == "Red/Green":
            out_r = Lr
            out_g = Rg * bal
            out_b = torch.zeros_like(out_r)
        elif anaglyph_type == "Amber/Blue":
            out_r = left_s[..., 0]
            out_g = left_s[..., 1]
            out_b = right_s[..., 2] * bal
        else:
            raise ValueError(f"Unknown anaglyph_type: {anaglyph_type}")

        out = torch.stack([out_r, out_g, out_b], dim=-1)
        out = torch.clamp(out, 0.0, 1.0)
        return (out,)
