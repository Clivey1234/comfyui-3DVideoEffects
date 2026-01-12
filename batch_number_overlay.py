import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class ImageBatchNumberOverlay:
    """Overlay a frame/index number on each image in a batch.

    Input:
      images: IMAGE tensor [B, H, W, 3] in 0..1

    Output:
      Images: IMAGE tensor [B, H, W, 3] in 0..1
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "start_index": (
                    "INT",
                    {
                        "default": 1,
                        "min": 0,
                        "max": 999,
                        "step": 1,
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("Images",)
    FUNCTION = "apply_numbers"
    CATEGORY = "🟢 3DVidTools"

    def _get_font(self, height: int):
        # Font size scales with image height (fallback to default if TTF not available)
        try:
            size = max(174, height // 10)
            return ImageFont.truetype("arial.ttf", size=size)
        except Exception:
            return ImageFont.load_default()

    def _draw_index(self, pil_img: Image.Image, text: str):
        draw = ImageDraw.Draw(pil_img)
        font = self._get_font(pil_img.height)

        # Measure text size
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except Exception:
            # Fallback for older Pillow versions
            text_w, text_h = draw.textsize(text, font=font)

        # Bottom-left with padding
        x = 164
        y = pil_img.height - text_h - 164

        # Simple outline for readability
        outline_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in outline_offsets:
            draw.text((x + dx, y + dy), text, font=font, fill=(0, 0, 0))

        # Main text (white)
        draw.text((x, y), text, font=font, fill=(255, 255, 255))

    def _prepare_batch(self, tensor: torch.Tensor, name: str):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected '{name}' to be a torch.Tensor of type IMAGE")

        if tensor.ndim != 4 or tensor.shape[-1] != 3:
            raise ValueError(
                f"Expected IMAGE tensor '{name}' of shape [B, H, W, 3], got {tensor.shape}"
            )

        arr = tensor.detach().cpu().numpy()
        arr = np.clip(arr, 0.0, 1.0).astype(np.float32)
        return arr

    def apply_numbers(self, images, start_index):
        """Overlay the batch index number on each frame and return the updated batch."""
        device = images.device

        img_np = self._prepare_batch(images, "images")
        batch = img_np.shape[0]

        out_np = np.empty_like(img_np, dtype=np.float32)

        for i in range(batch):
            index_text = str(start_index + i)
            img_uint8 = np.clip(img_np[i] * 255.0, 0, 255).astype(np.uint8)
            pil_img = Image.fromarray(img_uint8)
            self._draw_index(pil_img, index_text)
            out_np[i] = np.asarray(pil_img, dtype=np.float32) / 255.0

        out_tensor = torch.from_numpy(out_np).to(device)
        return (out_tensor,)
