import torch
import numpy as np
import comfy
import comfy.model_management as model_management


class DepthAnything2Image:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (
                    [
                        "depth_anything_vitl14.pth",
                        "depth_anything_vitb14.pth",
                        "depth_anything_vits14.pth",
                    ],
                    {"default": "depth_anything_vitl14.pth"},
                ),
                "resolution": (
                    "INT",
                    {
                        "default": 512,
                        "min": 64,
                        "max": 16384,
                        "step": 64,
                    },
                ),
                # Spinner 1–8, can also take INT input from another node
                "which_image": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 8,
                        "step": 1,
                    },
                ),
                # image1 is mandatory
                "image1": ("IMAGE",),
            },
            # image2–image8 are optional
            "optional": {
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "image6": ("IMAGE",),
                "image7": ("IMAGE",),
                "image8": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    CATEGORY = "🟢 3DVidTools"

    def execute(
        self,
        ckpt_name,
        resolution,
        which_image,
        image1,
        image2=None,
        image3=None,
        image4=None,
        image5=None,
        image6=None,
        image7=None,
        image8=None,
    ):
        from custom_controlnet_aux.depth_anything import DepthAnythingDetector

        # Put all images into a list for index-based access
        images = [image1, image2, image3, image4, image5, image6, image7, image8]

        # Clamp which_image to [1, 8] and convert to zero-based index
        idx = int(which_image)
        if idx < 1:
            idx = 1
        if idx > 8:
            idx = 8
        selected = images[idx - 1]

        # Fallback: if selected image is None (unconnected), use image1
        if selected is None:
            selected = image1

        device = model_management.get_torch_device()
        model = DepthAnythingDetector.from_pretrained(filename=ckpt_name).to(device)

        batch_size = selected.shape[0]
        pbar = comfy.utils.ProgressBar(batch_size)

        outputs = []
        for i in range(batch_size):
            # selected[i]: [H, W, C] in 0–1 float
            np_image = (selected[i].cpu().numpy() * 255.0).astype(np.uint8)

            np_result = model(
                np_image,
                output_type="np",
                detect_resolution=resolution,
            )

            out = torch.from_numpy(np_result.astype(np.float32) / 255.0)
            outputs.append(out)
            pbar.update(1)

        out_tensor = torch.stack(outputs, dim=0)
        del model
        return (out_tensor,)
