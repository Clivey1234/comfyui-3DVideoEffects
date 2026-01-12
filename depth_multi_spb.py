import torch
import numpy as np
import comfy
import comfy.model_management as model_management


class DepthMultiBackend2Image:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Select backend
                "backend": (
                    ["DepthAnything", "LeReS"],
                    {"default": "DepthAnything"},
                ),

                # DepthAnything-only (ignored for LeReS)
                "ckpt_name": (
                    [
                        "depth_anything_vitl14.pth",
                        "depth_anything_vitb14.pth",
                        "depth_anything_vits14.pth",
                    ],
                    {"default": "depth_anything_vitl14.pth"},
                ),

                # LeReS-only (ignored for DepthAnything)
                "rm_nearest": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.01},
                ),
                "rm_background": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.01},
                ),
                "boost": (
                    ["disable", "enable"],
                    {"default": "disable"},
                ),

                # Shared
                "resolution": (
                    "INT",
                    {"default": 512, "min": 64, "max": 16384, "step": 64},
                ),

                # Single image in
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("depth", "debug")
    FUNCTION = "execute"

    CATEGORY = "🟢 3DVidTools"

    def execute(
        self,
        backend,
        ckpt_name,
        rm_nearest,
        rm_background,
        boost,
        resolution,
        image,
    ):
        """
        One node containing both DepthAnything + LeReS.
        debug text output reports which one was used.
        """

        device = model_management.get_torch_device()

        # Load correct model
        if backend == "DepthAnything":
            from custom_controlnet_aux.depth_anything import DepthAnythingDetector

            model = DepthAnythingDetector.from_pretrained(
                filename=ckpt_name
            ).to(device)

            def run_model(np_img):
                return model(
                    np_img,
                    output_type="np",
                    detect_resolution=resolution,
                )

        else:  # LeReS
            from custom_controlnet_aux.leres import LeresDetector

            model = LeresDetector.from_pretrained().to(device)

            def run_model(np_img):
                return model(
                    np_img,
                    output_type="np",
                    detect_resolution=resolution,
                    thr_a=rm_nearest,
                    thr_b=rm_background,
                    boost=(boost == "enable"),
                )

        batch_size = image.shape[0]
        pbar = comfy.utils.ProgressBar(batch_size)

        outputs = []
        for i in range(batch_size):
            np_image = (image[i].cpu().numpy() * 255.0).astype(np.uint8)

            np_result = run_model(np_image)
            out = torch.from_numpy(np_result.astype(np.float32) / 255.0)
            outputs.append(out)

            pbar.update(1)

        depth_out = torch.stack(outputs, dim=0)

        # Debug text output
        debug_text = backend

        del model
        return (depth_out, debug_text)
