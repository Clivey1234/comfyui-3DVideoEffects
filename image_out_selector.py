import torch

class ImageOutSelector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ImageIn": ("IMAGE",),
                # Spinner that can also accept INT links
                "OutputSelect": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 2,
                        "step": 1,
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("ImageOut1", "ImageOut2")
    FUNCTION = "route_image"

    CATEGORY = "🟢 3DVidTools"

    def _make_blank(self, ImageIn):
        """
        Create a blank 256x256 image per batch item,
        matching dtype and device of the input.
        """
        batch = ImageIn.shape[0]
        blank = torch.zeros(
            (batch, 256, 256, 3),
            dtype=ImageIn.dtype,
            device=ImageIn.device,
        )
        return blank

    def route_image(self, ImageIn, OutputSelect):
        # ensure it's in the expected range in case upstream sends weird values
        if OutputSelect != 1:
            OutputSelect = 2

        blank = self._make_blank(ImageIn)

        if OutputSelect == 1:
            # Real image on Out1, blank on Out2
            return (ImageIn, blank)
        else:
            # Blank on Out1, real image on Out2
            return (blank, ImageIn)


NODE_CLASS_MAPPINGS = {
    "ImageOutSelector (SPB1234T)": ImageOutSelector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageOutSelector (SPB1234T)": "Image Out Selector (SPB1234T)",
}
