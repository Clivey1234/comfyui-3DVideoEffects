class ImageInSelector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ImageIn1": ("IMAGE",),
                "ImageIn2": ("IMAGE",),
                # Spinner that can also accept INT links
                "InputSelect": (
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

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("ImageOut",)
    FUNCTION = "route"

    CATEGORY = "🟢 3DVidTools"

    def route(self, ImageIn1, ImageIn2, InputSelect):
        # Normalise to 1 or 2
        if InputSelect != 1:
            InputSelect = 2

        if InputSelect == 1:
            return (ImageIn1,)
        else:
            return (ImageIn2,)


NODE_CLASS_MAPPINGS = {
    "ImageInSelector (SPB1234T)": ImageInSelector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageInSelector (SPB1234T)": "Image In Selector (SPB1234T)",
}
