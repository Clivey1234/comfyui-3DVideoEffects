import torch


class ImageKeyFrames:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "First_Custom": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 4096,
                    },
                ),
                "Second_Custom": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 4096,
                    },
                ),
                "Batch_Start_Custom": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 4096,
                    },
                ),
                "Batch_Stop_Custom": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 4096,
                    },
                ),
            }
        }

    CATEGORY = "🟢 3DVidTools"

    # first, middle, last, first_custom, second_custom, batched_custom
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = (
        "first_frame",
        "middle_frame",
        "last_frame",
        "first_custom",
        "second_custom",
        "batched_custom",
    )
    FUNCTION = "get_keyframes"

    def _pick_frame(self, images: torch.Tensor, idx_1_based: int):
        """
        Helper to pick a single frame as a (1, H, W, C) batch.

        idx_1_based: 1 = first frame, 2 = second, etc.
        If idx_1_based > num_frames, we clamp to the last frame.
        """
        num_frames = images.shape[0]

        # Ensure index is at least 1
        if idx_1_based < 1:
            idx_1_based = 1

        # Convert to 0-based index and clamp to last frame
        idx_0 = idx_1_based - 1
        if idx_0 >= num_frames:
            idx_0 = num_frames - 1

        return images[idx_0:idx_0 + 1, ...].contiguous()

    def _pick_batch(self, images: torch.Tensor, start_1_based: int, stop_1_based: int):
        """
        Helper to pick a batch of frames as (N, H, W, C).

        start_1_based, stop_1_based: 1-based inclusive.
        Handles:
          - clamping to [1, num_frames]
          - swapping if start > stop
          - ensuring at least one frame is returned
        """
        num_frames = images.shape[0]

        if num_frames < 1:
            raise RuntimeError("ImageKeyFrames: IMAGE batch has zero frames")

        # Normalise to at least 1
        if start_1_based < 1:
            start_1_based = 1
        if stop_1_based < 1:
            stop_1_based = 1

        # Swap if reversed
        if start_1_based > stop_1_based:
            start_1_based, stop_1_based = stop_1_based, start_1_based

        # Convert to 0-based
        start_0 = start_1_based - 1
        stop_0 = stop_1_based - 1

        # Clamp into [0, num_frames - 1]
        if start_0 >= num_frames:
            start_0 = num_frames - 1
        if stop_0 >= num_frames:
            stop_0 = num_frames - 1

        # Ensure stop_0 >= start_0
        if stop_0 < start_0:
            stop_0 = start_0

        # Slice (inclusive stop → +1)
        return images[start_0:stop_0 + 1, ...].contiguous()

    def get_keyframes(
        self,
        images,
        First_Custom,
        Second_Custom,
        Batch_Start_Custom,
        Batch_Stop_Custom,
    ):
        # images should be (N, H, W, C)
        if not isinstance(images, torch.Tensor):
            raise RuntimeError("ImageKeyFrames: images must be a torch.Tensor")

        if images.dim() != 4:
            raise RuntimeError(
                f"ImageKeyFrames: expected 4D IMAGE tensor (N, H, W, C), "
                f"got {images.dim()} dims and shape {tuple(images.shape)}"
            )

        num_frames = images.shape[0]
        if num_frames < 1:
            raise RuntimeError("ImageKeyFrames: IMAGE batch has zero frames")

        # First frame: index 0
        first = images[0:1, ...].contiguous()

        # Last frame: index -1
        last = images[-1:, ...].contiguous()

        # Middle frame:
        # 1 frame  -> that frame
        # 2 frames -> second frame
        # 3+       -> floor(N / 2)
        if num_frames == 1:
            middle = first
        elif num_frames == 2:
            middle = last
        else:
            mid_idx = num_frames // 2
            middle = images[mid_idx:mid_idx + 1, ...].contiguous()

        # Custom single frames (1-based indices from the spinners)
        first_custom_img = self._pick_frame(images, int(First_Custom))
        second_custom_img = self._pick_frame(images, int(Second_Custom))

        # Custom batch of frames
        batched_custom_img = self._pick_batch(
            images,
            int(Batch_Start_Custom),
            int(Batch_Stop_Custom),
        )

        return (
            first,
            middle,
            last,
            first_custom_img,
            second_custom_img,
            batched_custom_img,
        )
