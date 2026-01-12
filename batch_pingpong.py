import torch


class BatchPingPong:
    """
    Classic ping-pong (reverse) playback for IMAGE batches.

    - No smoothing
    - No easing
    - No turnaround slowdown
    - Pure index reversal
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                "loop_count": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "🟢 3DVidTools"

    def _pingpong_once(self, N: int):
        if N <= 1:
            return list(range(N))
        return list(range(0, N)) + list(range(N - 2, 0, -1))

    def _pingpong_loopable(self, N: int):
        if N <= 1:
            return list(range(N))
        return list(range(0, N)) + list(range(N - 2, -1, -1))

    def process(self, images, loop_count=0):
        if not isinstance(images, torch.Tensor):
            raise TypeError("images must be a torch.Tensor")
        if images.dim() != 4:
            raise ValueError(f"Expected images [N,H,W,C], got {tuple(images.shape)}")

        N = int(images.shape[0])
        if N <= 1:
            return (images,)

        loops = max(0, int(loop_count))

        if loops == 0:
            idx = self._pingpong_once(N)
        else:
            unit = self._pingpong_loopable(N)
            idx = []
            for i in range(loops + 1):
                if i == 0:
                    idx.extend(unit)
                else:
                    idx.extend(unit[1:] if len(unit) > 1 else unit)

        idx_t = torch.tensor(idx, device=images.device, dtype=torch.long)
        out = images.index_select(0, idx_t)
        return (out,)

