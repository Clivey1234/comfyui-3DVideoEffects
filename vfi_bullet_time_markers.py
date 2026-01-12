import math
import inspect
import torch


class _VFIHelperBase:
    def __init__(self):
        self._cached_gimmvfi_model = None

    def _get_nodes_module(self):
        import nodes
        return nodes

    def _get_node_mappings(self):
        nodes = self._get_nodes_module()
        m = getattr(nodes, "NODE_CLASS_MAPPINGS", None)
        if not isinstance(m, dict):
            raise RuntimeError("NODE_CLASS_MAPPINGS not available")
        return m

    def _resolve_override_key(self, override: str):
        if not override:
            return ""

        m = self._get_node_mappings()
        if override in m:
            return override

        nodes = self._get_nodes_module()
        disp = getattr(nodes, "NODE_DISPLAY_NAME_MAPPINGS", {})
        for k, v in disp.items():
            if v == override:
                return k
        return override

    def _extract_all_input_keys(self, node_cls):
        spec = node_cls.INPUT_TYPES()
        keys = set()
        for g in ("required", "optional"):
            keys |= set(spec.get(g, {}).keys())
        return keys

    def _get_default_inputs(self, node_cls):
        defaults = {}
        spec = node_cls.INPUT_TYPES()
        for g in ("required", "optional"):
            for k, v in spec.get(g, {}).items():
                if isinstance(v, (list, tuple)) and len(v) > 1 and isinstance(v[1], dict):
                    if "default" in v[1]:
                        defaults[k] = v[1]["default"]
        return defaults

    def _fill_required_dropdowns(self, node_cls, inputs):
        spec = node_cls.INPUT_TYPES()
        for k, v in spec.get("required", {}).items():
            if k in inputs:
                continue
            if isinstance(v, (list, tuple)) and isinstance(v[0], (list, tuple)):
                inputs[k] = v[0][0]
        return inputs

    def _ensure_gimmvfi_model(self, loader_key):
        if self._cached_gimmvfi_model is not None:
            return self._cached_gimmvfi_model

        nodes = self._get_nodes_module()
        key = self._resolve_override_key(loader_key)
        cls = nodes.NODE_CLASS_MAPPINGS[key]
        obj = cls()

        fn = getattr(obj, cls.FUNCTION)
        inputs = self._fill_required_dropdowns(cls, self._get_default_inputs(cls))
        model = fn(**inputs)[0]
        self._cached_gimmvfi_model = model
        return model

    def _call_vfi_x2(self, images, passes, vfi_key, loader_key):
        if passes <= 0:
            return images

        nodes = self._get_nodes_module()
        key = self._resolve_override_key(vfi_key)
        cls = nodes.NODE_CLASS_MAPPINGS[key]
        obj = cls()
        fn = getattr(obj, cls.FUNCTION)

        all_keys = self._extract_all_input_keys(cls)
        image_key = next(k for k in ("images", "image") if k in all_keys)

        model = self._ensure_gimmvfi_model(loader_key) if "gimmvfi_model" in all_keys else None

        for _ in range(passes):
            args = self._get_default_inputs(cls)
            args[image_key] = images
            if model is not None:
                args["gimmvfi_model"] = model
            if "interpolation_factor" in all_keys:
                args["interpolation_factor"] = 2
            if "ds_factor" in all_keys:
                args["ds_factor"] = 1.0
            images = fn(**args)[0]

        return images

    def _concat_no_dup(self, a, b):
        if a is None or a.numel() == 0:
            return b
        if b is None or b.numel() == 0:
            return a
        if torch.equal(a[-1], b[0]):
            return torch.cat([a, b[1:]], 0)
        return torch.cat([a, b], 0)


class VFIBulletTimeMarkers(_VFIHelperBase):

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "source_fps": ("INT", {"default": 16}),
                "target_fps": ("INT", {"default": 30}),
                "slow_start_frame": ("INT", {"default": 20}),
                "slow_end_frame": ("INT", {"default": 60}),
                "ramp_frames": ("INT", {"default": 5}),
                "ramp_vfi_passes_x2": ("INT", {"default": 1}),
                "plateau_vfi_passes_x2": ("INT", {"default": 1}),
                "output_mode": (["Keep slow (longer output)", "Retime to target fps"],),
                "final_hold_frames": ("INT", {"default": 0}),
            },
            "optional": {
                "vfi_node_key_override": ("STRING", {"default": "GIMM-VFI Interpolate"}),
                "vfi_model_loader_key_override": ("STRING", {"default": "(Down)Load GIMMVFI Model"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "🟢 3DVidTools"

    def process(
        self,
        images,
        source_fps,
        target_fps,
        slow_start_frame,
        slow_end_frame,
        ramp_frames,
        ramp_vfi_passes_x2,
        plateau_vfi_passes_x2,
        output_mode,
        final_hold_frames,
        vfi_node_key_override="",
        vfi_model_loader_key_override="",
    ):
        N = images.shape[0]
        A = max(0, min(int(slow_start_frame), N - 1))
        B = max(0, min(int(slow_end_frame), N - 1))
        if B < A:
            A, B = B, A

        R = max(0, int(ramp_frames))
        a0, a1 = max(0, A - R), min(N, A + R + 1)
        b0, b1 = max(0, B - R), min(N, B + R + 1)

        p0, p1, p2, p3, p4 = 0, a0, a1, b0, b1

        pre = images[p0:p1]
        ramp_in = images[p1:p2]
        plateau = images[p2:p3]
        ramp_out = images[p3:p4]
        post = images[p4:]

        if ramp_in.shape[0] > 1:
            ramp_in = self._call_vfi_x2(ramp_in, ramp_vfi_passes_x2, vfi_node_key_override, vfi_model_loader_key_override)
        if plateau.shape[0] > 1:
            plateau = self._call_vfi_x2(plateau, plateau_vfi_passes_x2, vfi_node_key_override, vfi_model_loader_key_override)
        if ramp_out.shape[0] > 1:
            ramp_out = self._call_vfi_x2(ramp_out, ramp_vfi_passes_x2, vfi_node_key_override, vfi_model_loader_key_override)

        out = None
        for part in (pre, ramp_in, plateau, ramp_out, post):
            out = self._concat_no_dup(out, part)

        if final_hold_frames > 0:
            out = torch.cat([out, out[-1:].repeat(final_hold_frames, 1, 1, 1)], 0)

        return (out,)

