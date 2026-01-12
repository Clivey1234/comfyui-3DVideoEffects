import math
import inspect
import torch


class VFIEaseEnds:
    """
    Split → VFI (ends) → Recombine → VFI (global) → FPS retime.

    Works with ComfyUI-GIMM-VFI by auto-detecting:
      - the interpolate node (e.g. GIMMVFI_interpolate)
      - the model loader node (e.g. DownloadAndLoadGIMMVFIModel) that outputs gimmvfi_model

    Caches gimmvfi_model so it's loaded once per session.

    One input connector: images
    One output: IMAGE
    """

    def __init__(self):
        self._cached_gimmvfi_model = None
        self._cached_gimmvfi_model_key = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),

                "ease_frames": ("INT", {"default": 6, "min": 0, "max": 1000, "step": 1}),
                "ease_mode": (["Both ends", "Start only", "End only"],),

                "source_fps": ("INT", {"default": 16, "min": 1, "max": 240, "step": 1}),
                "target_fps": ("INT", {"default": 30, "min": 1, "max": 240, "step": 1}),

                "Start_End_vfi_passes_x2": ("INT", {"default": 1, "min": 0, "max": 4, "step": 1}),
                "global_vfi_passes_x2": ("INT", {"default": 1, "min": 0, "max": 4, "step": 1}),

                "fps_resample": (["Even Drop", "Nearest"],),

                "vfi_node_select": (["Auto"],),
            },
            "optional": {
                # Can be internal key OR UI display name
                "vfi_node_key_override": ("STRING", {"default": "GIMMVFI_interpolate"}),
                "vfi_model_loader_key_override": ("STRING", {"default": "DownloadAndLoadGIMMVFIModel"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "🟢 3DVidTools"

    # ------------------------
    # Node registry helpers
    # ------------------------

    def _get_nodes_module(self):
        import nodes
        return nodes

    def _get_node_mappings(self):
        nodes = self._get_nodes_module()
        m = getattr(nodes, "NODE_CLASS_MAPPINGS", None)
        if not isinstance(m, dict) or not m:
            raise RuntimeError("Could not access nodes.NODE_CLASS_MAPPINGS.")
        return m

    def _resolve_override_key(self, override: str):
        """
        Accept either:
          - internal key in NODE_CLASS_MAPPINGS
          - UI display name in NODE_DISPLAY_NAME_MAPPINGS

        Returns resolved internal key (or "" if no override).
        """
        if not override or not override.strip():
            return ""

        override = override.strip()
        m = self._get_node_mappings()
        if override in m:
            return override

        nodes = self._get_nodes_module()
        disp = getattr(nodes, "NODE_DISPLAY_NAME_MAPPINGS", None)
        if isinstance(disp, dict) and disp:
            # exact
            for k, v in disp.items():
                if v == override:
                    return k
            # case-insensitive
            ov_l = override.lower()
            for k, v in disp.items():
                try:
                    if str(v).lower() == ov_l:
                        return k
                except Exception:
                    pass

        # unresolved; return original so we can error clearly
        return override

    def _extract_all_input_keys(self, node_cls):
        spec = node_cls.INPUT_TYPES()
        keys = set()
        for group in ("required", "optional"):
            g = spec.get(group, {})
            if isinstance(g, dict):
                keys |= set(g.keys())
        return keys

    def _get_default_inputs(self, node_cls):
        """
        Collect meta defaults (where present). Does NOT automatically fill required dropdowns.
        """
        defaults = {}
        if not hasattr(node_cls, "INPUT_TYPES"):
            return defaults
        try:
            spec = node_cls.INPUT_TYPES()
        except Exception:
            return defaults

        for group in ("required", "optional"):
            g = spec.get(group, {})
            if not isinstance(g, dict):
                continue
            for k, v in g.items():
                # v = ("TYPE", {"default":...}) or (["a","b"],) etc.
                if isinstance(v, (list, tuple)) and len(v) >= 2 and isinstance(v[1], dict):
                    meta = v[1]
                    if "default" in meta:
                        defaults[k] = meta["default"]
        return defaults

    def _fill_required_dropdowns(self, node_cls, inputs: dict):
        """
        For required inputs with no explicit default, pick the first dropdown option.
        This is necessary for GIMM loader required 'model', etc.
        """
        try:
            spec = node_cls.INPUT_TYPES()
        except Exception:
            return inputs

        req = spec.get("required", {})
        if not isinstance(req, dict):
            return inputs

        for k, v in req.items():
            if k in inputs:
                continue

            # Most common: (["opt1","opt2"],)  -> choose "opt1"
            if isinstance(v, (list, tuple)) and len(v) >= 1:
                first = v[0]
                if isinstance(first, (list, tuple)) and len(first) > 0:
                    inputs[k] = first[0]
                    continue

                # Sometimes required uses ("TYPE",) — can't infer safely; leave unset
        return inputs

    # ------------------------
    # Find nodes
    # ------------------------

    def _find_vfi_node_key(self, override_key: str = ""):
        m = self._get_node_mappings()

        resolved = self._resolve_override_key(override_key)
        if resolved:
            if resolved in m:
                return resolved, m[resolved]
            raise RuntimeError(
                f"vfi_node_key_override='{override_key}' not found in NODE_CLASS_MAPPINGS (resolved='{resolved}')."
            )

        # Auto detect interpolate
        candidates = []
        for key, cls_ in m.items():
            name = f"{key} {getattr(cls_, '__name__', '')}".lower()
            if "interpol" in name and ("gimm" in name or "vfi" in name):
                score = 0
                if "gimmvfi_interpolate" in name or "gimmvfi" in name:
                    score += 20
                if "gimm" in name:
                    score += 10
                if "interpol" in name:
                    score += 8
                if "vfi" in name:
                    score += 5
                if "load" in name or "download" in name:
                    score -= 6
                candidates.append((score, key, cls_))

        candidates.sort(reverse=True, key=lambda x: x[0])
        if not candidates:
            raise RuntimeError(
                "Could not auto-detect a GIMM-VFI interpolate node. "
                "Set vfi_node_key_override to the exact node key (or its UI display name)."
            )
        return candidates[0][1], candidates[0][2]

    def _find_gimmvfi_model_loader(self, override_key: str = ""):
        m = self._get_node_mappings()

        resolved = self._resolve_override_key(override_key)
        if resolved:
            if resolved in m:
                return resolved, m[resolved]
            raise RuntimeError(
                f"vfi_model_loader_key_override='{override_key}' not found in NODE_CLASS_MAPPINGS (resolved='{resolved}')."
            )

        candidates = []
        for key, cls_ in m.items():
            name = f"{key} {getattr(cls_, '__name__', '')}".lower()
            if "gimm" in name and ("load" in name or "download" in name or "model" in name):
                # prefer the actual DownloadAndLoad class
                score = 0
                if "downloadandload" in name:
                    score += 25
                if "download" in name:
                    score += 12
                if "load" in name:
                    score += 10
                if "model" in name:
                    score += 6
                candidates.append((score, key, cls_))

        candidates.sort(reverse=True, key=lambda x: x[0])
        if not candidates:
            raise RuntimeError(
                "Could not auto-detect a GIMM-VFI model loader node. "
                "Set vfi_model_loader_key_override to the exact node key (or its UI display name)."
            )
        return candidates[0][1], candidates[0][2]

    # ------------------------
    # Model load
    # ------------------------

    def _ensure_gimmvfi_model(self, loader_override_key: str = ""):
        if self._cached_gimmvfi_model is not None:
            return self._cached_gimmvfi_model

        key, cls_ = self._find_gimmvfi_model_loader(loader_override_key)
        obj = cls_()

        fn_name = getattr(cls_, "FUNCTION", None)
        if not fn_name or not hasattr(obj, fn_name):
            raise RuntimeError(f"Detected GIMM-VFI loader node '{key}' has no FUNCTION callable.")

        fn = getattr(obj, fn_name)

        # Start from meta defaults, then fill required dropdowns like 'model'
        call_inputs = self._get_default_inputs(cls_)
        call_inputs = self._fill_required_dropdowns(cls_, call_inputs)

        # Filter kwargs to the function signature
        try:
            sig = inspect.signature(fn)
            allowed = set(sig.parameters.keys())
            filtered = {k: v for k, v in call_inputs.items() if k in allowed}
        except Exception:
            filtered = call_inputs

        out = fn(**filtered)
        if not isinstance(out, (tuple, list)) or len(out) < 1:
            raise RuntimeError(f"GIMM-VFI loader '{key}' returned unexpected output type: {type(out)}")

        model_obj = out[0]
        self._cached_gimmvfi_model = model_obj
        self._cached_gimmvfi_model_key = key
        return model_obj

    # ------------------------
    # VFI call
    # ------------------------

    def _call_vfi_x2(self, images: torch.Tensor, passes_x2: int, vfi_override_key: str = "", loader_override_key: str = ""):
        if passes_x2 <= 0:
            return images

        key, cls_ = self._find_vfi_node_key(vfi_override_key)
        obj = cls_()

        fn_name = getattr(cls_, "FUNCTION", None)
        if not fn_name or not hasattr(obj, fn_name):
            raise RuntimeError(f"Detected VFI node '{key}' has no FUNCTION callable.")

        fn = getattr(obj, fn_name)

        base_inputs = self._get_default_inputs(cls_)
        all_keys = self._extract_all_input_keys(cls_)

        # Find image input
        image_key = None
        for cand in ("images", "image", "frames", "input_images", "imgs"):
            if cand in all_keys:
                image_key = cand
                break
        if image_key is None:
            raise RuntimeError(f"Detected VFI node '{key}' has no recognizable images input.")

        # Load model if required
        needs_model = "gimmvfi_model" in all_keys
        gimm_model = self._ensure_gimmvfi_model(loader_override_key) if needs_model else None

        for _ in range(passes_x2):
            call_inputs = dict(base_inputs)
            call_inputs[image_key] = images

            if needs_model:
                call_inputs["gimmvfi_model"] = gimm_model

            # Force "true x2" for this wrapper
            if "interpolation_factor" in all_keys:
                call_inputs["interpolation_factor"] = 2
            if "ds_factor" in all_keys:
                call_inputs["ds_factor"] = 1.0
            if "control_after_generate" in all_keys:
                call_inputs["control_after_generate"] = "fixed"

            # Filter to signature
            try:
                sig = inspect.signature(fn)
                allowed = set(sig.parameters.keys())
                filtered = {k: v for k, v in call_inputs.items() if k in allowed}
            except Exception:
                filtered = call_inputs

            out = fn(**filtered)
            if not isinstance(out, (tuple, list)) or len(out) < 1:
                raise RuntimeError(f"VFI node '{key}' returned unexpected output type: {type(out)}")

            images = out[0]
            if not isinstance(images, torch.Tensor):
                raise RuntimeError(f"VFI node '{key}' output[0] is not a torch.Tensor")

        return images

    # ------------------------
    # Combine + retime
    # ------------------------

    def _concat_no_dup(self, a: torch.Tensor, b: torch.Tensor):
        if a is None:
            return b
        if b is None:
            return a
        if a.numel() == 0:
            return b
        if b.numel() == 0:
            return a
        try:
            if torch.equal(a[-1], b[0]):
                return torch.cat([a, b[1:]], dim=0)
        except Exception:
            pass
        return torch.cat([a, b], dim=0)

    def _retime_even_drop(self, images: torch.Tensor, fps_in: int, fps_out: int):
        N = int(images.shape[0])
        if N <= 1 or fps_in == fps_out:
            return images

        duration = (N - 1) / float(fps_in)
        M = int(round(duration * fps_out)) + 1
        M = max(2, M)

        idx = []
        for k in range(M):
            src = int(math.floor((k * fps_in) / float(fps_out) + 0.5))
            src = max(0, min(src, N - 1))
            idx.append(src)

        # reduce consecutive duplicates from rounding
        for k in range(1, len(idx)):
            if idx[k] == idx[k - 1] and idx[k] < N - 1:
                idx[k] += 1

        idx_t = torch.tensor(idx, device=images.device, dtype=torch.long)
        return images.index_select(0, idx_t)

    def _retime_nearest(self, images: torch.Tensor, fps_in: int, fps_out: int):
        N = int(images.shape[0])
        if N <= 1 or fps_in == fps_out:
            return images

        duration = (N - 1) / float(fps_in)
        M = int(round(duration * fps_out)) + 1
        M = max(2, M)

        idx = []
        for k in range(M):
            t = k / float(fps_out)
            src = int(round(t * fps_in))
            src = max(0, min(src, N - 1))
            idx.append(src)

        idx_t = torch.tensor(idx, device=images.device, dtype=torch.long)
        return images.index_select(0, idx_t)

    # ------------------------
    # Main
    # ------------------------

    def process(
        self,
        images,
        ease_frames,
        ease_mode,
        source_fps,
        target_fps,
        Start_End_vfi_passes_x2,
        global_vfi_passes_x2,
        fps_resample,
        vfi_node_select,
        vfi_node_key_override="",
        vfi_model_loader_key_override="",
    ):
        if not isinstance(images, torch.Tensor):
            raise TypeError("images must be a torch.Tensor")
        if images.dim() != 4:
            raise ValueError(f"Expected images [N,H,W,C], got {tuple(images.shape)}")

        N = int(images.shape[0])
        if N <= 1:
            return (images,)

        X = int(ease_frames)
        X = max(0, min(X, (N - 1) // 2))

        src_fps = int(source_fps)
        dst_fps = int(target_fps)
        if src_fps <= 0 or dst_fps <= 0:
            raise ValueError("source_fps and target_fps must be > 0")

        end_passes = max(0, int(Start_End_vfi_passes_x2))
        glob_passes = max(0, int(global_vfi_passes_x2))

        # Split
        if X == 0:
            start = images[:0]
            mid = images
            end = images[:0]
        else:
            start = images[:X]
            mid = images[X:N - X]
            end = images[N - X:]

        # Densify ends
        start_d = start
        end_d = end

        if ease_mode in ("Both ends", "Start only") and X > 0 and end_passes > 0:
            start_d = self._call_vfi_x2(start_d, end_passes, vfi_node_key_override, vfi_model_loader_key_override)

        if ease_mode in ("Both ends", "End only") and X > 0 and end_passes > 0:
            # Improve symmetry vs. the start: reverse the tail slice so the VFI model gets "future" context,
            # then reverse back. This often makes ease-out feel as smooth as ease-in.
            end_rev = torch.flip(end_d, dims=[0])
            end_rev = self._call_vfi_x2(end_rev, end_passes, vfi_node_key_override, vfi_model_loader_key_override)
            end_d = torch.flip(end_rev, dims=[0])

        # Recombine
        tmp = None
        tmp = self._concat_no_dup(tmp, start_d)
        tmp = self._concat_no_dup(tmp, mid)
        tmp = self._concat_no_dup(tmp, end_d)

        # Global VFI
        full = self._call_vfi_x2(tmp, glob_passes, vfi_node_key_override, vfi_model_loader_key_override)

        fps_full = src_fps * (2 ** glob_passes)

        # Retime to target fps
        if fps_resample == "Even Drop":
            out = self._retime_even_drop(full, fps_full, dst_fps)
        else:
            out = self._retime_nearest(full, fps_full, dst_fps)

        return (out,)

