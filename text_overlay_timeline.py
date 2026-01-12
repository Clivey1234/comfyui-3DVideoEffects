import os
import re
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


def _parse_color(s: str) -> Tuple[int, int, int]:
    """
    Accepts:
      - "#RRGGBB"
      - "R,G,B"
      - "R G B"
      - "R;G;B"
    """
    if s is None:
        return (255, 255, 255)
    s = str(s).strip()
    if not s:
        return (255, 255, 255)
    if s.startswith("#") and len(s) == 7:
        try:
            r = int(s[1:3], 16)
            g = int(s[3:5], 16)
            b = int(s[5:7], 16)
            return (r, g, b)
        except Exception:
            return (255, 255, 255)

    parts = re.split(r"[,\s;]+", s)
    parts = [p for p in parts if p != ""]
    if len(parts) >= 3:
        try:
            r, g, b = int(parts[0]), int(parts[1]), int(parts[2])
            r = max(0, min(255, r))
            g = max(0, min(255, g))
            b = max(0, min(255, b))
            return (r, g, b)
        except Exception:
            return (255, 255, 255)

    return (255, 255, 255)


def _resolve_font(font_name: str, font_size: int, font_path: str = "") -> ImageFont.ImageFont:
    """
    Best-effort font resolution:
      1) If font_path points to a file, use it.
      2) If font_name looks like a path and exists, use it.
      3) Try common system font locations for known names.
      4) Fallback to PIL default font.
    """
    font_size = int(font_size)
    font_size = max(1, font_size)

    # explicit path
    if font_path:
        fp = os.path.expandvars(os.path.expanduser(str(font_path)))
        if os.path.isfile(fp):
            try:
                return ImageFont.truetype(fp, font_size)
            except Exception:
                pass

    # name is a path
    if font_name:
        maybe_path = os.path.expandvars(os.path.expanduser(str(font_name)))
        if os.path.isfile(maybe_path):
            try:
                return ImageFont.truetype(maybe_path, font_size)
            except Exception:
                pass

    name = (font_name or "").strip().lower()
    candidates: List[str] = []

    # Windows common
    win_dir = os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts")
    if os.path.isdir(win_dir):
        mapping = {
            "arial": "arial.ttf",
            "arial bold": "arialbd.ttf",
            "times new roman": "times.ttf",
            "times new roman bold": "timesbd.ttf",
            "courier new": "cour.ttf",
            "courier new bold": "courbd.ttf",
            "verdana": "verdana.ttf",
            "verdana bold": "verdanab.ttf",
            "tahoma": "tahoma.ttf",
            "calibri": "calibri.ttf",
            "segoe ui": "segoeui.ttf",
            "segoe ui bold": "segoeuib.ttf",
            "consolas": "consola.ttf",
        }
        if name in mapping:
            candidates.append(os.path.join(win_dir, mapping[name]))
        if name.endswith((".ttf", ".otf", ".ttc")):
            candidates.append(os.path.join(win_dir, font_name))

    # Linux common
    linux_dirs = [
        "/usr/share/fonts",
        "/usr/local/share/fonts",
        os.path.expanduser("~/.fonts"),
        os.path.expanduser("~/.local/share/fonts"),
    ]
    linux_map = {
        "dejavu sans": ["DejaVuSans.ttf", "dejavu/DejaVuSans.ttf"],
        "dejavu sans bold": ["DejaVuSans-Bold.ttf", "dejavu/DejaVuSans-Bold.ttf"],
        "dejavu serif": ["DejaVuSerif.ttf", "dejavu/DejaVuSerif.ttf"],
        "dejavu mono": ["DejaVuSansMono.ttf", "dejavu/DejaVuSansMono.ttf"],
        "liberation sans": ["LiberationSans-Regular.ttf", "liberation/LiberationSans-Regular.ttf"],
        "liberation serif": ["LiberationSerif-Regular.ttf", "liberation/LiberationSerif-Regular.ttf"],
        "liberation mono": ["LiberationMono-Regular.ttf", "liberation/LiberationMono-Regular.ttf"],
    }
    if name in linux_map:
        for root in linux_dirs:
            for rel in linux_map[name]:
                candidates.append(os.path.join(root, "truetype", rel))
                candidates.append(os.path.join(root, rel))

    for c in candidates:
        if os.path.isfile(c):
            try:
                return ImageFont.truetype(c, font_size)
            except Exception:
                continue

    return ImageFont.load_default()


def _parse_timeline(multiline: str) -> List[Dict[str, Any]]:
    """
    Lines like:
      30-50 | top_center | Hello
      70-100 | x=120 y=800 | you

    Comments allowed with leading #, or blank lines ignored.
    """
    events: List[Dict[str, Any]] = []
    if multiline is None:
        return events

    for raw in str(multiline).splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 3:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue

        frame_part, pos_part = parts[0], parts[1]
        text_part = "|".join(parts[2:]).strip()

        m = re.match(r"^\s*(\d+)\s*-\s*(\d+)\s*$", frame_part)
        if not m:
            continue
        start_f = int(m.group(1))
        end_f = int(m.group(2))
        if end_f < start_f:
            start_f, end_f = end_f, start_f

        pos = pos_part.strip().lower()
        anchor = None
        x = None
        y = None

        anchor_set = {
            "top_left", "top_center", "top_right",
            "center_left", "center", "center_right",
            "bottom_left", "bottom_center", "bottom_right",
        }
        if pos in anchor_set:
            anchor = pos
        else:
            mx = re.search(r"x\s*=\s*(-?\d+)", pos_part, re.IGNORECASE)
            my = re.search(r"y\s*=\s*(-?\d+)", pos_part, re.IGNORECASE)
            if mx and my:
                x = int(mx.group(1))
                y = int(my.group(1))
            else:
                anchor = "top_left"

        events.append({"start": start_f, "end": end_f, "text": text_part, "anchor": anchor, "x": x, "y": y})

    return events


def _compute_xy(W: int, H: int, text_w: int, text_h: int, anchor: str, offset_x: int, offset_y: int):
    if anchor in ("top_left", "center_left", "bottom_left"):
        ax = 0
    elif anchor in ("top_center", "center", "bottom_center"):
        ax = (W - text_w) // 2
    elif anchor in ("top_right", "center_right", "bottom_right"):
        ax = W - text_w
    else:
        ax = 0

    if anchor in ("top_left", "top_center", "top_right"):
        ay = 0
    elif anchor in ("center_left", "center", "center_right"):
        ay = (H - text_h) // 2
    elif anchor in ("bottom_left", "bottom_center", "bottom_right"):
        ay = H - text_h
    else:
        ay = 0

    return int(ax + offset_x), int(ay + offset_y)


def _draw_text_with_outline(draw: ImageDraw.ImageDraw, x: int, y: int, text: str, font: ImageFont.ImageFont,
                           fill_rgba, outline_rgba, outline_thickness: int):
    ot = max(0, int(outline_thickness))
    if ot > 0:
        for dy in range(-ot, ot + 1):
            for dx in range(-ot, ot + 1):
                if dx == 0 and dy == 0:
                    continue
                draw.multiline_text((x + dx, y + dy), text, font=font, fill=outline_rgba, align="left")
    draw.multiline_text((x, y), text, font=font, fill=fill_rgba, align="left")


class TextOverlayTimeline:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "timeline_text": ("STRING", {
                    "multiline": True,
                    "default": "30-50 | top_center | Hello\n70-100 | x=120 y=800 | you"
                }),

                "font_name": ([
                    "default",
                    "Arial",
                    "Arial Bold",
                    "Calibri",
                    "Consolas",
                    "Courier New",
                    "Courier New Bold",
                    "Segoe UI",
                    "Segoe UI Bold",
                    "Tahoma",
                    "Times New Roman",
                    "Times New Roman Bold",
                    "Verdana",
                    "Verdana Bold",
                    "DejaVu Sans",
                    "DejaVu Sans Bold",
                    "DejaVu Serif",
                    "DejaVu Mono",
                    "Liberation Sans",
                    "Liberation Serif",
                    "Liberation Mono",
                ], {"default": "default"}),

                "font_path_optional": ("STRING", {"default": ""}),
                "font_size": ("INT", {"default": 48, "min": 6, "max": 512, "step": 1}),

                "fill_color": ("STRING", {"default": "255,255,255"}),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),

                "outline_thickness": ("INT", {"default": 3, "min": 0, "max": 64, "step": 1}),
                "outline_color": ("STRING", {"default": "0,0,0"}),

                "global_anchor": ([
                    "top_left", "top_center", "top_right",
                    "center_left", "center", "center_right",
                    "bottom_left", "bottom_center", "bottom_right",
                ], {"default": "top_center"}),

                "offset_x": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1}),
                "offset_y": ("INT", {"default": 40, "min": -4096, "max": 4096, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "VHS/text"

    def run(self, images, timeline_text, font_name, font_path_optional, font_size,
            fill_color, opacity, outline_thickness, outline_color,
            global_anchor, offset_x, offset_y):

        events = _parse_timeline(timeline_text)

        # map dropdown label to resolver key
        label_to_key = {
            "default": "default",
            "Arial": "arial",
            "Arial Bold": "arial bold",
            "Calibri": "calibri",
            "Consolas": "consolas",
            "Courier New": "courier new",
            "Courier New Bold": "courier new bold",
            "Segoe UI": "segoe ui",
            "Segoe UI Bold": "segoe ui bold",
            "Tahoma": "tahoma",
            "Times New Roman": "times new roman",
            "Times New Roman Bold": "times new roman bold",
            "Verdana": "verdana",
            "Verdana Bold": "verdana bold",
            "DejaVu Sans": "dejavu sans",
            "DejaVu Sans Bold": "dejavu sans bold",
            "DejaVu Serif": "dejavu serif",
            "DejaVu Mono": "dejavu mono",
            "Liberation Sans": "liberation sans",
            "Liberation Serif": "liberation serif",
            "Liberation Mono": "liberation mono",
        }
        key = label_to_key.get(str(font_name), str(font_name))
        font = _resolve_font(key, int(font_size), str(font_path_optional))

        fill_rgb = _parse_color(fill_color)
        outline_rgb = _parse_color(outline_color)
        a = int(max(0.0, min(1.0, float(opacity))) * 255.0)
        fill_rgba = (fill_rgb[0], fill_rgb[1], fill_rgb[2], a)
        outline_rgba = (outline_rgb[0], outline_rgb[1], outline_rgb[2], a)

        # ComfyUI IMAGE is typically a torch tensor [B,H,W,C] float32 in 0..1.
        # Convert to CPU numpy for PIL drawing, then return torch on the original device.
        if isinstance(images, torch.Tensor):
            device = images.device
            batch = images.detach().cpu().numpy()
        else:
            device = None
            batch = images if isinstance(images, np.ndarray) else np.asarray(images)

        B, H, W, C = batch.shape
        out = np.empty((B, H, W, C), dtype=np.float32)

        for i in range(B):
            frame = (np.clip(batch[i], 0.0, 1.0) * 255.0).astype(np.uint8)
            pil = Image.fromarray(frame, mode="RGB")
            draw = ImageDraw.Draw(pil, "RGBA")

            for ev in events:
                if i < ev["start"] or i > ev["end"]:
                    continue

                text = ev["text"]
                if not text or str(text).strip() == "":
                    continue

                bbox = draw.multiline_textbbox((0, 0), text, font=font, align="left")
                text_w = int(bbox[2] - bbox[0])
                text_h = int(bbox[3] - bbox[1])

                if ev["x"] is not None and ev["y"] is not None:
                    x = int(ev["x"])
                    y = int(ev["y"])
                else:
                    anchor = ev["anchor"] if ev["anchor"] is not None else str(global_anchor)
                    x, y = _compute_xy(W, H, text_w, text_h, anchor, int(offset_x), int(offset_y))

                _draw_text_with_outline(draw, x, y, text, font, fill_rgba, outline_rgba, int(outline_thickness))

            out[i] = (np.asarray(pil).astype(np.float32) / 255.0)

        if device is not None:
            out_t = torch.from_numpy(out).to(device=device)
            return (out_t,)
        return (out,)
