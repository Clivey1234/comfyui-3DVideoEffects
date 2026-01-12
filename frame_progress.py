import time
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# Persistent state
_loop_count = 0
_last_loop_time = None
_avg_loop_time = None

# Fixed progress bar size
BAR_WIDTH = 128
BAR_HEIGHT = 128


class FrameProgress:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "SetFramePerBatch": ("INT", {"default": 12, "min": 1, "max": 10_000_000}),
                "TotalFrames": ("INT", {"default": 1, "min": 1, "max": 10_000_000}),
                "Reset": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("INT", "FLOAT", "STRING", "IMAGE")
    RETURN_NAMES = ("Framecount", "Percentage_Complete", "DebugText", "ProgressBarImage")
    FUNCTION = "run"
    CATEGORY = "🟢 3DVidTools"

    @staticmethod
    def run(SetFramePerBatch, TotalFrames, Reset):
        global _loop_count, _last_loop_time, _avg_loop_time

        now = time.time()

        # --- Reset handling ---
        if Reset:
            _loop_count = 0
            _last_loop_time = None
            _avg_loop_time = None
        else:
            # Each call = start of a new loop
            _loop_count += 1

            # Loop timing:
            if _loop_count == 1:
                _last_loop_time = now
            elif _loop_count > 1 and _last_loop_time is not None:
                loop_duration = max(now - _last_loop_time, 1e-6)
                _last_loop_time = now

                if _avg_loop_time is None:
                    _avg_loop_time = loop_duration
                else:
                    _avg_loop_time = (_avg_loop_time * 0.7) + (loop_duration * 0.3)

        # --- Frames done (completed loops only) ---
        frames_done = max((_loop_count - 1) * SetFramePerBatch, 0)
        if frames_done > TotalFrames:
            frames_done = TotalFrames

        # --- Percentage ---
        if TotalFrames <= 0:
            percentage = 0.0
        else:
            percentage = (frames_done / TotalFrames) * 100.0
        percentage = min(max(percentage, 0.0), 100.0)

        # --- ETA ---
        eta_text_short = "--:--"
        if (
            _avg_loop_time is not None
            and frames_done > 0
            and frames_done < TotalFrames
        ):
            remaining_frames = max(TotalFrames - frames_done, 0)
            loops_left = remaining_frames / float(SetFramePerBatch)

            eta_seconds = max(0.0, loops_left * _avg_loop_time * 1.1)
            mins = int(eta_seconds // 60)
            secs = int(eta_seconds % 60)
            eta_text_short = f"{mins:02d}:{secs:02d}"

        # Debug output
        if _avg_loop_time is not None:
            debug = (
                f"FramesDone={frames_done}, "
                f"TotalFrames={TotalFrames}, "
                f"Percentage={percentage:.2f}%, "
                f"ETA={eta_text_short}, "
                f"Loops={_loop_count}, "
                f"AvgLoop={_avg_loop_time:.3f}s"
            )
        else:
            debug = (
                f"FramesDone={frames_done}, "
                f"TotalFrames={TotalFrames}, "
                f"Percentage={percentage:.2f}%, "
                f"ETA={eta_text_short}, "
                f"Loops={_loop_count}"
            )

        # --- Create progress bar image ---
        img = Image.new("RGBA", (BAR_WIDTH, BAR_HEIGHT), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        bg_color = (235, 235, 235, 255)
        fill_color = (36, 116, 183, 255)

        draw.rectangle((0, 0, BAR_WIDTH - 1, BAR_HEIGHT - 1), fill=bg_color)

        fill_width = int((percentage / 100.0) * BAR_WIDTH)
        if fill_width > 0:
            draw.rectangle((0, 0, fill_width - 1, BAR_HEIGHT - 1), fill=fill_color)

        # --- BIGGER Fonts (triple sized) ---
        try:
            font_large = ImageFont.truetype("DejaVuSans-Bold.ttf", 60)   # 3× size
            font_small = ImageFont.truetype("DejaVuSans-Bold.ttf", 48)   # 3× size
        except Exception:
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()

        # Text values
        pct_text = f"{percentage:.0f}%"
        eta_text = f"ETA  {eta_text_short}"   # <-- TWO SPACES

        # Measure each line
        pct_bbox = draw.textbbox((0, 0), pct_text, font=font_large)
        pct_w = pct_bbox[2] - pct_bbox[0]
        pct_h = pct_bbox[3] - pct_bbox[1]

        eta_bbox = draw.textbbox((0, 0), eta_text, font=font_small)
        eta_w = eta_bbox[2] - eta_bbox[0]
        eta_h = eta_bbox[3] - eta_bbox[1]

        line_spacing = 10
        total_text_h = pct_h + line_spacing + eta_h

        start_y = (BAR_HEIGHT - total_text_h) // 2

        pct_x = (BAR_WIDTH - pct_w) // 2
        pct_y = start_y

        eta_x = (BAR_WIDTH - eta_w) // 2
        eta_y = pct_y + pct_h + line_spacing

        # Draw percentage
        draw.text(
            (pct_x, pct_y),
            pct_text,
            fill=(0, 0, 0, 255),
            font=font_large,
        )

        # Draw ETA
        draw.text(
            (eta_x, eta_y),
            eta_text,
            fill=(0, 0, 0, 255),
            font=font_small,
        )

        # Convert to tensor (B, H, W, C)
        rgb = img.convert("RGB")
        arr = np.array(rgb, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr)[None, ...]

        return (frames_done, percentage, debug, tensor)
