import os
import re
import subprocess

def _extract_all_mp4_paths(vhs_filenames):
    s = str(vhs_filenames)

    candidates = re.findall(r'([A-Za-z]:[\\/][^"\'\s]+?\.mp4)', s, flags=re.IGNORECASE)
    if not candidates:
        candidates = re.findall(r'((?:/|\.{1,2}/)[^"\'\s]+?\.mp4)', s, flags=re.IGNORECASE)

    seen = set()
    out = []
    for p in candidates:
        key = p.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out

def _run(cmd):
    return subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

class YouTube3DMetadataSBSLR:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "VHSCombineFilename": ("VHS_FILENAMES",),
            },
            "optional": {
                "file_picker": ("STRING", {"default": ""}),
                "overwrite": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "Video/Export"

    def run(self, VHSCombineFilename, file_picker="", overwrite=False):
        paths = _extract_all_mp4_paths(VHSCombineFilename)

        if (not paths) and file_picker:
            paths = [file_picker]

        if not paths:
            return ()

        errors = []
        for src in paths:
            try:
                if not src.lower().endswith(".mp4"):
                    continue
                if not os.path.isfile(src):
                    raise FileNotFoundError(f"Input video not found: {src}")

                base = os.path.splitext(src)[0]
                out_path = base + "_YTReady.mkv"

                if os.path.exists(out_path) and (not overwrite):
                    continue

                # IMPORTANT: some FFmpeg builds don't have the -stereo_mode option.
                # For Matroska/WebM, FFmpeg reads/writes stereo mode via metadata tag "stereo_mode".
                cmd = [
                    "ffmpeg",
                    "-y" if overwrite else "-n",
                    "-i", src,
                    "-map", "0",
                    "-c", "copy",
                    "-metadata:s:v:0", "stereo_mode=left_right",
                    out_path
                ]
                _run(cmd)

            except subprocess.CalledProcessError as e:
                try:
                    err = e.stderr.decode("utf-8", errors="ignore")
                except Exception:
                    err = ""
                errors.append(f"FFmpeg failed for: {src}\n{err}")
            except Exception as e:
                errors.append(f"{src}: {e}")

        if errors:
            raise RuntimeError("YouTube 3D Metadata (SBS LR) encountered errors:\n\n" + "\n\n".join(errors))

        return ()
