import re
import os

class VHSPathCleaner:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "VHSCombineFilename": ("VHS_FILENAMES",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("CleanedPath",)
    FUNCTION = "extract"

    CATEGORY = "🟢 3DVidTools"

    def extract(self, VHSCombineFilename):
        s = str(VHSCombineFilename)
        mp4_path = ""

        candidates = re.findall(
            r'([A-Za-z]:\\[^"\']*?\.mp4|/[^"\']*?\.mp4)',
            s,
            flags=re.IGNORECASE,
        )

        if candidates:
            candidate = candidates[-1]

            # Validate structure only (absolute path + .mp4)
            if (
                candidate.lower().endswith(".mp4")
                and (os.path.isabs(candidate) or re.match(r'^[A-Za-z]:\\', candidate))
            ):
                mp4_path = candidate
            else:
                mp4_path = ""

        return (mp4_path,)


NODE_CLASS_MAPPINGS = {
    "VHSPathCleaner (SPB1234T)": VHSPathCleaner
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VHSPathCleaner (SPB1234T)": "VHS Path Cleaner (SPB1234T)"
}
