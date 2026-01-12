# multi_prompt_clip.py

import math
from nodes import CLIPTextEncode


class CLIPMultiPrompt_SelectByLoop_SBP1234T:
    """
    Select one of four positive text prompts based on a LoopCounter,
    then encode it to CONDITIONING using the standard CLIPTextEncode node.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "LoopCounter": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 4,
                        "step": 1,
                    },
                ),
                "Prompt_1": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "lines": 5,
                    },
                ),
                "Prompt_2": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "lines": 5,
                    },
                ),
                "Prompt_3": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "lines": 5,
                    },
                ),
                "Prompt_4": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "lines": 5,
                    },
                ),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "🟢 3DVidTools"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Force this node to be treated as changed on every run
        return math.nan

    def encode(
        self,
        clip,
        LoopCounter,
        Prompt_1,
        Prompt_2,
        Prompt_3,
        Prompt_4,
    ):
        # Pick the prompt based on LoopCounter
        if LoopCounter == 1:
            text = Prompt_1
        elif LoopCounter == 2:
            text = Prompt_2
        elif LoopCounter == 3:
            text = Prompt_3
        elif LoopCounter == 4:
            text = Prompt_4
        else:
            # Fallback: in case something weird happens, use Prompt_1
            text = Prompt_1

        # Use the built-in CLIPTextEncode node to get a proper CONDITIONING object
        encoder = CLIPTextEncode()
        # encoder.encode returns (conditioning,)
        return encoder.encode(clip, text)
