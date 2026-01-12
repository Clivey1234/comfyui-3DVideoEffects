# clip_text_encode_loop.py

class CLIPTextEncodeLoop:
    """
    A CLIP text encoder that selects one of four prompts based on an integer LoopCounter.

    Inputs:
      - clip: CLIP model
      - LoopCounter: INT (1–4). Selects which prompt to use.
      - Prompt_1..Prompt_4: multiline strings

    Output:
      - CONDITIONING: standard ComfyUI conditioning:
          [[cond_tensor, {"pooled_output": pooled_tensor}]]
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "LoopCounter": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 4,
                    "step": 1,
                }),
                "Prompt_1": ("STRING", {
                    "multiline": True,
                    "default": "",
                    # some UIs ignore this, but it's safe to include
                    "lines": 5,
                }),
                "Prompt_2": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "lines": 5,
                }),
                "Prompt_3": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "lines": 5,
                }),
                "Prompt_4": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "lines": 5,
                }),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "🟢 3DVidTools"

    def encode(self, clip, LoopCounter, Prompt_1, Prompt_2, Prompt_3, Prompt_4):
        # Clamp LoopCounter to [1, 4] to avoid errors
        if LoopCounter < 1:
            idx = 1
        elif LoopCounter > 4:
            idx = 4
        else:
            idx = LoopCounter

        prompts = [Prompt_1, Prompt_2, Prompt_3, Prompt_4]
        selected_prompt = prompts[idx - 1] if prompts[idx - 1] is not None else ""

        # Match stock CLIPTextEncode behavior:
        # tokenize -> encode_from_tokens(return_pooled=True)
        tokens = clip.tokenize(selected_prompt)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)

        conditioning = [[cond, {"pooled_output": pooled}]]
        return (conditioning,)
