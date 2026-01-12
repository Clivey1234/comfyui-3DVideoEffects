# loop_n_helper.py

class LoopNHelper:
    """
    Loop N Times helper node.

    Behaviour:
      - Maintains an internal loop counter across executions.
      - On each execution:
          * If reset is True -> counter is cleared to 0 first.
          * Counter is incremented by 1.
          * If counter > max_loops -> raises an Exception to stop the workflow.
          * Otherwise -> outputs the current counter value as LoopCounter (INT).

    Typical usage:
      - Set max_loops = 4.
      - Connect LoopCounter output to your CLIPTextEncodeLoop node's LoopCounter input.
      - Queue the same prompt multiple times (e.g. queue size 10).
      - Runs 1..4 will output 1,2,3,4 and use Prompt_1..Prompt_4.
      - On run 5, it throws and halts further runs.
    """

    # Class-level state shared across runs while ComfyUI is running
    current_loop = 0

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "max_loops": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 9999,
                    "step": 1,
                }),
                "reset": ("BOOL", {
                    "default": False,
                }),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("LoopCounter",)
    FUNCTION = "compute"
    CATEGORY = "🟢 3DVidTools"

    def compute(self, max_loops, reset):
        # Reset state if requested
        if reset:
            type(self).current_loop = 0

        # Increment loop count
        type(self).current_loop += 1
        current = type(self).current_loop

        # If we've exceeded the allowed number of loops, stop the workflow
        if current > max_loops:
            # Optional: reset back to 0 so the next run starts fresh
            # type(self).current_loop = 0
            raise Exception(
                f"LoopNHelper: maximum loops ({max_loops}) reached. "
                "Stopping workflow."
            )

        # Normal case: return the current loop index
        return (current,)
