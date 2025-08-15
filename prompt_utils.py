# ComfyUI/custom_nodes/minicpmv4_llamacpp/prompt_utils.py
class VisionPromptBuilder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "user_prompt": ("STRING", {"multiline": True, "default": "Describe the image."}),
            },
            "optional": {
                "system_prompt": ("STRING", {"multiline": True, "default": "You are a helpful vision-language assistant."}),
                "prepend_image_token": ("BOOLEAN", {"default": True}),
                "chat_style": ("BOOLEAN", {"default": True, "tooltip": "Wrap with <|im_start|> chat tags"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "build"
    CATEGORY = "LLM/Multimodal"

    def build(self, user_prompt, system_prompt="", prepend_image_token=True, chat_style=True):
        up = (user_prompt or "").strip()
        sp = (system_prompt or "").strip()
        if prepend_image_token and "<image>" not in up:
            up = "<image>\n" + up
        if chat_style:
            prompt = ""
            if sp:
                prompt += f"<|im_start|>system\n{sp}<|im_end|>\n"
            prompt += f"<|im_start|>user\n{up}<|im_end|>\n<|im_start|>assistant\n"
            return (prompt,)
        return (up,)

NODE_CLASS_MAPPINGS = {"VisionPromptBuilder": VisionPromptBuilder}
NODE_DISPLAY_NAME_MAPPINGS = {"VisionPromptBuilder": "Vision Prompt Builder"}
