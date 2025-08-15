# ComfyUI/custom_nodes/ComfyUI_minicpmv4/sanity_image.py
import numpy as np

class GenCheckerImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "size": ("INT", {"default": 128, "min": 16, "max": 1024, "step": 16}),
                "tiles": ("INT", {"default": 8, "min": 2, "max": 64}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "Utils/Test"

    def run(self, size=128, tiles=8):
        s = size
        t = tiles
        img = np.zeros((s, s, 3), dtype=np.float32)
        tile = s // t
        for y in range(t):
            for x in range(t):
                if (x + y) % 2 == 0:
                    img[y*tile:(y+1)*tile, x*tile:(x+1)*tile] = np.array([1.0, 1.0, 1.0])
                else:
                    img[y*tile:(y+1)*tile, x*tile:(x+1)*tile] = np.array([0.1, 0.1, 0.1])
        img = img[None, ...]  # batch dimension
        return (img,)

NODE_CLASS_MAPPINGS = {"GenCheckerImage": GenCheckerImage}
NODE_DISPLAY_NAME_MAPPINGS = {"GenCheckerImage": "Generate Checker Image"}
