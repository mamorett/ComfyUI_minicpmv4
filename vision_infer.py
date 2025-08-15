# ComfyUI/custom_nodes/ComfyUI_minicpmv4/vision_infer.py
import io
import base64
import numpy as np
from PIL import Image
from typing import List

from llama_cpp import Llama
from .loader import LlamaHandle

def _to_numpy(arr) -> np.ndarray:
    try:
        import torch
        if isinstance(arr, torch.Tensor):
            arr = arr.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(arr)

def _pil_from_comfy(image_batch) -> Image.Image:
    x = _to_numpy(image_batch)
    if x.ndim == 4:
        x = x[0]
    x = np.clip(x, 0.0, 1.0)
    x = (x * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(x)

def _pil_list_from_comfy(batch) -> List[Image.Image]:
    x = _to_numpy(batch)
    if x.ndim == 3:
        return [_pil_from_comfy(x)]
    elif x.ndim == 4:
        imgs = []
        for i in range(x.shape[0]):
            xi = np.clip(x[i], 0.0, 1.0)
            xi = (xi * 255.0 + 0.5).astype(np.uint8)
            imgs.append(Image.fromarray(xi))
        return imgs
    else:
        raise RuntimeError(f"[MiniCPM-V-4] Unsupported image tensor shape: {getattr(x, 'shape', type(x))}")

def _data_url_from_pil(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def _b64_from_pil(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

_GREET_SNIPPETS = (
    "how can i assist you today",
    "how can i help you today",
    "please upload an image",
    "provide an image",
    "i can help you analyze images",
)

class MiniCPMV4VisionInfer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llama": ("LLAMA",),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": "Describe the image in detail. List objects, colors, and any visible text. Answer in English."}),
            },
            "optional": {
                "max_tokens": ("INT", {"default": 320, "min": 1, "max": 4096}),
                "temperature": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 2.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 40, "min": 1, "max": 1000}),
                "repeat_penalty": ("FLOAT", {"default": 1.1, "min": 0.5, "max": 2.0, "step": 0.01}),
                "stop_1": ("STRING", {"default": ""}),
                "stop_2": ("STRING", {"default": ""}),
                "prepend_image_token": ("BOOLEAN", {"default": True, "tooltip": "Add <image> at the start of the user text"}),
                "stream_tokens": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "run"
    CATEGORY = "LLM/Multimodal"

    def _build_messages_text_first(self, pil_images: List[Image.Image], prompt: str, prepend_image_token: bool):
        user_text = prompt or ""
        if prepend_image_token and not user_text.lstrip().startswith("<image>"):
            user_text = "<image>\n" + user_text

        # text first
        parts = [{"type": "input_text", "text": user_text}]
        # include both representations for maximum compatibility in 0.2.89
        for im in pil_images:
            parts.append({"type": "image_url", "image_url": {"url": _data_url_from_pil(im)}})
            parts.append({"type": "input_image", "image": _b64_from_pil(im)})

        # assistant primer (empty) helps some chatml templates start generation properly
        messages = [
            {"role": "user", "content": parts},
            {"role": "assistant", "content": [{"type": "input_text", "text": ""}]},
        ]
        return messages

    def _build_messages_images_first(self, pil_images: List[Image.Image], prompt: str, prepend_image_token: bool):
        user_text = prompt or ""
        if prepend_image_token and not user_text.lstrip().startswith("<image>"):
            user_text = "<image>\n" + user_text

        # images first
        parts = []
        for im in pil_images:
            parts.append({"type": "image_url", "image_url": {"url": _data_url_from_pil(im)}})
            parts.append({"type": "input_image", "image": _b64_from_pil(im)})
        parts.append({"type": "input_text", "text": user_text})

        messages = [
            {"role": "user", "content": parts},
            {"role": "assistant", "content": [{"type": "input_text", "text": ""}]},
        ]
        return messages

    def _run_once(self, llm: Llama, messages, max_tokens, temperature, top_p, top_k, repeat_penalty, stops, stream_tokens):
        sampling = dict(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            stop=stops if stops else None,
        )
        text = ""
        if stream_tokens:
            for chunk in llm.create_chat_completion(messages=messages, stream=True, **sampling):
                piece = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                if piece:
                    text += piece
                    print(piece, end="", flush=True)
            print()
        else:
            out = llm.create_chat_completion(messages=messages, **sampling)
            text = out["choices"][0]["message"].get("content", "")
        return text.strip()

    def run(self, llama: LlamaHandle, image, prompt,
            max_tokens=320, temperature=0.6, top_p=0.95, top_k=40, repeat_penalty=1.1,
            stop_1="", stop_2="", prepend_image_token=True, stream_tokens=True):

        if llama is None or llama.llm is None or not llama.vision_ok:
            raise RuntimeError("[MiniCPM-V-4] Vision not available: ensure loader passed a vision-capable LlamaHandle.")

        pil_images = _pil_list_from_comfy(image)
        if len(pil_images) == 0:
            raise RuntimeError("[MiniCPM-V-4] No images provided to Vision Infer.")

        llm: Llama = llama.llm
        stops = [s.strip() for s in (stop_1, stop_2) if s and s.strip()]

        # Try text-first schema
        messages = self._build_messages_text_first(pil_images, prompt, prepend_image_token)
        out = self._run_once(llm, messages, max_tokens, temperature, top_p, top_k, repeat_penalty, stops, stream_tokens)
        low = out.lower()
        if not any(sn in low for sn in _GREET_SNIPPETS) and ("image" not in low or "upload" not in low):
            return (out,)

        # If it looked like a greeting / “upload image”, try images-first schema
        messages = self._build_messages_images_first(pil_images, prompt, prepend_image_token)
        out2 = self._run_once(llm, messages, max_tokens, temperature, top_p, top_k, repeat_penalty, stops, stream_tokens)
        return (out2,)
        

NODE_CLASS_MAPPINGS = {"MiniCPMV4VisionInfer": MiniCPMV4VisionInfer}
NODE_DISPLAY_NAME_MAPPINGS = {"MiniCPMV4VisionInfer": "MiniCPM-V-4 (GGUF) Vision Infer"}
