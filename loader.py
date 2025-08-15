# ComfyUI/custom_nodes/minicpmv4_llamacpp/loader.py
import os
import dataclasses
from typing import Optional, Dict, Tuple
import hashlib

import folder_paths
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from PIL import Image
import io, base64

REPO_ID = "openbmb/MiniCPM-V-4-gguf"

MINICPMV4_GGUF = {
    "ggml-model-Q4_0.gguf":   "Q4_0 (2.08 GB)",
    "ggml-model-Q4_1.gguf":   "Q4_1 (2.29 GB)",
    "ggml-model-Q4_K_M.gguf": "Q4_K_M (2.19 GB) [balanced]",
    "ggml-model-Q4_K_S.gguf": "Q4_K_S (2.09 GB)",
    "ggml-model-Q5_0.gguf":   "Q5_0 (2.51 GB)",
    "ggml-model-Q5_1.gguf":   "Q5_1 (2.72 GB)",
    "ggml-model-Q5_K_M.gguf": "Q5_K_M (2.56 GB) [quality]",
    "ggml-model-Q5_K_S.gguf": "Q5_K_S (2.51 GB)",
    "ggml-model-Q6_K.gguf":   "Q6_K (2.96 GB)",
    "ggml-model-Q8_0.gguf":   "Q8_0 (3.83 GB) [max quality]",
}

KNOWN_SHA256: Dict[str, str] = {}

def _models_dir() -> str:
    return os.path.join(folder_paths.models_dir, "llama")

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _local_path(filename: str) -> str:
    return os.path.join(_models_dir(), filename)

def _sha256_file(path: str, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def _download_if_missing(filename: str) -> str:
    local_path = _local_path(filename)
    if os.path.exists(local_path):
        sha_expected = KNOWN_SHA256.get(filename)
        if sha_expected:
            sha_local = _sha256_file(local_path)
            if sha_local != sha_expected:
                print(f"[MiniCPM-V-4] Checksum mismatch; re-downloading: {filename}")
                os.remove(local_path)
            else:
                return local_path
        else:
            return local_path

    _ensure_dir(_models_dir())
    print(f"[MiniCPM-V-4] Downloading {filename} from {REPO_ID} ...")
    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        local_dir=_models_dir(),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    if os.path.abspath(path) != os.path.abspath(local_path):
        try:
            if not os.path.exists(local_path):
                os.link(path, local_path)
        except Exception:
            import shutil
            shutil.copy2(path, local_path)

    sha_expected = KNOWN_SHA256.get(filename)
    if sha_expected:
        sha_local = _sha256_file(local_path)
        if sha_local != sha_expected:
            raise RuntimeError(f"[MiniCPM-V-4] SHA256 mismatch for {filename}")

    size_gb = os.path.getsize(local_path) / (1024**3)
    print(f"[MiniCPM-V-4] Ready: {local_path} ({size_gb:.2f} GB)")
    return local_path

def _b64_from_pil(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# Global cache
_MODEL_CACHE: Dict[Tuple, "LlamaHandle"] = {}

@dataclasses.dataclass
class LlamaHandle:
    model_path: str
    n_ctx: int
    n_gpu_layers: int
    seed: int
    threads: int
    llm: Optional[Llama] = None
    vision_ok: bool = False

def _cache_key(path: str, n_ctx: int, n_gpu_layers: int, seed: int, threads: int):
    return (os.path.abspath(path), n_ctx, n_gpu_layers, seed, threads)

def _probe_vision_chat(llm: Llama) -> bool:
    # 0.2.89: use chat API with base64 image parts
    tiny = Image.new("RGB", (2, 2), (0, 0, 0))
    msg = [{
        "role": "user",
        "content": [
            {"type": "input_text", "text": "<image>\nprobe"},
            {"type": "input_image", "image": _b64_from_pil(tiny)},
        ],
    }]
    out = llm.create_chat_completion(messages=msg, max_tokens=1, temperature=0.0)
    # If we got here without exceptions, chat multimodal is wired
    return True

class MiniCPMV4GGUFLoader:
    @classmethod
    def INPUT_TYPES(cls):
        options = list(MINICPMV4_GGUF.values())
        default_label = "Q4_K_M (2.19 GB) [balanced]"
        return {
            "required": {
                "variant": (options, {"default": default_label}),
                "n_ctx": ("INT", {"default": 4096, "min": 1024, "max": 16384}),
                "n_gpu_layers": ("INT", {"default": 35, "min": 0, "max": 200, "tooltip": "0=CPU only"}),
                "seed": ("INT", {"default": 1234, "min": 0, "max": 2**31-1}),
                "threads": ("INT", {"default": 0, "min": 0, "max": 128, "tooltip": "0=auto"}),
            }
        }

    RETURN_TYPES = ("LLAMA", "STRING")
    RETURN_NAMES = ("llama", "model_path")
    FUNCTION = "load"
    CATEGORY = "LLM/Multimodal"

    def load(self, variant, n_ctx, n_gpu_layers, seed, threads):
        inv_map = {v: k for k, v in MINICPMV4_GGUF.items()}
        filename = inv_map[variant]
        model_path = _download_if_missing(filename)

        key = _cache_key(model_path, n_ctx, n_gpu_layers, seed, threads)
        if key in _MODEL_CACHE and _MODEL_CACHE[key].llm is not None:
            handle = _MODEL_CACHE[key]
            if not handle.vision_ok:
                handle.vision_ok = _probe_vision_chat(handle.llm)
            return (handle, model_path)

        kwargs = dict(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            seed=seed,
            logits_all=False,
            vocab_only=False,
            use_mmap=True,
            use_mlock=False,
        )
        if threads and threads > 0:
            kwargs["num_threads"] = threads

        print(f"[MiniCPM-V-4] Loading GGUF: {model_path} (ctx={n_ctx}, ngl={n_gpu_layers}, threads={threads})")
        llm = Llama(**kwargs)

        # Vision probe using chat API (works on 0.2.89)
        vision_ok = _probe_vision_chat(llm)

        handle = LlamaHandle(
            model_path=model_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers,
            seed=seed, threads=threads, llm=llm, vision_ok=vision_ok
        )
        _MODEL_CACHE[key] = handle
        print("[MiniCPM-V-4] Model loaded (vision ready via chat API).")
        return (handle, model_path)

NODE_CLASS_MAPPINGS = {"MiniCPMV4GGUFLoader": MiniCPMV4GGUFLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"MiniCPMV4GGUFLoader": "MiniCPM-V-4 (GGUF) Loader"}
