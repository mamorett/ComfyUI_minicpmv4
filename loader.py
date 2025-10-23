# ComfyUI/custom_nodes/ComfyUI_minicpmv4/loader.py
import os
import sys
import io
import gc
import torch
import folder_paths
from huggingface_hub import hf_hub_download
from pathlib import Path

try:
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import Llava15ChatHandler
    LLAMA_CPP_AVAILABLE = True
    LLAMA_CPP_ERROR = None
except Exception as e:
    LLAMA_CPP_AVAILABLE = False
    LLAMA_CPP_ERROR = str(e)
    class Llama:
        pass
    class Llava15ChatHandler:
        pass

REPO_ID = "openbmb/MiniCPM-V-4-gguf"

MINICPMV4_GGUF = {
    "Q4_0 (2.08 GB)": "ggml-model-Q4_0.gguf",
    "Q4_1 (2.29 GB)": "ggml-model-Q4_1.gguf",
    "Q4_K_M (2.19 GB) [recommended]": "ggml-model-Q4_K_M.gguf",
    "Q4_K_S (2.09 GB)": "ggml-model-Q4_K_S.gguf",
    "Q5_0 (2.51 GB)": "ggml-model-Q5_0.gguf",
    "Q5_1 (2.72 GB)": "ggml-model-Q5_1.gguf",
    "Q5_K_M (2.56 GB) [quality]": "ggml-model-Q5_K_M.gguf",
    "Q5_K_S (2.51 GB)": "ggml-model-Q5_K_S.gguf",
    "Q6_K (2.96 GB)": "ggml-model-Q6_K.gguf",
    "Q8_0 (3.83 GB) [max quality]": "ggml-model-Q8_0.gguf",
}

MMPROJ_FILENAME = "mmproj-model-f16.gguf"

_MODEL_CACHE = {}

def _models_dir() -> str:
    models_dir = Path(folder_paths.models_dir).resolve()
    llm_dir = (models_dir / "LLM" / "GGUF").resolve()
    llm_dir.mkdir(parents=True, exist_ok=True)
    return str(llm_dir)

def _download_if_missing(filename: str) -> str:
    local_dir = _models_dir()
    local_path = os.path.join(local_dir, filename)
    
    if os.path.exists(local_path):
        return local_path

    print(f"[MiniCPM-V-4] Downloading {filename}...")
    downloaded_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    return downloaded_path

class MiniCPMV4GGUFLoader:
    @classmethod
    def INPUT_TYPES(cls):
        if not LLAMA_CPP_AVAILABLE:
            return {
                "required": {
                    "error": ("STRING", {"default": f"llama-cpp-python not available: {LLAMA_CPP_ERROR}"})
                }
            }
        
        return {
            "required": {
                "model": (list(MINICPMV4_GGUF.keys()), {"default": "Q4_K_M (2.19 GB) [recommended]"}),
                "processing_mode": (["Auto", "GPU", "CPU"], {"default": "Auto"}),
            }
        }

    RETURN_TYPES = ("MINICPM_HANDLE",)
    FUNCTION = "load"
    CATEGORY = "LLM/Multimodal"

    def load(self, model, processing_mode):
        if not LLAMA_CPP_AVAILABLE:
            raise RuntimeError(f"llama-cpp-python not available: {LLAMA_CPP_ERROR}")
        
        filename = MINICPMV4_GGUF[model]
        model_path = _download_if_missing(filename)
        mmproj_path = _download_if_missing(MMPROJ_FILENAME)
        
        if processing_mode == "Auto":
            processing_mode = "GPU" if torch.cuda.is_available() else "CPU"
        
        cache_key = f"{model_path}_{processing_mode}"
        
        if cache_key in _MODEL_CACHE:
            print(f"[MiniCPM-V-4] Using cached model")
            return (_MODEL_CACHE[cache_key],)
        
        n_ctx = 4096
        n_batch = 2048
        n_threads = max(4, os.cpu_count() // 2) if os.cpu_count() else 4
        n_gpu_layers = -1 if processing_mode == "GPU" else 0
        
        print(f"[MiniCPM-V-4] Loading {model} on {processing_mode}")
        print(f"[MiniCPM-V-4] GPU layers: {n_gpu_layers}")
        
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        
        try:
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()
            
            llm = Llama(
                model_path=str(model_path),
                n_ctx=n_ctx,
                n_batch=n_batch,
                n_threads=n_threads,
                n_gpu_layers=n_gpu_layers,
                verbose=True,
                chat_handler=Llava15ChatHandler(clip_model_path=str(mmproj_path)),
                offload_kqv=True,
                numa=False,
                use_mlock=False,
                use_mmap=True,
                n_parts=1,
            )
            
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        
        handle = {
            "llm": llm,
            "n_gpu_layers": n_gpu_layers,
            "processing_mode": processing_mode
        }
        
        _MODEL_CACHE[cache_key] = handle
        
        print(f"[MiniCPM-V-4] ✓ Loaded successfully on {processing_mode}")
        
        return (handle,)

NODE_CLASS_MAPPINGS = {"MiniCPMV4GGUFLoader": MiniCPMV4GGUFLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"MiniCPMV4GGUFLoader": "MiniCPM-V-4 GGUF Loader"}
