# ComfyUI/custom_nodes/ComfyUI_minicpmv4/__init__.py
from .loader import NODE_CLASS_MAPPINGS as LOADER_MAPPINGS
from .loader import NODE_DISPLAY_NAME_MAPPINGS as LOADER_DISPLAY
from .loader import LLAMA_CPP_AVAILABLE, LLAMA_CPP_ERROR

from .vision_infer import NODE_CLASS_MAPPINGS as INFER_MAPPINGS
from .vision_infer import NODE_DISPLAY_NAME_MAPPINGS as INFER_DISPLAY

NODE_CLASS_MAPPINGS = {
    **LOADER_MAPPINGS,
    **INFER_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **LOADER_DISPLAY,
    **INFER_DISPLAY,
}

print("\n" + "="*60)
print("MiniCPM-V-4 GGUF Node Loaded")
if LLAMA_CPP_AVAILABLE:
    print("✓ llama-cpp-python available")
else:
    print(f"✗ llama-cpp-python NOT available: {LLAMA_CPP_ERROR}")
    print("  Install: pip install llama-cpp-python")
print("="*60 + "\n")

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
