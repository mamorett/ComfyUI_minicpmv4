# ComfyUI/custom_nodes/ComfyUI_minicpmv4/vision_infer.py
import base64
import io
import numpy as np
from PIL import Image
import re
import time
import hashlib

def _to_numpy(arr) -> np.ndarray:
    try:
        import torch
        if isinstance(arr, torch.Tensor):
            arr = arr.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(arr)

def _pil_from_comfy(batch) -> Image.Image:
    """Convert ComfyUI image batch to PIL Image (first image only)"""
    x = _to_numpy(batch)
    if x.ndim == 4:
        x = x[0]
    
    x = np.clip(x, 0.0, 1.0)
    x = (x * 255.0 + 0.5).astype(np.uint8)
    
    return Image.fromarray(x)

def _image_to_data_uri(image: Image.Image) -> str:
    """Convert PIL Image to base64 data URI"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    
    return f"data:image/png;base64,{img_base64}"

def _get_image_hash(image: Image.Image) -> str:
    """Get hash of image for cache busting"""
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return hashlib.md5(buffer.getvalue()).hexdigest()[:8]

def _clean_output(text: str) -> str:
    """Clean up model output"""
    if not text:
        return text
    
    patterns = [
        r'^[\s\-•*]+',
        r'^(?!1\.)\d+[\.\)\s\-]+',
        r'^(Assistant|User|MiniCPM|AI):\s*',
        r'^[A-Z][a-z]+:\s*',
        r'<\|im_start\|>.*?<\|im_end\|>',
        r'<image>',
        r'\[INST\].*?\[/INST\]',
    ]
    
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
    
    text = text.strip()
    return text

class MiniCPMV4VisionInfer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "handle": ("MINICPM_HANDLE",),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": "Describe this image in detail."}),
            },
            "optional": {
                "max_tokens": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 40, "min": 0, "max": 200}),
                "repeat_penalty": ("FLOAT", {"default": 1.1, "min": 0.0, "max": 2.0, "step": 0.01}),
                "force_new_context": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "run"
    CATEGORY = "LLM/Multimodal"

    def run(self, handle, image, prompt,
            max_tokens=512, temperature=0.7, top_p=0.9, top_k=40,
            repeat_penalty=1.1, force_new_context=True):

        if handle is None or not isinstance(handle, dict) or "llm" not in handle:
            raise RuntimeError("[MiniCPM-V-4] Invalid handle - model not loaded")

        llm = handle["llm"]
        
        # ✅ AGGRESSIVE CACHE CLEARING
        print(f"\n{'='*60}")
        print(f"[MiniCPM-V-4] Starting NEW vision inference")
        
        if force_new_context:
            print("[MiniCPM-V-4] Force clearing all caches...")
            
            # Method 1: Reset the model
            try:
                llm.reset()
                print("  ✓ Model reset")
            except Exception as e:
                print(f"  ✗ Model reset failed: {e}")
            
            # Method 2: Clear KV cache directly
            try:
                if hasattr(llm, '_ctx'):
                    llm._ctx.kv_cache_clear()
                    print("  ✓ KV cache cleared")
            except Exception as e:
                print(f"  ✗ KV cache clear failed: {e}")
            
            # Method 3: Clear eval cache
            try:
                if hasattr(llm, 'eval_tokens'):
                    llm.eval_tokens.clear()
                    print("  ✓ Eval tokens cleared")
            except Exception as e:
                print(f"  ✗ Eval tokens clear failed: {e}")
            
            # Method 4: Reset chat handler state
            try:
                chat_handler = handle.get('chat_handler')
                if chat_handler and hasattr(chat_handler, 'reset'):
                    chat_handler.reset()
                    print("  ✓ Chat handler reset")
            except Exception as e:
                print(f"  ✗ Chat handler reset failed: {e}")
        
        print(f"  Prompt: {prompt[:80]}...")
        print(f"  Max tokens: {max_tokens}")
        print(f"  Temperature: {temperature}")
        print(f"  Processing mode: {handle.get('processing_mode', 'unknown')}")
        print(f"{'='*60}")
        
        # Convert image
        print("[MiniCPM-V-4] Converting image...")
        pil_image = _pil_from_comfy(image)
        image_hash = _get_image_hash(pil_image)
        print(f"[MiniCPM-V-4] Image size: {pil_image.size}, mode: {pil_image.mode}, hash: {image_hash}")
        
        # Encode to data URI
        print("[MiniCPM-V-4] Encoding image to base64...")
        data_uri = _image_to_data_uri(pil_image)
        print(f"[MiniCPM-V-4] Base64 length: {len(data_uri)} chars")
        
        # ✅ Add unique identifier to prevent cache hits
        timestamp = int(time.time() * 1000)
        unique_prompt = f"{prompt.strip()}\n\n[Image ID: {image_hash}-{timestamp}]"
        
        # ✅ Message format with cache-busting
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_uri
                        }
                    },
                    {
                        "type": "text",
                        "text": unique_prompt
                    }
                ]
            }
        ]
        
        # ✅ Generation parameters with unique seed
        gen_params = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repeat_penalty": repeat_penalty,
            "stream": False,
            "seed": timestamp % (2**31),  # Unique seed per request
        }
        
        print("[MiniCPM-V-4] Generating response...")
        print(f"  Parameters: temp={temperature}, top_p={top_p}, top_k={top_k}, seed={gen_params['seed']}")
        
        try:
            response = llm.create_chat_completion(**gen_params)
            
            print(f"[MiniCPM-V-4] Response received")
            
            # Extract content
            content = ""
            try:
                if "choices" in response and len(response["choices"]) > 0:
                    choice = response["choices"][0]
                    finish_reason = choice.get('finish_reason')
                    print(f"[MiniCPM-V-4] Finish reason: {finish_reason}")
                    
                    if "message" in choice:
                        message = choice["message"]
                        content = message.get("content", "")
                    elif "text" in choice:
                        content = choice.get("text", "")
                        
                    # Log token usage
                    if "usage" in response:
                        usage = response["usage"]
                        prompt_tokens = usage.get('prompt_tokens', 0)
                        completion_tokens = usage.get('completion_tokens', 0)
                        print(f"[MiniCPM-V-4] Tokens - Prompt: {prompt_tokens}, Completion: {completion_tokens}")
                        
                        # ✅ Warn if completion is suspiciously short
                        if completion_tokens < 10:
                            print(f"[MiniCPM-V-4] ⚠️  WARNING: Very few completion tokens ({completion_tokens})")
                        
            except Exception as e:
                print(f"[MiniCPM-V-4] Error extracting content: {e}")
            
            # If still empty, try alternative extraction
            if not content:
                print("[MiniCPM-V-4] WARNING: Empty content, attempting alternative extraction...")
                print(f"[MiniCPM-V-4] Response: {response}")
                
                if isinstance(response, str):
                    content = response
                elif isinstance(response, dict):
                    if "text" in response:
                        content = response["text"]
                    elif "content" in response:
                        content = response["content"]
            
            # Clean the output
            if content:
                original_length = len(content)
                # Remove the cache-busting suffix before cleaning
                content = re.sub(r'\[Image ID:.*?\]', '', content, flags=re.IGNORECASE)
                content = _clean_output(content)
                print(f"[MiniCPM-V-4] Cleaned output: {original_length} -> {len(content)} chars")
            
            # Final check
            if not content or len(content.strip()) < 10:
                error_msg = (
                    f"[MiniCPM-V-4] ERROR: Model returned insufficient content.\n\n"
                    f"Content received: '{content}'\n"
                    f"Content length: {len(content) if content else 0}\n\n"
                    f"This suggests the model/mmproj combination is not working correctly.\n\n"
                    f"Troubleshooting steps:\n"
                    f"1. Verify MMPROJ file matches the model architecture\n"
                    f"2. Try Q8_0 quantization for testing\n"
                    f"3. Check llama-cpp-python version: pip show llama-cpp-python\n"
                    f"4. Ensure llama.cpp was built with LLAVA support\n"
                    f"5. Try a different image (simpler, smaller)\n"
                    f"6. Increase max_tokens to 1024+\n\n"
                    f"Debug info:\n"
                    f"  MMPROJ: {handle.get('mmproj_path')}\n"
                    f"  Processing: {handle.get('processing_mode')}\n"
                    f"  GPU layers: {handle.get('n_gpu_layers')}\n"
                    f"  Image: {pil_image.size}, hash: {image_hash}\n"
                    f"  Prompt length: {len(prompt)}\n"
                )
                print(error_msg)
                return (error_msg,)
            
            print(f"\n{'='*60}")
            print(f"[MiniCPM-V-4] ✓ Generated {len(content)} characters")
            print(f"  Preview: {content[:200]}...")
            print(f"  Image hash: {image_hash}")
            print(f"{'='*60}\n")
            
            return (content,)
            
        except Exception as e:
            import traceback
            error_msg = (
                f"[MiniCPM-V-4] Generation error: {e}\n\n"
                f"Traceback:\n{traceback.format_exc()}\n\n"
                f"Debug info:\n"
                f"  Model path: {getattr(llm, 'model_path', 'unknown')}\n"
                f"  MMPROJ path: {handle.get('mmproj_path', 'unknown')}\n"
                f"  Processing mode: {handle.get('processing_mode', 'unknown')}\n"
                f"  GPU layers: {handle.get('n_gpu_layers', 'unknown')}\n"
                f"  Has chat_handler: {handle.get('chat_handler') is not None}\n"
                f"  Image hash: {image_hash}\n"
            )
            print(error_msg)
            return (error_msg,)

NODE_CLASS_MAPPINGS = {"MiniCPMV4VisionInfer": MiniCPMV4VisionInfer}
NODE_DISPLAY_NAME_MAPPINGS = {"MiniCPMV4VisionInfer": "MiniCPM-V-4 (GGUF) Vision Infer"}
