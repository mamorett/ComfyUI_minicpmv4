# ComfyUI/custom_nodes/ComfyUI_minicpmv4/vision_infer.py
import base64
import io
import numpy as np
from PIL import Image
import re

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
        x = x[0]  # Take first image
    
    # Clip and convert to uint8
    x = np.clip(x, 0.0, 1.0)
    x = (x * 255.0 + 0.5).astype(np.uint8)
    
    return Image.fromarray(x)

def _image_to_data_uri(image: Image.Image) -> str:
    """Convert PIL Image to base64 data URI"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize to 336x336 (standard for vision models)
    image = image.resize((336, 336), Image.Resampling.BILINEAR)
    
    # Convert to base64
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    
    return f"data:image/png;base64,{img_base64}"

def _clean_output(text: str) -> str:
    """Clean up model output"""
    if not text:
        return text
    
    # Remove common prefixes
    patterns = [
        r'^[\s\-•*]+',
        r'^(?!1\.)\d+[\.\)\s\-]+',
        r'^(Assistant|User|MiniCPM|AI):\s*',
        r'^[A-Z][a-z]+:\s*'
    ]
    
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
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
                "system_prompt": ("STRING", {"multiline": True, "default": "You are a helpful AI assistant."}),
                "max_tokens": ("INT", {"default": 1024, "min": 64, "max": 4096}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 40, "min": 0, "max": 200}),
                "repeat_penalty": ("FLOAT", {"default": 1.05, "min": 0.0, "max": 2.0, "step": 0.01}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "run"
    CATEGORY = "LLM/Multimodal"

    def run(self, handle, image, prompt,
            system_prompt="You are a helpful AI assistant.",
            max_tokens=1024, temperature=0.7, top_p=0.8, top_k=40,
            repeat_penalty=1.05, seed=-1):

        if handle is None or not isinstance(handle, dict) or "llm" not in handle:
            raise RuntimeError("[MiniCPM-V-4] Invalid handle - model not loaded")

        llm = handle["llm"]
        
        print(f"\n{'='*60}")
        print(f"[MiniCPM-V-4] Starting inference")
        print(f"  Prompt: {prompt[:80]}...")
        print(f"  Max tokens: {max_tokens}")
        print(f"  Temperature: {temperature}")
        print(f"  GPU layers: {handle.get('n_gpu_layers', 'unknown')}")
        print(f"{'='*60}")
        
        # Convert ComfyUI image to PIL
        print("[MiniCPM-V-4] Converting image...")
        pil_image = _pil_from_comfy(image)
        print(f"[MiniCPM-V-4] Image size: {pil_image.size}, mode: {pil_image.mode}")
        
        # Convert image to base64 data URI
        print("[MiniCPM-V-4] Encoding image to base64...")
        data_uri = _image_to_data_uri(pil_image)
        print(f"[MiniCPM-V-4] Base64 length: {len(data_uri)} chars")
        
        # Build messages in chat format (simplified - no system message in user content)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {"type": "text", "text": prompt.strip()}
                ]
            }
        ]
        
        # Prepare generation parameters (simplified)
        gen_params = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repeat_penalty": repeat_penalty,
            "stream": False,
        }
        
        if isinstance(seed, int) and seed >= 0:
            gen_params["seed"] = seed
        
        print("[MiniCPM-V-4] Generating response...")
        print(f"  Parameters: temp={temperature}, top_p={top_p}, top_k={top_k}")
        
        try:
            # Generate response
            response = llm.create_chat_completion(**gen_params)
            
            print(f"[MiniCPM-V-4] Raw response received")
            print(f"  Response keys: {response.keys()}")
            
            # Extract content - try multiple paths
            content = ""
            
            # Path 1: Standard chat completion format
            try:
                if "choices" in response and len(response["choices"]) > 0:
                    choice = response["choices"][0]
                    print(f"  Choice keys: {choice.keys()}")
                    
                    if "message" in choice:
                        message = choice["message"]
                        print(f"  Message keys: {message.keys()}")
                        content = message.get("content", "")
                    elif "text" in choice:
                        content = choice["text"]
                    
                    print(f"  Extracted content length: {len(content)}")
            except Exception as e:
                print(f"  Error extracting content: {e}")
            
            # If still empty, try alternative extraction
            if not content:
                print("[MiniCPM-V-4] Content empty, trying alternative extraction...")
                try:
                    # Sometimes the response is directly in the dict
                    if "content" in response:
                        content = response["content"]
                    elif "text" in response:
                        content = response["text"]
                except Exception as e:
                    print(f"  Alternative extraction failed: {e}")
            
            # If STILL empty, print full response for debugging
            if not content:
                print("[MiniCPM-V-4] WARNING: Empty content!")
                print(f"  Full response: {response}")
                
                # Try one more time with different parameters
                print("[MiniCPM-V-4] Retrying with adjusted parameters...")
                retry_params = {
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": 0.8,
                    "top_p": 0.9,
                    "stream": False,
                }
                
                retry_response = llm.create_chat_completion(**retry_params)
                print(f"  Retry response: {retry_response}")
                
                try:
                    content = retry_response["choices"][0]["message"]["content"]
                except Exception:
                    try:
                        content = retry_response["choices"][0]["text"]
                    except Exception:
                        pass
            
            # Clean output
            if content:
                content = _clean_output(content)
            
            if not content:
                error_msg = (
                    "[MiniCPM-V-4] ERROR: Model returned empty response.\n\n"
                    "Possible causes:\n"
                    "1. Model not compatible with llama-cpp-python version\n"
                    "2. Vision projector not loading correctly\n"
                    "3. Image format issue\n\n"
                    "Try:\n"
                    "- Rebuild llama-cpp-python with CUDA support\n"
                    "- Use different quantization (Q4_K_M or Q5_K_M)\n"
                    "- Check if model files are corrupted\n"
                )
                print(error_msg)
                return (error_msg,)
            
            print(f"\n{'='*60}")
            print(f"[MiniCPM-V-4] ✓ Generated {len(content)} characters")
            print(f"  Preview: {content[:100]}...")
            print(f"{'='*60}\n")
            
            return (content,)
            
        except Exception as e:
            import traceback
            error_msg = f"[MiniCPM-V-4] Generation error: {e}\n\n{traceback.format_exc()}"
            print(error_msg)
            return (error_msg,)

NODE_CLASS_MAPPINGS = {"MiniCPMV4VisionInfer": MiniCPMV4VisionInfer}
NODE_DISPLAY_NAME_MAPPINGS = {"MiniCPMV4VisionInfer": "MiniCPM-V-4 (GGUF) Vision Infer"}

