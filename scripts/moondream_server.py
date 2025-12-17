#!/usr/bin/env ../venv/bin/python3
"""
Moondream2 Gaze Detection Server
Communicates with Rust via JSON over stdin/stdout
"""

import sys
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import io
import base64

def log_debug(msg):
    """Debug logging to stderr (won't interfere with stdout JSON)"""
    print(f"[MOONDREAM] {msg}", file=sys.stderr, flush=True)

def main():
    log_debug("Initializing Moondream2...")
    
    # Load model and tokenizer
    model_id = "vikhyatk/moondream2"
    revision = "2025-06-21"  # Latest revision with fixed API
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        log_debug("Model loaded successfully!")
    except Exception as e:
        log_debug(f"ERROR loading model: {e}")
        sys.exit(1)
    
    log_debug("Ready for requests. Waiting for stdin...")
    
    # Process requests from stdin
    for line in sys.stdin:
        try:
            line = line.strip()
            if not line:
                continue
                
            request = json.loads(line)
            
            # Load image from path
            image_path = request.get('image_path')
            if not image_path:
                raise ValueError("Missing 'image_path' in request")
            
            image = Image.open(image_path).convert('RGB')
            
            # Try different Moondream methods - prioritize point() for coordinates
            try:
                # BEST: Use point() API for direct coordinates
                if hasattr(model, 'point'):
                    # Moondream's point() method returns actual x,y coordinates
                    # Try multiple prompts to see what works
                    prompts_for_point = [
                        "the direction the person is looking",
                        "where the person's eyes are directed",
                        "the person's gaze target",
                        "what the person is looking at"
                    ]
                    
                    param_found = False
                    for prompt in prompts_for_point:
                        result_dict = model.point(image, prompt)
                        points = result_dict.get("points", [])
                        
                        if points and len(points) > 0:
                            # Got coordinates! Use first point
                            x, y = points[0]
                            response = f"COORDS:{x:.4f},{y:.4f}"
                            log_debug(f"point() returned coordinates: ({x:.4f}, {y:.4f}) with prompt '{prompt}'")
                            param_found = True
                            break
                            
                    if not param_found:
                        # point() didn't return coordinates, fall back to query
                        log_debug("point() returned no coordinates with any prompt, falling back to query()")
                        query_prompts = [
                            "Where is the person looking? Give me screen coordinates.",
                            "What direction is the person's gaze pointing?"
                        ]
                        result_dict = model.query(image, query_prompts[0])
                        response = result_dict.get("answer", str(result_dict))
                
                # Fallback to query if point not available
                elif hasattr(model, 'query'):
                    query_prompts = [
                        "Where is the person looking? Give me screen coordinates.",
                        "What direction is the person's gaze pointing?"
                    ]
                    result_dict = model.query(image, query_prompts[0])
                    response = result_dict.get("answer", str(result_dict))
                
                # Last resort: caption
                elif hasattr(model, 'caption'):
                    caption_dict = model.caption(image, tokenizer, length="normal")
                    response = f"[CAPTION FALLBACK] {caption_dict.get('caption', str(caption_dict))}"
                else:
                    # Log available methods for debugging
                    available = [m for m in dir(model) if not m.startswith('_')]
                    raise AttributeError(f"Unknown Moondream API. Available methods: {available[:20]}")
            except Exception as api_error:
                raise Exception(f"Moondream API error: {api_error}")
            
            # Parse response and create result
            result = {
                "status": "success",
                "response": response,
                "timestamp": request.get('timestamp', 0),
                "method": "point" if response.startswith("COORDS:") else "query"
            }
            
            # Write result to stdout
            print(json.dumps(result), flush=True)
            log_debug(f"Processed request: {response[:50]}...")
            
        except Exception as e:
            error_result = {
                "status": "error",
                "error": str(e),
                "timestamp": request.get('timestamp', 0) if 'request' in locals() else 0
            }
            print(json.dumps(error_result), flush=True)
            log_debug(f"ERROR: {e}")

if __name__ == "__main__":
    main()
