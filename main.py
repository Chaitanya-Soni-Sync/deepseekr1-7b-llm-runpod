import runpod
import torch
import os
import logging
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

# Change these lines in main.py
CACHE_DIR = "/runpod-volume/hf_cache"  # was "/workspace/hf_cache"
LOCAL_MODEL_PATH = "/runpod-volume/model"  # was "/workspace/model"

# Set environment variables for HuggingFace cache
os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR

def setup_directories():
    """Create necessary directories"""
    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
    Path(LOCAL_MODEL_PATH).mkdir(parents=True, exist_ok=True)

def load_model_and_tokenizer():
    """Load model and tokenizer with proper error handling"""
    try:
        logger.info("Loading tokenizer and model...")
        
        # Try to load from local path first, fallback to remote
        if Path(LOCAL_MODEL_PATH).exists() and any(Path(LOCAL_MODEL_PATH).iterdir()):
            logger.info(f"Loading model from local path: {LOCAL_MODEL_PATH}")
            tokenizer = AutoTokenizer.from_pretrained(
                LOCAL_MODEL_PATH, 
                trust_remote_code=True,
                local_files_only=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                LOCAL_MODEL_PATH,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
                local_files_only=True,
                low_cpu_mem_usage=True
            )
        else:
            logger.info(f"Loading model from HuggingFace: {MODEL_NAME}")
            tokenizer = AutoTokenizer.from_pretrained(
                MODEL_NAME, 
                trust_remote_code=True,
                cache_dir=CACHE_DIR
            )
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
                cache_dir=CACHE_DIR,
                low_cpu_mem_usage=True
            )
        
        model.eval()
        logger.info("Model loaded successfully")
        return tokenizer, model
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

def run_inference(model, tokenizer, prompt):
    """Run model inference with proper error handling"""
    try:
        with torch.no_grad():
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=1000,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the input prompt from the output
            response = generated_text[len(prompt):].strip()
            return response
            
    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")
        raise

# Initialize model and tokenizer globally
setup_directories()
tokenizer, model = load_model_and_tokenizer()

def handler(event):
    """Main handler function for RunPod serverless"""
    logger.info("Handler started")
    
    try:
        # Extract and validate input
        user_input = event.get("input", {})
        prompt = user_input.get("prompt")
        
        if not prompt:
            return {
                "status": "error",
                "message": "Missing 'prompt' in input"
            }
        
        if not isinstance(prompt, str) or len(prompt.strip()) == 0:
            return {
                "status": "error",
                "message": "Prompt must be a non-empty string"
            }
        
        # Run inference
        generated_text = run_inference(model, tokenizer, prompt)
        
        return {
            "status": "success",
            "input_prompt": prompt,
            "generated_text": generated_text
        }
        
    except Exception as e:
        logger.error(f"Handler error: {str(e)}")
        return {
            "status": "error",
            "message": f"Internal server error: {str(e)}"
        }

if __name__ == '__main__':
    runpod.serverless.start({
        'handler': handler
    })
