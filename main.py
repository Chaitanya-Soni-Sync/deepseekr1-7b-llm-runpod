import runpod
import torch
import os
import logging
from pathlib import Path
import shutil
from transformers import AutoTokenizer, AutoModelForCausalLM  # ✅ ADD THIS LINE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
CACHE_DIR = "/runpod-volume/hf_cache"
LOCAL_MODEL_PATH = "/runpod-volume/model"

# Set environment variables for HuggingFace cache
os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.environ["HF_HUB_CACHE"] = CACHE_DIR

def setup_directories():
    """Create necessary directories and clear existing cache."""
    try:
        logger.info(f"Attempting to clear and set up directories: {CACHE_DIR}, {LOCAL_MODEL_PATH}")
        
        if Path(CACHE_DIR).exists():
            logger.info(f"Clearing existing cache in {CACHE_DIR}...")
            shutil.rmtree(CACHE_DIR, ignore_errors=True)
        
        if Path(LOCAL_MODEL_PATH).exists():
            logger.info(f"Clearing existing model in {LOCAL_MODEL_PATH}...")
            shutil.rmtree(LOCAL_MODEL_PATH, ignore_errors=True)
        
        Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
        Path(LOCAL_MODEL_PATH).mkdir(parents=True, exist_ok=True)
        logger.info(f"Directories created: {CACHE_DIR}, {LOCAL_MODEL_PATH}")
    except Exception as e:
        logger.error(f"Failed to create/clear directories: {e}")
        raise

def check_disk_space():
    """Check available disk space"""
    try:
        total, used, free = shutil.disk_usage("/runpod-volume")
        free_gb = free // (1024**3)
        logger.info(f"Available disk space: {free_gb}GB")
        if free_gb < 5:
            logger.warning(f"Low disk space: {free_gb}GB remaining")
        return free_gb
    except Exception as e:
        logger.warning(f"Could not check disk space: {e}")
        return None

def load_model_and_tokenizer():
    """Load model and tokenizer with proper error handling"""
    try:
        logger.info("Starting model loading process...")
        check_disk_space()
        
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
                device_map = {"": 0},
                torch_dtype=torch.bfloat16,
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
                device_map = {"": 0},
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                cache_dir=CACHE_DIR,
                low_cpu_mem_usage=True
            )
            
            # Save model locally for future use
            logger.info(f"Saving model to local path: {LOCAL_MODEL_PATH}")
            model.save_pretrained(LOCAL_MODEL_PATH)
            tokenizer.save_pretrained(LOCAL_MODEL_PATH)
        
        # ✅ CRITICAL FIX: Set pad_token if missing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Set pad_token to eos_token")
        
        model.eval()
        logger.info("Model loaded and set to eval mode successfully")
        
        # Log model info
        logger.info(f"Model device: {model.device}")
        logger.info(f"Model dtype: {model.dtype}")
        
        return tokenizer, model
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

def run_inference(model, tokenizer, prompt, **kwargs):
    """Run model inference with proper error handling"""
    try:
        # ✅ SAFE PARAMETER VALIDATION
        max_new_tokens = min(kwargs.get('max_new_tokens', 512), 1024)
        temperature = max(0.1, min(kwargs.get('temperature', 0.7), 2.0))  # Clamp 0.1-2.0
        top_p = max(0.1, min(kwargs.get('top_p', 0.9), 1.0))  # Clamp 0.1-1.0
        do_sample = kwargs.get('do_sample', True)
        
        logger.info(f"Running inference with prompt length: {len(prompt)}")
        logger.info(f"Params - temp: {temperature}, top_p: {top_p}, max_tokens: {max_new_tokens}")
        
        # ✅ INPUT VALIDATION
        if not prompt or not prompt.strip():
            raise ValueError("Empty prompt provided")
        
        with torch.no_grad():
            # ✅ PROPER TOKENIZATION WITH TRUNCATION
            inputs = tokenizer(
                prompt.strip(), 
                return_tensors="pt", 
                padding=True,
                truncation=True,
                max_length=2048
            ).to(model.device)
            
            # ✅ DEBUG LOGGING
            logger.info(f"Input shape: {inputs['input_ids'].shape}")
            logger.info(f"pad_token_id: {tokenizer.pad_token_id}, eos_token_id: {tokenizer.eos_token_id}")
            
            # Check for potential issues with input
            if inputs['input_ids'].shape[1] > 2048:
                logger.warning(f"Input length ({inputs['input_ids'].shape[1]}) may be too long")
            
            # ✅ SAFER GENERATION WITH MORE PARAMETERS
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else 1.0,
                top_p=top_p if do_sample else 1.0,
                pad_token_id=tokenizer.pad_token_id,  # Use pad_token_id instead of eos
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,  # Prevent repetition
                use_cache=True,
                output_scores=False,  # Save memory
                return_dict_in_generate=False
            )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the input prompt from the output
            response = generated_text[len(prompt):].strip()
            
            logger.info(f"Generated response length: {len(response)}")
            return response
            
    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        raise

# Initialize model and tokenizer globally
logger.info("Initializing application...")
setup_directories()

try:
    tokenizer, model = load_model_and_tokenizer()
    logger.info("Application initialization complete")
except Exception as e:
    logger.error(f"Failed to initialize model: {e}")
    raise

def handler(event):
    """Main handler function for RunPod serverless"""
    logger.info("Handler started")
    try:
        # Extract and validate input
        user_input = event.get("input", {})
        prompt = user_input.get("prompt")
        
        # Input validation
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
        
        # ✅ VALIDATE PROMPT LENGTH
        if len(prompt) > 8000:
            return {
                "status": "error",
                "message": "Prompt too long (max 8000 characters)"
            }
        
        # ✅ SAFE PARAMETER EXTRACTION
        generation_params = {
            'max_new_tokens': min(user_input.get('max_new_tokens', 512), 1024),
            'temperature': max(0.1, min(user_input.get('temperature', 0.7), 2.0)),
            'top_p': max(0.1, min(user_input.get('top_p', 0.9), 1.0)),
            'do_sample': user_input.get('do_sample', True)
        }
        
        # Run inference
        generated_text = run_inference(model, tokenizer, prompt, **generation_params)
        
        return {
            "status": "success",
            "input_prompt": prompt,
            "generated_text": generated_text,
            "generation_params": generation_params
        }
        
    except Exception as e:
        logger.error(f"Handler error: {str(e)}")
        return {
            "status": "error",
            "message": f"Internal server error: {str(e)}"
        }

if __name__ == '__main__':
    logger.info("Starting RunPod serverless...")
    runpod.serverless.start({
        'handler': handler
    })
