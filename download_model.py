import os
import logging
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
LOCAL_PATH = "/workspace/model"
CACHE_DIR = "/workspace/hf_cache"

# Set environment variables
os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR

def download_model():
    """Download and save model locally"""
    try:
        # Create directories
        Path(LOCAL_PATH).mkdir(parents=True, exist_ok=True)
        Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Downloading tokenizer for {MODEL_NAME}...")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, 
            trust_remote_code=True,
            cache_dir=CACHE_DIR
        )
        tokenizer.save_pretrained(LOCAL_PATH)
        logger.info("Tokenizer downloaded and saved successfully")
        
        logger.info(f"Downloading model for {MODEL_NAME}...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, 
            trust_remote_code=True,
            cache_dir=CACHE_DIR,
            torch_dtype="auto",
            low_cpu_mem_usage=True
        )
        model.save_pretrained(LOCAL_PATH)
        logger.info("Model downloaded and saved successfully")
        
    except Exception as e:
        logger.error(f"Failed to download model: {str(e)}")
        raise

if __name__ == "__main__":
    download_model()