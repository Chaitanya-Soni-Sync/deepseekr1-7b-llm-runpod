# Use the official RunPod PyTorch image with CUDA and Python pre-installed
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Set environment variables for HuggingFace cache and unbuffered Python output
ENV HF_HOME=/runpod-volume/hf_cache
ENV TRANSFORMERS_CACHE=/runpod-volume/hf_cache
ENV HF_DATASETS_CACHE=/runpod-volume/hf_cache
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /workspace

# (Optional) Install git, wget, curl if you need them for your code or debugging
RUN apt-get update && apt-get install -y git wget curl && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all application files into the container
COPY . .

# Expose the HTTP port (matches your RunPod endpoint config)
EXPOSE 8080

# Start the serverless handler
CMD ["python", "main.py"]
