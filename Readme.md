# DeepSeek LLM RunPod Serverless

A production-ready serverless deployment of DeepSeek-R1-Distill-Qwen-7B on RunPod.

## Features

- ✅ Proper error handling and logging
- ✅ Optimized Docker image with CUDA support
- ✅ Local model caching to avoid repeated downloads
- ✅ Memory-efficient model loading
- ✅ Health checks and monitoring
- ✅ Development and production configurations

## Quick Start

### Local Development

```bash
# Build and run locally
docker-compose up --build

# Test the endpoint
curl -X POST http://localhost:8080 \
    -H 'Content-Type: application/json' \
    -d '{"input":{"prompt":"Hello, how are you?"}}'