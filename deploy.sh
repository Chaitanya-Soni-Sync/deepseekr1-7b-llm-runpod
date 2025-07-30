#!/bin/bash

# RunPod Deployment Script
# This script helps deploy and test the DeepSeek R1 7B model on RunPod

set -e

echo "🚀 DeepSeek R1 7B RunPod Deployment Script"
echo "=========================================="

# Check if runpod CLI is installed
if ! command -v runpod &> /dev/null; then
    echo "❌ RunPod CLI not found. Please install it first:"
    echo "   pip install runpod"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "main.py" ]; then
    echo "❌ main.py not found. Please run this script from the project root."
    exit 1
fi

echo "✅ Environment check passed"

# Build and deploy
echo ""
echo "📦 Building and deploying to RunPod..."
runpod deploy

echo ""
echo "🎉 Deployment completed!"
echo ""
echo "📋 Next steps:"
echo "1. Test your endpoint with the provided test_input.json"
echo "2. Monitor logs for any issues"
echo "3. Check the README.md for troubleshooting tips"
echo ""
echo "🔧 To test locally first, run:"
echo "   python test_model.py" 