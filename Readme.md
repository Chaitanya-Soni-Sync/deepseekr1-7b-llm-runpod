# DeepSeek R1 7B LLM on RunPod

This repository contains a RunPod serverless implementation of the DeepSeek-R1-Distill-Qwen-7B model with comprehensive error handling and fallback strategies.

## Features

- **Robust Error Handling**: Implements fallback strategies for probability tensor errors
- **Safe Parameter Validation**: Prevents invalid generation parameters
- **Automatic Model Caching**: Saves model locally for faster subsequent loads
- **Comprehensive Logging**: Detailed logging for debugging and monitoring

## Recent Fixes (v2.0)

### Probability Tensor Error Resolution

The main issue addressed in this version is the "probability tensor contains either `inf`, `nan` or element < 0" error. This has been resolved through:

1. **Safer Parameter Bounds**:
   - Temperature: clamped to 0.5-1.5 (was 0.1-2.0)
   - Top-p: clamped to 0.5-1.0 (was 0.1-1.0)
   - Max tokens: limited to 1024

2. **Fallback Generation Strategy**:
   - Primary attempt with user parameters
   - Fallback 1: Greedy decoding with temperature=1.0
   - Fallback 2: Minimal parameters with reduced max tokens

3. **Enhanced Error Detection**:
   - Specific handling for probability tensor errors
   - Graceful degradation to safer generation methods
   - Comprehensive logging of fallback attempts

## Quick Start

### 1. Deploy to RunPod

```bash
# Build and deploy
runpod deploy
```

### 2. Test the Endpoint

```bash
# Test with basic prompt
curl -X POST "YOUR_ENDPOINT_URL" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "Hello, how are you today?",
      "max_new_tokens": 100,
      "temperature": 0.7,
      "top_p": 0.9,
      "do_sample": true
    }
  }'
```

### 3. Local Testing

```bash
# Run local test script
```

## API Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `prompt` | string | required | 1-8000 chars | Input text for generation |
| `max_new_tokens` | integer | 512 | 1-1024 | Maximum tokens to generate |
| `temperature` | float | 0.7 | 0.5-1.5 | Controls randomness (higher = more random) |
| `top_p` | float | 0.9 | 0.5-1.0 | Nucleus sampling parameter |
| `do_sample` | boolean | true | true/false | Whether to use sampling vs greedy decoding |

## Troubleshooting

### Common Issues

1. **Probability Tensor Error**
   - ✅ **FIXED**: Automatic fallback to safer parameters
   - If still occurring, check input prompt length and content

2. **Model Loading Issues**
   - Ensure sufficient disk space (>10GB recommended)
   - Check internet connectivity for initial model download
   - Verify GPU memory availability

3. **Memory Issues**
   - Reduce `max_new_tokens` parameter
   - Use shorter input prompts
   - Consider using quantization (uncomment in requirements.txt)

### Debug Mode

Enable detailed logging by setting the log level:

```python
logging.basicConfig(level=logging.DEBUG)
```

### Performance Optimization

1. **For Production**:
   - Uncomment `bitsandbytes` in requirements.txt for quantization
   - Use `flash-attn` for faster attention computation
   - Consider model sharding for large models

2. **For Development**:
   - Use local model caching (already implemented)
   - Test with smaller parameter ranges first

## Error Handling Strategy

The implementation uses a three-tier fallback system:

1. **Primary**: User-specified parameters with validation
2. **Fallback 1**: Greedy decoding with safe temperature
3. **Fallback 2**: Minimal parameters with reduced complexity

This ensures the service remains operational even when encountering numerical instability issues.

## Monitoring

Key metrics to monitor:
- Generation success rate
- Fallback usage frequency
- Response times
- Memory usage
- Error rates by error type

## Support

For issues related to:
- **Probability tensor errors**: ✅ Resolved in this version
- **Model loading**: Check disk space and network connectivity
- **Performance**: Consider quantization and optimization flags
- **API usage**: Refer to parameter documentation above

## Changelog

### v2.0 (Current)
- ✅ Fixed probability tensor errors with fallback strategies
- ✅ Improved parameter validation and bounds
- ✅ Enhanced error handling and logging
- ✅ Added comprehensive test suite
- ✅ Updated documentation and troubleshooting guide

### v1.0
- Initial implementation
- Basic error handling
- Model caching support
