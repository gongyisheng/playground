# Deployment
## File Formats
```
gguf: used for cpu inference
onnx: used for deployment, run on every device
pt: used for research (python pickle), has vulnerability and performance issues
safetensor: used for community sharing (huggingface)
tensorRT: used for deployment, bind with gpu device and tensorRT version
```

## Lifecycle
research: .pt
open-source: .safetensor
normalize: onnx
deployment: 
    cloud: tensorRT
    device: gguf

## Reference
https://donge.org/posts/2025-module-format/