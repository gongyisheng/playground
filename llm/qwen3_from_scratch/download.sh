#!/bin/bash
pip install huggingface_hub
huggingface-cli download Qwen/Qwen3-0.6B --local-dir ./checkpoint
