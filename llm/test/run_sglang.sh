#!/bin/bash
# test_sglang.sh - Test SGLang with same config as miles

set -ex

# Kill any existing sglang processes
pkill -9 sglang || true
sleep 2

MODEL_PATH="/root/Qwen2.5-0.5B-Instruct/"
PORT=30000

# Launch SGLang server with same settings from your run
python -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --trust-remote-code \
    --port $PORT \
    --mem-fraction-static 0.7 \
    --attention-backend flashinfer \
    --disable-radix-cache \
    --random-seed 1234 \
    --enable-deterministic-inference \
    &

SERVER_PID=$!

# Wait for server to be ready
echo "Waiting for server to start..."
for i in {1..60}; do
    if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo "Server is ready!"
        break
    fi
    sleep 2
done

# Test generation with same prompt format as GSM8K eval
echo "Testing generation..."
curl -s "http://localhost:$PORT/generate" \
    -H "Content-Type: application/json" \
    -d '{
        "text": "<|im_start|>system\nYou are a helpful assistant. Please put the answer within \\boxed{}.<|im_end|>\n<|im_start|>user\nA restaurant has 40 tables with 4 legs and 50 tables with 3 legs. Calculate the total number of legs the restaurants tables have.<|im_end|>\n<|im_start|>assistant\n",
        "sampling_params": {
            "max_new_tokens": 256,
            "temperature": 1.0,
            "top_k": 1
        }
    }' | python -m json.tool

# Cleanup
echo "Stopping server..."
kill $SERVER_PID 2>/dev/null || true

echo "Done!"
