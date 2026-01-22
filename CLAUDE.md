# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

Personal monorepo for learning and experimentation, covering AI/ML, systems programming, and full-stack development. Key areas:

- **ml/**, **llm/**: PyTorch-based ML experiments, LLM fine-tuning, distributed training, inference optimization
- **python/**: Python language features and patterns
- **go/**, **rust/**: Systems programming experiments
- **cuda/**: GPU acceleration with CUDA 12.8, cuDNN, NCCL

## Common Commands

```bash
# Python (pytest-based testing)
pytest test_*.py -v
pytest -k "test_name" -v  # single test

# FastAPI server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Go
go test ./...
go run main.go

# Rust
cargo test
cargo run

# React
npm install && npm start
```

## Tech Stack

- **ML/DL**: PyTorch, Hugging Face Transformers, CUDA 12.8
- **Backend**: FastAPI, uvicorn
- **Frontend**: React 18
- **Testing**: pytest (Python), go test, cargo test

## Git Commit Prefixes

- `[feat]`: new feature
- `[fix]`: bug fix
- `[chore]`: routine tasks, maintenance, refactor
- `[docs]`: documentation changes
- `[style]`: formatting (no logic changes)
- `[perf]`: performance improvement
- `[test]`: adding or fixing tests
- `[build]`: build system or dependency changes
- `[ci]`: CI/CD configuration
- `[deps]`: dependency upgrades/downgrades
