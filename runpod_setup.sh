#!/bin/bash
# runpod_setup.sh - RunPod Cloud GPU Environment Setup Environment Script

echo "=== System Optimization & Dependency Allocation Initiated ==="

# 1. Update system package repository registry indices
apt-get update && apt-get install -y ffmpeg libsndfile1 git git-lfs

# 2. Force install production dependencies mapping matching repository configurations
pip install --upgrade pip
pip install torch torchaudio pydub fastapi uvicorn websockets runpod

# 3. Double-check GPU compute environment matrix status
python3 -c "
import torch
print('\n--- Cloud GPU Hardware Verification ---')
print(f'CUDA Acceleration Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Active Engine Hardware Card: {torch.cuda.get_device_name(0)}')
"

echo "=== Environment Configuration Completed. Ready for Training Operations ==="
