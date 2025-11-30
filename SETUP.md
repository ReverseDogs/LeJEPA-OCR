# Setup Instructions

## ✅ Code Already Pushed to GitHub

Repository: https://github.com/ReverseDogs/LeJEPA-OCR

## RunPod Setup

SSH into your RunPod instance:

```bash
ssh ptkallityvuttu-64411168@ssh.runpod.io -i ~/.ssh/id_ed25519_runpod
```

### 1. Clone the repository on RunPod

```bash
cd /workspace
git clone https://github.com/ReverseDogs/LeJEPA-OCR.git
cd LeJEPA-OCR
```

### 2. Set up cache directories

```bash
# Create cache directory on your persistent volume
mkdir -p /workspace/cache/hf /workspace/cache/wds

# Set environment variables (add to ~/.bashrc to persist)
export HF_HOME=/workspace/cache/hf
export TRANSFORMERS_CACHE=/workspace/cache/hf
export WDS_CACHE_BASE=/workspace/cache/wds

# Make it permanent
echo 'export HF_HOME=/workspace/cache/hf' >> ~/.bashrc
echo 'export TRANSFORMERS_CACHE=/workspace/cache/hf' >> ~/.bashrc
echo 'export WDS_CACHE_BASE=/workspace/cache/wds' >> ~/.bashrc
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify tokenizer

```bash
python - <<'PY'
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("tencent/HunyuanOCR", trust_remote_code=True)
print("Tokenizer loaded successfully!")
print("Additional special tokens:", tok.additional_special_tokens)
PY
```

### 5. Run Phase 2 Training (LeJEPA)

```bash
python train_lejepa.py \
  --backbone vit_large_patch16_siglip_384 \
  --batch-size 4 \
  --grad-accum 32 \
  --max-res 1024 \
  --max-steps 5000 \
  --log-every 50 \
  --lambda-sig 0.05
```

**If you get OOM errors:**
- Try `--max-res 896` first
- If still OOM, try `--max-res 768`
- Or reduce `--batch-size 2`

### 6. Run Phase 3 Training (Alignment) - After Phase 2 checkpoint exists

```bash
python train_alignment.py \
  --dataset ocr_vqa \
  --checkpoint checkpoints/phase2/step_5000.pt \
  --batch-size 2 \
  --grad-accum 16 \
  --stage-a-steps 1000 \
  --stage-b-steps 2000 \
  --max-len 512 \
  --max-res 1024 \
  --log-every 50 \
  --tokenizer tencent/HunyuanOCR
```

**Alternative dataset:**
For DocVQA, use `--dataset docvqa` instead of `ocr_vqa`

## Monitoring Training

You can monitor training progress by:
- Checking the console output (logged every 50 steps)
- Checkpoints are saved to `checkpoints/phase2/` and `checkpoints/phase3/`

## Syncing Changes

If you make changes locally and want to sync to RunPod:

```bash
# On local machine
git add .
git commit -m "Your commit message"
git push

# On RunPod
cd /workspace/LeJEPA-OCR
git pull
```
