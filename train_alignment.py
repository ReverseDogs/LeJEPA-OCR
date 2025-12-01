"""
Phase 3: Adapter Bridge alignment.

Stage A (Probe):
- Load Phase 2 ViT weights, drop projector.
- Freeze ViT and decoder; train only the Adaptive MLP Adapter with lr=1e-3.

Stage B (Handshake):
- Unfreeze ViT with lr=1e-6.
- Train adapter/decoder with lr=1e-5 end-to-end.
"""

import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.data import TextImagePairStream
from src.model import AdapterConfig, DecoderConfig, HunyuanLeJEPA, ProjectorConfig, VisionConfig


def causal_mask(size: int, device) -> torch.Tensor:
    mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
    return mask.bool()


def build_dataloader(
    dataset_name: str,
    tokenizer,
    batch_size: int,
    workers: int,
    max_len: int,
    max_res: int,
    answer_key: str = None,
    split: str = "train",
    limit_samples: int = None,
) -> DataLoader:
    dataset = TextImagePairStream(
        dataset_name=dataset_name,
        split=split,
        max_size=max_res,
        answer_key=answer_key,
        max_samples=limit_samples,
    )

    def collate(batch):
        images = torch.stack([b["image"] for b in batch])
        texts = [b["text"] for b in batch]
        enc = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        return {"images": images, "input_ids": enc["input_ids"]}

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=True,
        collate_fn=collate,
    )


def freeze_module(module):
    for p in module.parameters():
        p.requires_grad = False


def unfreeze_module(module):
    for p in module.parameters():
        p.requires_grad = True


def load_checkpoint_if_available(model: HunyuanLeJEPA, path: str):
    if not path:
        return
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    model_state = model.state_dict()
    filtered = {}
    skipped = []
    for k, v in state.items():
        if k.startswith("decoder."):
            continue  # decoder shapes differ between phases; skip
        if k not in model_state:
            continue
        if model_state[k].shape != v.shape:
            skipped.append((k, v.shape, model_state[k].shape))
            continue
        filtered[k] = v
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print(
        f"Loaded checkpoint with {len(filtered)} tensors. "
        f"Missing={len(missing)}, unexpected={len(unexpected)}, skipped_shape_mismatch={len(skipped)}"
    )


def run_stage(
    accelerator: Accelerator,
    model: HunyuanLeJEPA,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    steps: int,
    pad_token_id: int,
    stage_name: str,
    grad_accum: int,
    log_every: int,
):
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    model.train()
    optimizer.zero_grad()
    step_in_stage = 0
    for batch in dataloader:
        step_in_stage += 1
        images = batch["images"]
        input_ids = batch["input_ids"]
        tgt = input_ids[:, 1:].contiguous()
        inp = input_ids[:, :-1].contiguous()
        mask = causal_mask(inp.size(1), device=inp.device)
        logits = model.forward_decoder(images, inp, tgt_mask=mask)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt.reshape(-1),
            ignore_index=pad_token_id,
        )
        loss = loss / grad_accum
        accelerator.backward(loss)

        if step_in_stage % grad_accum == 0:
            optimizer.step()
            optimizer.zero_grad()

        if accelerator.is_local_main_process and step_in_stage % log_every == 0:
            accelerator.print(f"{stage_name} step {step_in_stage}: loss={loss.item() * grad_accum:.4f}")

        if step_in_stage >= steps:
            break

    accelerator.wait_for_everyone()
    return accelerator.unwrap_model(model)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="HF dataset with image/text fields.")
    parser.add_argument("--split", type=str, default="train", help="HF split, e.g., train[:2048] to test quickly.")
    parser.add_argument("--limit-samples", type=int, default=None, help="Optional cap on samples per epoch.")
    parser.add_argument("--checkpoint", type=str, default="", help="Phase 2 checkpoint path.")
    parser.add_argument("--backbone", type=str, default="vit_large_patch16_siglip_384")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--max-res", type=int, default=1024)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lr-probe", type=float, default=1e-3, help="Stage A adapter lr.")
    parser.add_argument("--lr-handshake", type=float, default=1e-5, help="Stage B adapter/decoder lr.")
    parser.add_argument("--lr-vit", type=float, default=1e-6, help="Stage B ViT lr.")
    parser.add_argument(
        "--skip-stage-a",
        action="store_true",
        help="Skip Stage A (probe) and start directly with Stage B (handshake).",
    )
    parser.add_argument("--stage-a-steps", type=int, default=1000)
    parser.add_argument("--stage-b-steps", type=int, default=2000)
    parser.add_argument("--tokenizer", type=str, default="tencent/HunyuanOCR")
    parser.add_argument("--save-dir", type=str, default="checkpoints/phase3")
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()

    accelerator = Accelerator()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    decoder_cfg = DecoderConfig(vocab_size=len(tokenizer), max_len=args.max_len)
    model = HunyuanLeJEPA(
        vision_cfg=VisionConfig(backbone=args.backbone),
        projector_cfg=ProjectorConfig(),
        adapter_cfg=AdapterConfig(adapter_dim=decoder_cfg.hidden_size),
        decoder_cfg=decoder_cfg,
        with_projector=True,
    )
    load_checkpoint_if_available(model, args.checkpoint)
    model.drop_projector()  # Discard LeJEPA projector for alignment.

    # Stage A: freeze ViT + decoder, train adapter only.
    lower_name = args.dataset.lower()
    if args.dataset in {"ocr_vqa", "docvqa"} or "ocr-vqa" in lower_name:
        answer_key = "answers"
    else:
        answer_key = None
    if args.skip_stage_a:
        accelerator.print("Skipping Stage A: starting Stage B directly.")
    else:
        freeze_module(model.vision)
        freeze_module(model.decoder)
        unfreeze_module(model.adapter)

        dataloader_a = build_dataloader(
            dataset_name=args.dataset,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            workers=args.num_workers,
            max_len=args.max_len,
            max_res=args.max_res,
            answer_key=answer_key,
            split=args.split,
            limit_samples=args.limit_samples,
        )
        optimizer_a = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr_probe,
            weight_decay=args.weight_decay,
        )

        model = run_stage(
            accelerator,
            model,
            dataloader_a,
            optimizer_a,
            steps=args.stage_a_steps,
            pad_token_id=tokenizer.pad_token_id,
            stage_name="StageA-Probe",
            grad_accum=args.grad_accum,
            log_every=args.log_every,
        )

    # Stage B: unfreeze ViT; train adapter + decoder + ViT with differential LRs.
    unfreeze_module(model.vision)
    unfreeze_module(model.decoder)
    unfreeze_module(model.adapter)

    dataloader_b = build_dataloader(
        dataset_name=args.dataset,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        workers=args.num_workers,
        max_len=args.max_len,
        max_res=args.max_res,
        answer_key=answer_key,
        split=args.split,
        limit_samples=args.limit_samples,
    )

    optimizer_b = torch.optim.AdamW(
        [
            {"params": model.vision.parameters(), "lr": args.lr_vit},
            {
                "params": list(model.adapter.parameters()) + list(model.decoder.parameters()),
                "lr": args.lr_handshake,
            },
        ],
        weight_decay=args.weight_decay,
    )

    model = run_stage(
        accelerator,
        model,
        dataloader_b,
        optimizer_b,
        steps=args.stage_b_steps,
        pad_token_id=tokenizer.pad_token_id,
        stage_name="StageB-Handshake",
        grad_accum=args.grad_accum,
        log_every=args.log_every,
    )

    if accelerator.is_local_main_process:
        save_path = Path(args.save_dir) / "phase3_final.pt"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict()}, save_path)
        accelerator.print(f"Saved Phase 3 checkpoint to {save_path}")


if __name__ == "__main__":
    main()
