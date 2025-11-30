"""
Phase 2: Unsupervised LeJEPA pre-training.

Key points:
- Uses block-masked context view (no random cropping).
- Loss = (1 - lambda) * MSE + lambda * SIGReg (Epps-Pulley, 1024 slices, [-5, 5]).
- Logs loss_sigreg separately for collapse detection.
"""

import argparse
from pathlib import Path
from typing import Dict

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import HybridDocumentStream
from src.losses import lejepa_loss
from src.model import (
    AdapterConfig,
    DecoderConfig,
    HunyuanLeJEPA,
    ProjectorConfig,
    VisionConfig,
)


def build_dataloader(batch_size: int, workers: int, max_size: int) -> DataLoader:
    dataset = HybridDocumentStream(max_size=max_size)

    def collate(batch):
        global_view = torch.stack([b["global"] for b in batch])
        context_view = torch.stack([b["context"] for b in batch])
        return {"global": global_view, "context": context_view}

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=True,
        collate_fn=collate,
    )


def save_checkpoint(model: HunyuanLeJEPA, optimizer: torch.optim.Optimizer, step: int, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--lambda-sig", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=16, help="Per-device batch size.")
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1_000_000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-dir", type=str, default="checkpoints/phase2")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-res", type=int, default=1024)
    parser.add_argument("--backbone", type=str, default="vit_large_patch16_siglip_384")
    args = parser.parse_args()

    accelerator = Accelerator()
    vision_cfg = VisionConfig(backbone=args.backbone)
    projector_cfg = ProjectorConfig()
    adapter_cfg = AdapterConfig()
    decoder_cfg = DecoderConfig()

    model = HunyuanLeJEPA(
        vision_cfg=vision_cfg,
        projector_cfg=projector_cfg,
        adapter_cfg=adapter_cfg,
        decoder_cfg=decoder_cfg,
        with_projector=True,
    )
    # Only projector + backbone used in Phase 2; freeze adapter/decoder to save memory.
    for p in model.adapter.parameters():
        p.requires_grad = False
    for p in model.decoder.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    dataloader = build_dataloader(args.batch_size, args.num_workers, args.max_res)
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    model.train()

    progress = tqdm(
        total=args.max_steps,
        disable=not accelerator.is_local_main_process,
        desc="Phase2",
    )
    optimizer.zero_grad()

    sigreg_kwargs: Dict = {"num_slices": 1024, "t_range": (-5.0, 5.0), "num_t": 17}

    for step, batch in enumerate(dataloader, start=1):
        global_imgs = batch["global"]
        context_imgs = batch["context"]

        proj_global, _ = model.forward_lejepa(global_imgs)
        proj_context, _ = model.forward_lejepa(context_imgs)

        emb_global = proj_global.mean(dim=1)
        emb_context = proj_context.mean(dim=1)

        global_embeds = emb_global.unsqueeze(1)  # (B, 1, D)
        all_embeds = torch.stack([emb_global, emb_context], dim=1)

        loss, metrics = lejepa_loss(
            global_embeds=global_embeds,
            all_view_embeds=all_embeds,
            lambd=args.lambda_sig,
            global_step=step,
            sigreg_kwargs=sigreg_kwargs,
        )

        loss = loss / args.grad_accum
        accelerator.backward(loss)

        if step % args.grad_accum == 0:
            optimizer.step()
            optimizer.zero_grad()

        if accelerator.is_local_main_process and step % args.log_every == 0:
            accelerator.print(
                f"step {step} | loss={metrics['loss_total'].item():.4f} "
                f"pred={metrics['loss_pred'].item():.4f} "
                f"sigreg={metrics['loss_sigreg'].item():.4f}"
            )

        if accelerator.is_local_main_process and step % (args.log_every * 10) == 0:
            ckpt_path = Path(args.save_dir) / f"step_{step}.pt"
            save_checkpoint(model, optimizer, step, ckpt_path)

        progress.update(1)
        if step >= args.max_steps:
            break

    progress.close()


if __name__ == "__main__":
    main()
