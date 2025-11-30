"""
Data pipeline utilities for Hunyuan-LeJEPA.

- Streaming mixture from Hugging Face and WebDataset shards:
  * chainyo/rvl-cdip (50%)
  * pixparse/pdfa-eng-wds (30%) via WebDataset URLs
  * naver-clova-ix/cord-v2 (20%)
- Preprocessing: resize to max 1024x1024 while preserving aspect ratio, then pad to
  square.
- View generation: block-wise masking (30-50%) for the context view instead of random
  cropping.
"""

import math
import os
import random
from io import BytesIO
from typing import Dict, Iterator, Tuple

import datasets
import torch
from pathlib import Path

from PIL import Image
from torch.utils.data import IterableDataset
from torchvision.transforms import functional as TF
import webdataset as wds


def resize_and_pad(img: Image.Image, max_size: int = 1024, pad_value: int = 128) -> Image.Image:
    """Resizes preserving aspect ratio and pads to a square canvas."""
    w, h = img.size
    scale = min(max_size / max(w, h), 1.0)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    img = img.resize((new_w, new_h), resample=Image.BILINEAR)

    pad_w, pad_h = max_size - new_w, max_size - new_h
    left, right = pad_w // 2, pad_w - pad_w // 2
    top, bottom = pad_h // 2, pad_h - pad_h // 2

    canvas = Image.new("RGB", (max_size, max_size), color=(pad_value,) * 3)
    canvas.paste(img, (left, top))
    return canvas


def blockwise_mask(img_tensor: torch.Tensor, ratio_range: Tuple[float, float] = (0.3, 0.5)) -> torch.Tensor:
    """Masks a contiguous rectangular block covering 30-50% of the area."""
    c, h, w = img_tensor.shape
    ratio = random.uniform(*ratio_range)
    block_area = ratio * h * w
    block_h = max(1, int(round(math.sqrt(block_area * h / w))))
    block_w = max(1, int(round(block_area / max(block_h, 1))))
    block_h = min(block_h, h)
    block_w = min(block_w, w)

    y0 = random.randint(0, h - block_h)
    x0 = random.randint(0, w - block_w)
    masked = img_tensor.clone()
    fill = img_tensor.mean(dim=(1, 2), keepdim=True)
    masked[:, y0 : y0 + block_h, x0 : x0 + block_w] = fill
    return masked


class HybridDocumentStream(IterableDataset):
    """
    Streams documents from HF with the mandated 50/30/20 mixture and produces
    (global, context) view pairs for LeJEPA.
    """

    def __init__(
        self,
        split: str = "train",
        max_size: int = 1024,
        mask_ratio: Tuple[float, float] = (0.3, 0.5),
        seed: int = 0,
        pdfa_local_dir: str = None,
    ):
        super().__init__()
        self.max_size = max_size
        self.mask_ratio = mask_ratio
        self.rng = random.Random(seed)
        self.pdfa_local_dir = pdfa_local_dir or os.environ.get("PDFA_WDS_DIR")

        allow_stream_pdfa = os.environ.get("ALLOW_PDFA_STREAM", "0") == "1"
        include_pdfa = self.pdfa_local_dir is not None or allow_stream_pdfa

        base_sources = [
            ("rvl_cdip", 0.5, "hf", "chainyo/rvl-cdip"),
            ("cord", 0.2, "hf", "naver-clova-ix/cord-v2"),
        ]
        if include_pdfa:
            base_sources.insert(1, ("pdfa", 0.3, "wds", None))

        total_weight = sum(w for _, w, _, _ in base_sources)
        self.sources = [
            (name, weight / total_weight, kind, hf_name)
            for name, weight, kind, hf_name in base_sources
        ]
        self.source_names = [name for name, _, _, _ in self.sources]
        self.datasets = {}
        self.iters = {}
        for name, _, kind, hf_name in self.sources:
            if kind == "hf":
                ds = datasets.load_dataset(hf_name, split=split, streaming=True)
                self.datasets[name] = ds
                self.iters[name] = iter(ds.shuffle(buffer_size=2048, seed=seed))
            elif kind == "wds":
                # Prefer local tar shards if provided to avoid network flakiness.
                urls = None
                if self.pdfa_local_dir:
                    local_dir = Path(self.pdfa_local_dir)
                    urls = sorted(str(p) for p in local_dir.glob("*.tar"))
                    if not urls:
                        print(f"[HybridDocumentStream] No local PDFa tar shards found in {local_dir}")

                if not urls:
                urls = (
                    "https://huggingface.co/datasets/pixparse/pdfa-eng-wds/resolve/main/"
                    "pdfa-eng-train-{000000..000099}.tar"
                )

                ds = (
                    wds.WebDataset(
                        urls,
                        resampled=False,
                        handler=wds.handlers.warn_and_continue,
                        shardshuffle=1000,
                    )
                    .shuffle(1000)
                    .decode("pil")
                )
                self.datasets[name] = ds
                self.iters[name] = iter(ds)
        self.weights = [w for _, w, _, _ in self.sources]

    def _get_image(self, sample: Dict) -> Image.Image:
        for key in ("image", "img", "jpg", "jpeg", "png"):
            if key in sample:
                img = sample[key]
                break
        else:
            raise KeyError("No image key found in sample.")

        if isinstance(img, Image.Image):
            return img.convert("RGB")
        if isinstance(img, bytes):
            return Image.open(BytesIO(img)).convert("RGB")
        if hasattr(img, "convert"):
            return img.convert("RGB")
        # Fallback: if path is provided
        return Image.open(img).convert("RGB")

    def _next_sample(self, name: str) -> Dict:
        try:
            return next(self.iters[name])
        except StopIteration:
            # HF streams: reshuffle; WDS (resampled) should not exhaust but handle anyway.
            seed = self.rng.randint(0, 1_000_000)
            if isinstance(self.datasets[name], datasets.IterableDataset):
                self.iters[name] = iter(self.datasets[name].shuffle(buffer_size=2048, seed=seed))
            else:
                self.iters[name] = iter(self.datasets[name])
            return next(self.iters[name])

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        while True:
            name = self.rng.choices(self.source_names, weights=self.weights, k=1)[0]
            sample = self._next_sample(name)
            img = self._get_image(sample)
            img = resize_and_pad(img, max_size=self.max_size)
            img_tensor = TF.to_tensor(img)  # (C, H, W)
            context = blockwise_mask(img_tensor, self.mask_ratio)
            yield {
                "global": img_tensor,
                "context": context,
                "source": name,
            }


class TextImagePairStream(IterableDataset):
    """
    Streams text-image pairs for Phase 3 alignment. Assumes dataset exposes image and
    text fields (configurable).
    """

    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        text_key: str = "text",
        image_key: str = "image",
        max_size: int = 1024,
        seed: int = 0,
    ):
        super().__init__()
        self.text_key = text_key
        self.image_key = image_key
        self.max_size = max_size
        self.ds = datasets.load_dataset(dataset_name, split=split, streaming=True)
        self.iter = iter(self.ds.shuffle(buffer_size=2048, seed=seed))

    def _get_image(self, sample: Dict) -> Image.Image:
        img = sample[self.image_key]
        if isinstance(img, Image.Image):
            return img.convert("RGB")
        if isinstance(img, bytes):
            return Image.open(BytesIO(img)).convert("RGB")
        if hasattr(img, "convert"):
            return img.convert("RGB")
        return Image.open(img).convert("RGB")

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        while True:
            try:
                sample = next(self.iter)
            except StopIteration:
                self.iter = iter(self.ds.shuffle(buffer_size=2048))
                sample = next(self.iter)
            text = sample[self.text_key]
            img = self._get_image(sample)
            img = resize_and_pad(img, max_size=self.max_size)
            img_tensor = TF.to_tensor(img)
            yield {"image": img_tensor, "text": text}
