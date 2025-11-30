import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


@dataclass
class VisionConfig:
    """Configuration for the adaptive SigLIP-style ViT backbone."""

    backbone: str = "vit_large_patch14_siglip_384"
    pretrained: bool = False
    img_size: Optional[int] = None  # None enables fully dynamic shapes
    drop_rate: float = 0.1
    drop_path_rate: float = 0.1


@dataclass
class ProjectorConfig:
    """Configuration for the LeJEPA projector (Phase 2 only)."""

    hidden_dim: int = 4096
    output_dim: int = 8192


@dataclass
class AdapterConfig:
    """Configuration for the adaptive MLP adapter that bridges ViT and decoder."""

    adapter_dim: int = 1024
    num_adapter_tokens: int = 32
    dropout: float = 0.1


@dataclass
class DecoderConfig:
    """Configuration for the OCR Transformer decoder (Hunyuan-0.5B style lite stub)."""

    vocab_size: int = 32000
    hidden_size: int = 1024
    num_layers: int = 12
    num_heads: int = 16
    dropout: float = 0.1
    max_len: int = 4096


class VisionBackbone(nn.Module):
    """
    SigLIP-style ViT wrapper with adaptive patching (accepts arbitrary H, W without
    forcing a fixed square resize). Relies on timm's dynamic_img_size support.
    """

    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.model = timm.create_model(
            cfg.backbone,
            pretrained=cfg.pretrained,
            num_classes=0,
            global_pool="",
            img_size=cfg.img_size,
            drop_rate=cfg.drop_rate,
            drop_path_rate=cfg.drop_path_rate,
            dynamic_img_size=True,
        )
        # Many SigLIP variants expose num_features and patch_embed attributes.
        self.embed_dim = getattr(self.model, "num_features", None) or getattr(
            self.model, "embed_dim", None
        )
        if self.embed_dim is None:
            raise ValueError("Backbone does not expose embed_dim/num_features.")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Returns patch tokens (no cls token) and the patch grid size (H_p, W_p).
        """
        features = self.model.forward_features(x)
        if isinstance(features, dict):
            tokens = (
                features.get("x_norm_patchtokens")
                or features.get("x_norm")
                or features.get("x")
            )
        else:
            tokens = features

        if tokens.dim() != 3:
            raise RuntimeError(f"Unexpected feature shape: {tokens.shape}")

        # Drop class token if present.
        if tokens.shape[1] == getattr(self.model, "num_tokens", 0):
            tokens = tokens[:, 1:]
        elif tokens.shape[1] == getattr(self.model.patch_embed, "num_patches", 0) + 1:
            tokens = tokens[:, 1:]

        # Derive patch grid using the dynamic shape from patch_embed.
        grid = getattr(self.model.patch_embed, "grid_size", None)
        if grid is None:
            # Fallback: infer grid from spatial dims assuming square patching.
            num_patches = tokens.shape[1]
            side = int(math.sqrt(num_patches))
            grid = (side, num_patches // max(side, 1))

        return tokens, grid


class LeJEPAProjector(nn.Module):
    """Three-layer MLP projector used only during Phase 2 (LeJEPA pre-training)."""

    def __init__(self, in_dim: int, cfg: ProjectorConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AdaptiveMLPAdapter(nn.Module):
    """
    Token-compressing MLP adapter. Learns a small set of queries that pool the ViT
    patch grid, then projects to decoder hidden size.
    """

    def __init__(self, in_dim: int, cfg: AdapterConfig, out_dim: Optional[int] = None):
        super().__init__()
        out_dim = out_dim or cfg.adapter_dim
        self.num_tokens = cfg.num_adapter_tokens
        self.query_pool = nn.Parameter(torch.randn(self.num_tokens, in_dim))
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, cfg.adapter_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.adapter_dim, out_dim),
        )
        nn.init.normal_(self.query_pool, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C)
        attn = torch.einsum("q c, b n c -> b q n", self.query_pool, x)
        attn = attn / math.sqrt(x.size(-1))
        weights = attn.softmax(dim=-1)
        pooled = torch.einsum("b q n, b n c -> b q c", weights, x)
        return self.proj(pooled)


class TransformerOCRDecoder(nn.Module):
    """
    Lightweight Transformer decoder that cross-attends over adapter outputs. This is
    a stand-in for the larger Hunyuan-0.5B language model.
    """

    def __init__(self, cfg: DecoderConfig, adapter_dim: int):
        super().__init__()
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(cfg.max_len, cfg.hidden_size))
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=cfg.hidden_size,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.hidden_size * 4,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=cfg.num_layers)
        self.adapter_proj = nn.Linear(adapter_dim, cfg.hidden_size)
        self.output_head = nn.Linear(cfg.hidden_size, cfg.vocab_size)
        self.dropout = nn.Dropout(cfg.dropout)
        nn.init.normal_(self.pos_embed, std=0.01)

    def forward(
        self,
        tokens: torch.Tensor,
        adapter_feats: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        tokens: (B, L) token ids
        adapter_feats: (B, T, C_a) adapter outputs
        tgt_mask: optional causal mask
        """
        bsz, seqlen = tokens.shape
        pos = self.pos_embed[:seqlen].unsqueeze(0)
        x = self.token_embed(tokens) + pos
        x = self.dropout(x)
        memory = self.adapter_proj(adapter_feats)
        decoded = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=tgt_mask,
        )
        return self.output_head(decoded)


class HunyuanLeJEPA(nn.Module):
    """
    Full hybrid architecture that supports both Phase 2 (LeJEPA) and Phase 3 (adapter
    alignment + OCR decoding).
    """

    def __init__(
        self,
        vision_cfg: VisionConfig,
        projector_cfg: ProjectorConfig,
        adapter_cfg: AdapterConfig,
        decoder_cfg: DecoderConfig,
        with_projector: bool = True,
    ):
        super().__init__()
        self.vision = VisionBackbone(vision_cfg)
        self.projector = LeJEPAProjector(self.vision.embed_dim, projector_cfg) if with_projector else None
        self.adapter = AdaptiveMLPAdapter(
            in_dim=self.vision.embed_dim,
            cfg=adapter_cfg,
            out_dim=decoder_cfg.hidden_size,
        )
        self.decoder = TransformerOCRDecoder(decoder_cfg, adapter_dim=decoder_cfg.hidden_size)

    def encode(self, images: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Returns raw ViT patch tokens and patch grid."""
        return self.vision(images)

    def encode_project(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Used in Phase 2: returns backbone tokens and projector outputs."""
        tokens, grid = self.encode(images)
        if self.projector is None:
            raise RuntimeError("Projector is disabled for this model instance.")
        projected = self.projector(tokens)
        return projected, tokens

    def forward_lejepa(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Phase 2 forward: returns projector outputs and backbone tokens for loss
        computation.
        """
        projected, tokens = self.encode_project(images)
        return projected, tokens

    def forward_decoder(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Phase 3 forward: adapter compression + Transformer decoder.
        """
        tokens, _ = self.encode(images)
        adapter_feats = self.adapter(tokens)
        logits = self.decoder(input_ids, adapter_feats, tgt_mask=tgt_mask)
        return logits

    def drop_projector(self):
        """Utility for Phase 3: remove projector to save memory."""
        self.projector = None
