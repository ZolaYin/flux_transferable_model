"""
FluxHierarchicalMoEModel
========================
Local (HLS patch) + Global (30-day temporal sequence) fusion for carbon flux regression.

Architecture:
  - tabular_encoder : MLP on daily MODIS+ERA5 features
  - metadata_encoder: MLP on IGBP+Köppen one-hot
  - sequence_encoder: Mamba/GRU on 30-day daily sequence
  - image_encoder   : ResNet18 -> layer2/3/4 feature maps,
                      FiLM-conditioned by temporal vector,
                      multi-scale pooled and concatenated
  - moe_head        : shared expert + N routed experts (top-k),
                      outputs GPP scalar

Forward interface identical to FluxTransferModel:
  train → (predictions, aux_loss)
  eval  → predictions
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torchvision.models as tv_models
    _TV_OK = True
except Exception:
    tv_models = None
    _TV_OK = False

from .flux_transfer_model import TemporalMambaEncoder

logger = logging.getLogger(__name__)


# ── shared helper ──────────────────────────────────────────────────────────────

def _make_mlp(input_dim: int, hidden_dims: List[int], dropout: float) -> nn.Sequential:
    layers: list = []
    cur = input_dim
    for h in hidden_dims:
        layers += [nn.Linear(cur, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
        cur = h
    return nn.Sequential(*layers)


# ── multi-scale ResNet18 with FiLM conditioning ────────────────────────────────

class MultiScaleResNetWithFiLM(nn.Module):
    """
    ResNet18 backbone that extracts feature maps from layer2, layer3, layer4,
    projects each to `adapter_dim` channels, modulates them with FiLM gating
    from a temporal context vector, then global-average-pools and concatenates.

    Output dim: 3 * adapter_dim
    """

    _LAYER_DIMS_R18 = {"layer2": 128, "layer3": 256, "layer4": 512}

    def __init__(
        self,
        in_channels: int,
        pretrained: bool,
        adapter_dim: int,
        temporal_dim: int,
        dropout: float,
    ):
        super().__init__()
        assert _TV_OK and tv_models is not None, "torchvision is required"

        weights = tv_models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = tv_models.resnet18(weights=weights)

        # Adapt first conv to accept in_channels spectral bands
        old = backbone.conv1
        new_conv = nn.Conv2d(
            in_channels, old.out_channels,
            kernel_size=old.kernel_size, stride=old.stride,
            padding=old.padding, bias=False,
        )
        with torch.no_grad():
            nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
            if pretrained and in_channels > 0:
                # Average RGB weights, tile them across spectral bands, and preserve
                # the approximate activation scale of the original 3-channel conv.
                rgb_mean = old.weight.mean(dim=1, keepdim=True)  # (64,1,7,7)
                scale = old.in_channels / float(in_channels)
                new_conv.weight.copy_(rgb_mean.expand_as(new_conv.weight) * scale)
        backbone.conv1 = new_conv
        backbone.fc = nn.Identity()
        self.backbone = backbone

        # Forward hooks to capture intermediate feature maps
        self._cache: Dict[str, torch.Tensor] = {}
        self._scales = ["layer2", "layer3", "layer4"]
        for name in self._scales:
            getattr(self.backbone, name).register_forward_hook(self._hook(name))

        # 1×1 conv adapters: channel-normalise to adapter_dim
        self.adapters = nn.ModuleDict({
            name: nn.Sequential(
                nn.Conv2d(self._LAYER_DIMS_R18[name], adapter_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(adapter_dim),
                nn.ReLU(inplace=True),
            )
            for name in self._scales
        })

        # FiLM projectors: temporal_vec → (gamma, beta) for each scale
        self.film_projs = nn.ModuleDict({
            name: nn.Linear(temporal_dim, 2 * adapter_dim)
            for name in self._scales
        })

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.out_dim = adapter_dim * len(self._scales)  # 3 * adapter_dim

    def _hook(self, name: str):
        def _fn(_, __, output):
            self._cache[name] = output
        return _fn

    def forward(self, patch: torch.Tensor, temporal_vec: torch.Tensor) -> torch.Tensor:
        """
        patch       : (B, C_in, H, W)
        temporal_vec: (B, temporal_dim)
        returns     : (B, 3 * adapter_dim)
        """
        self._cache.clear()
        self.backbone(patch)  # populates cache via hooks

        vecs = []
        for name in self._scales:
            feat = self._cache[name]                      # (B, C, H, W)
            feat = self.adapters[name](feat)              # (B, D, H, W)

            # FiLM: temporal context modulates spatial features
            film = self.film_projs[name](temporal_vec)    # (B, 2D)
            gamma, beta = film.chunk(2, dim=-1)           # each (B, D)
            feat = feat * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]

            vecs.append(self.pool(feat).flatten(1))       # (B, D)

        return self.drop(torch.cat(vecs, dim=-1))         # (B, 3D)


# ── MoE regression head ────────────────────────────────────────────────────────

class FluxMoEHead(nn.Module):
    """
    1 shared expert + n_experts routed experts, top-k routing.
    Input : (B, input_dim)
    Output: (B, output_dim), scalar aux_loss
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int = 1,
        n_experts: int = 4,
        top_k: int = 2,
        aux_loss_alpha: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = min(top_k, n_experts)
        self.alpha = aux_loss_alpha

        def _expert() -> nn.Module:
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, input_dim),
            )

        self.shared_expert  = _expert()
        self.routed_experts = nn.ModuleList([_expert() for _ in range(n_experts)])
        self.router         = nn.Linear(input_dim, n_experts, bias=False)
        self.output_proj    = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        shared = self.shared_expert(x)                              # (B, D)

        scores = self.router(x).softmax(dim=-1)                     # (B, E)
        topk_s, topk_i = scores.topk(self.top_k, dim=-1)           # (B, k)
        topk_s = topk_s / topk_s.sum(-1, keepdim=True).clamp(1e-6) # renorm

        routed = torch.zeros_like(x)
        for j in range(self.n_experts):
            # weight for expert j across the batch
            w = (topk_s * (topk_i == j).float()).sum(dim=-1)        # (B,)
            if w.any():
                routed = routed + self.routed_experts[j](x) * w.unsqueeze(-1)

        out = self.output_proj(shared + routed)                     # (B, out_dim)

        # Load-balancing auxiliary loss
        aux = x.new_zeros(())
        if self.training and self.alpha > 0:
            f = F.one_hot(topk_i, self.n_experts).float().sum(1) / self.top_k  # (B,E)
            aux = (f.mean(0) * scores.mean(0)).sum() * self.n_experts * self.alpha

        return out, aux


# ── main model ─────────────────────────────────────────────────────────────────

class FluxHierarchicalMoEModel(nn.Module):
    """
    Lightweight multi-scale temporal FiLM fusion of spatial (HLS patch) and
    temporal (dense 30-day sequence) features with a MoE regression head.
    """

    def __init__(
        self,
        # tabular / metadata
        tabular_input_dim: int,
        metadata_input_dim: int,
        tabular_hidden_dims: Optional[List[int]] = None,
        metadata_hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        # temporal branch
        use_sequence_branch: bool = True,
        sequence_input_dim: int = 0,
        sequence_encoder_type: str = "mamba",
        sequence_embedding_dim: int = 128,
        mamba_d_model: int = 128,
        mamba_layers: int = 2,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        transformer_nhead: int = 4,
        transformer_dim_feedforward: Optional[int] = None,
        # image branch
        use_image_branch: bool = True,
        image_channels: int = 6,
        image_resnet_pretrained: bool = False,
        adapter_dim: int = 128,
        # MoE head
        moe_hidden_dim: int = 256,
        moe_n_experts: int = 4,
        moe_top_k: int = 2,
        moe_aux_loss_alpha: float = 1.0,
        use_moe_head: bool = True,
        output_dim: int = 1,
        **kwargs,
    ):
        super().__init__()
        self.use_sequence_branch = use_sequence_branch
        self.use_image_branch    = use_image_branch
        self.use_moe_head = use_moe_head

        tabular_hidden_dims  = tabular_hidden_dims  or [256, 128]
        metadata_hidden_dims = metadata_hidden_dims or [32]

        self.tabular_encoder  = _make_mlp(tabular_input_dim,  tabular_hidden_dims,  dropout)
        self.metadata_encoder = _make_mlp(metadata_input_dim, metadata_hidden_dims, dropout)
        tab_out  = tabular_hidden_dims[-1]
        meta_out = metadata_hidden_dims[-1]

        # temporal encoder
        seq_out = 0
        self.sequence_encoder = None
        if use_sequence_branch:
            if sequence_input_dim <= 0:
                raise ValueError("sequence_input_dim must be >0 when use_sequence_branch=True")
            self.sequence_encoder = TemporalMambaEncoder(
                input_dim=sequence_input_dim,
                d_model=mamba_d_model,
                out_dim=sequence_embedding_dim,
                num_layers=mamba_layers,
                d_state=mamba_d_state,
                d_conv=mamba_d_conv,
                expand=mamba_expand,
                dropout=dropout,
                encoder_type=sequence_encoder_type,
                transformer_nhead=transformer_nhead,
                transformer_dim_feedforward=transformer_dim_feedforward,
            )
            seq_out = sequence_embedding_dim

        # image encoder with FiLM from temporal (fallback: tabular)
        img_out = 0
        self.image_encoder = None
        if use_image_branch:
            film_dim = seq_out if seq_out > 0 else tab_out
            self.image_encoder = MultiScaleResNetWithFiLM(
                in_channels=image_channels,
                pretrained=image_resnet_pretrained,
                adapter_dim=adapter_dim,
                temporal_dim=film_dim,
                dropout=dropout,
            )
            img_out = self.image_encoder.out_dim  # 3 * adapter_dim

        fusion_dim = tab_out + meta_out + seq_out + img_out
        if use_moe_head:
            self.moe_head = FluxMoEHead(
                input_dim=fusion_dim,
                hidden_dim=moe_hidden_dim,
                output_dim=output_dim,
                n_experts=moe_n_experts,
                top_k=moe_top_k,
                aux_loss_alpha=moe_aux_loss_alpha,
                dropout=dropout,
            )
            self.fusion_head = None
            self.regression_head = None
        else:
            self.moe_head = None
            self.fusion_head = _make_mlp(fusion_dim, [moe_hidden_dim], dropout)
            self.regression_head = nn.Linear(moe_hidden_dim, output_dim)

        logger.info(
            "FluxHierarchicalMoEModel: "
            "tab=%d meta=%d seq=%d img=%d -> fusion=%d | moe=%s experts=%d top_k=%d",
            tab_out, meta_out, seq_out, img_out, fusion_dim, use_moe_head, moe_n_experts, moe_top_k,
        )

    # ------------------------------------------------------------------

    def print_trainable_parameters_summary(self):
        parts = {
            "Tabular Encoder" : self.tabular_encoder,
            "Metadata Encoder": self.metadata_encoder,
            "Sequence Encoder": self.sequence_encoder,
            "Image Encoder"   : self.image_encoder,
            "MoE Head"        : self.moe_head,
            "Fusion Head"     : self.fusion_head,
            "Regression Head" : self.regression_head,
        }
        total = trainable = 0
        logger.info("-" * 60 + "\nFluxHierarchicalMoEModel parameter summary:")
        for name, part in parts.items():
            if part is None:
                continue
            pt = sum(p.numel() for p in part.parameters())
            tr = sum(p.numel() for p in part.parameters() if p.requires_grad)
            total += pt; trainable += tr
            logger.info("  %-25s Total=%.2fM  Trainable=%.2fM", name, pt/1e6, tr/1e6)
        logger.info("  %-25s Total=%.2fM  Trainable=%.2fM", "TOTAL", total/1e6, trainable/1e6)
        logger.info("-" * 60)

    # ------------------------------------------------------------------

    def forward(
        self, batch: Dict[str, Any]
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        tab_enc  = self.tabular_encoder(batch["tabular_features"])
        meta_enc = self.metadata_encoder(batch["metadata_features"])
        parts = [tab_enc, meta_enc]

        # temporal branch
        seq_enc = None
        if self.use_sequence_branch and self.sequence_encoder is not None:
            seq_feats = batch.get("sequence_features")
            if seq_feats is None:
                raise ValueError("sequence_features missing from batch")
            seq_enc = self.sequence_encoder(
                torch.nan_to_num(seq_feats),
                batch.get("sequence_mask"),
            )
            parts.append(seq_enc)

        # image branch (FiLM vector: temporal if available, else tabular)
        if self.use_image_branch and self.image_encoder is not None:
            patch = batch.get("patch_tensor")
            if patch is None:
                raise ValueError("patch_tensor missing from batch")
            film_vec = seq_enc if seq_enc is not None else tab_enc
            patch = torch.nan_to_num(patch)
            if patch.ndim == 3:
                patch = patch.unsqueeze(0)
            if patch.ndim == 5:
                batch_size, num_patches, channels, height, width = patch.shape
                flat_patch = patch.reshape(batch_size * num_patches, channels, height, width)
                flat_film = film_vec.unsqueeze(1).expand(batch_size, num_patches, -1).reshape(
                    batch_size * num_patches, -1
                )
                flat_img = self.image_encoder(flat_patch, flat_film)
                patch_img = flat_img.reshape(batch_size, num_patches, -1)
                patch_mask = batch.get("patch_mask")
                if patch_mask is None:
                    patch_mask = patch.new_ones((batch_size, num_patches))
                patch_mask = patch_mask.to(device=patch_img.device, dtype=patch_img.dtype)
                denom = patch_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
                img_enc = (patch_img * patch_mask.unsqueeze(-1)).sum(dim=1) / denom
            else:
                img_enc = self.image_encoder(patch, film_vec)
            parts.append(img_enc)

        fused = torch.cat(parts, dim=-1)
        if self.use_moe_head and self.moe_head is not None:
            pred, aux = self.moe_head(fused)
        else:
            pred = self.regression_head(self.fusion_head(fused))
            aux = pred.new_zeros(())
        pred = pred.squeeze(-1)

        if self.training:
            return pred, aux
        return pred
