import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

try:
    import torchvision.models as tv_models
    _TORCHVISION_IMPORTED = True
    _TORCHVISION_IMPORT_ERROR = None
except Exception as exc:
    tv_models = None
    _TORCHVISION_IMPORTED = False
    _TORCHVISION_IMPORT_ERROR = exc


logger = logging.getLogger(__name__)

TemporalMambaCore = None
_TEMPORAL_MAMBA_IMPORTED = False
_TEMPORAL_MAMBA_IMPORT_ERROR = None
_TEMPORAL_MAMBA_RESOLVED = False


def _resolve_temporal_mamba_backend():
    global TemporalMambaCore
    global _TEMPORAL_MAMBA_IMPORTED
    global _TEMPORAL_MAMBA_IMPORT_ERROR
    global _TEMPORAL_MAMBA_RESOLVED

    if _TEMPORAL_MAMBA_RESOLVED:
        return _TEMPORAL_MAMBA_IMPORTED, TemporalMambaCore, _TEMPORAL_MAMBA_IMPORT_ERROR

    try:
        from mamba_ssm.modules.mamba_simple import Mamba as _ResolvedTemporalMambaCore

        TemporalMambaCore = _ResolvedTemporalMambaCore
        _TEMPORAL_MAMBA_IMPORTED = True
        _TEMPORAL_MAMBA_IMPORT_ERROR = None
    except Exception as exc:
        TemporalMambaCore = None
        _TEMPORAL_MAMBA_IMPORTED = False
        _TEMPORAL_MAMBA_IMPORT_ERROR = exc

    _TEMPORAL_MAMBA_RESOLVED = True
    return _TEMPORAL_MAMBA_IMPORTED, TemporalMambaCore, _TEMPORAL_MAMBA_IMPORT_ERROR


def _make_mlp(input_dim: int, hidden_dims: List[int], dropout: float) -> nn.Sequential:
    layers = []
    current_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.extend([
            nn.Linear(current_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        ])
        current_dim = hidden_dim
    return nn.Sequential(*layers)


def _largest_group_divisor(num_channels: int, max_groups: int) -> int:
    for groups in range(min(max_groups, num_channels), 0, -1):
        if num_channels % groups == 0:
            return groups
    return 1


def _replace_batchnorm2d_with_groupnorm(module: nn.Module, max_groups: int = 32) -> None:
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            groups = _largest_group_divisor(child.num_features, max_groups)
            setattr(module, name, nn.GroupNorm(groups, child.num_features, eps=child.eps, affine=True))
        else:
            _replace_batchnorm2d_with_groupnorm(child, max_groups=max_groups)


class ImagePatchCNN(nn.Module):
    def __init__(self, in_channels: int, conv_channels: List[int], out_dim: int, dropout: float):
        super().__init__()
        layers = []
        current_channels = in_channels
        for out_channels in conv_channels:
            layers.extend([
                nn.Conv2d(current_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.GELU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ])
            current_channels = out_channels
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(current_channels, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, patch_tensor: torch.Tensor) -> torch.Tensor:
        encoded = self.features(patch_tensor)
        encoded = self.pool(encoded)
        return self.proj(encoded)


class ResNetImageEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        variant: str,
        pretrained: bool,
        out_dim: int,
        dropout: float,
        trainable: bool = True,
        norm_layer: str = "batchnorm",
        groupnorm_num_groups: int = 32,
    ):
        super().__init__()
        if not _TORCHVISION_IMPORTED or tv_models is None:
            raise ImportError(
                f"torchvision is required for image_encoder_type='{variant}', but import failed: {_TORCHVISION_IMPORT_ERROR}"
            )

        if variant == "resnet18":
            weights = tv_models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = tv_models.resnet18(weights=weights)
            feature_dim = 512
        elif variant == "resnet50":
            weights = tv_models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            backbone = tv_models.resnet50(weights=weights)
            feature_dim = 2048
        else:
            raise ValueError(f"Unsupported ResNet variant: {variant}")

        backbone.conv1 = self._build_input_conv(backbone.conv1, in_channels, pretrained)
        backbone.fc = nn.Identity()
        norm_layer = str(norm_layer or "batchnorm").lower()
        if norm_layer in {"groupnorm", "gn"}:
            _replace_batchnorm2d_with_groupnorm(backbone, max_groups=int(groupnorm_num_groups))
        elif norm_layer not in {"batchnorm", "bn"}:
            raise ValueError(f"Unsupported ResNet norm_layer: {norm_layer}")
        self.backbone = backbone

        if not trainable:
            for param in self.backbone.parameters():
                param.requires_grad_(False)

        self.proj = nn.Sequential(
            nn.Linear(feature_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    @staticmethod
    def _build_input_conv(old_conv: nn.Conv2d, in_channels: int, pretrained: bool) -> nn.Conv2d:
        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=(old_conv.bias is not None),
        )

        with torch.no_grad():
            if pretrained:
                if in_channels == old_conv.in_channels:
                    new_conv.weight.copy_(old_conv.weight)
                elif in_channels > old_conv.in_channels:
                    new_conv.weight[:, :old_conv.in_channels].copy_(old_conv.weight)
                    mean_weight = old_conv.weight.mean(dim=1, keepdim=True)
                    repeat_count = in_channels - old_conv.in_channels
                    new_conv.weight[:, old_conv.in_channels:].copy_(mean_weight.repeat(1, repeat_count, 1, 1))
                else:
                    new_conv.weight.copy_(old_conv.weight[:, :in_channels])
                if old_conv.bias is not None and new_conv.bias is not None:
                    new_conv.bias.copy_(old_conv.bias)
            else:
                nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
                if new_conv.bias is not None:
                    nn.init.zeros_(new_conv.bias)

        return new_conv

    def forward(self, patch_tensor: torch.Tensor) -> torch.Tensor:
        features = self.backbone(patch_tensor)
        return self.proj(features)


class _FallbackTemporalSequenceModel(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.gru = nn.GRU(input_size=d_model, hidden_size=d_model, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.gru(x)
        return self.dropout(output)


class _GRUTemporalSequenceModel(nn.Module):
    def __init__(self, d_model: int, dropout: float, num_layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.gru(x)
        return self.dropout(output)


class TemporalMambaBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        dropout: float,
    ):
        super().__init__()
        self.pre_norm = nn.LayerNorm(d_model)
        temporal_mamba_imported, temporal_mamba_core, temporal_mamba_import_error = _resolve_temporal_mamba_backend()
        if temporal_mamba_imported and temporal_mamba_core is not None:
            self.sequence_model = temporal_mamba_core(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            self.using_mamba = True
        else:
            self.sequence_model = _FallbackTemporalSequenceModel(d_model=d_model, dropout=dropout)
            self.using_mamba = False
            logger.warning(
                "Temporal Mamba block is using GRU fallback because mamba_ssm could not be imported: %s",
                temporal_mamba_import_error,
            )

        self.post_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sequence_model(self.pre_norm(x))
        x = x + self.ffn(self.post_norm(x))
        return x


class TemporalMambaEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        out_dim: int,
        num_layers: int,
        d_state: int,
        d_conv: int,
        expand: int,
        dropout: float,
        encoder_type: str = "mamba",
        transformer_nhead: int = 4,
        transformer_dim_feedforward: Optional[int] = None,
    ):
        super().__init__()
        self.encoder_type = encoder_type.lower()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList()
        self.sequence_model = None
        if self.encoder_type == "mamba":
            self.blocks = nn.ModuleList(
                [
                    TemporalMambaBlock(
                        d_model=d_model,
                        d_state=d_state,
                        d_conv=d_conv,
                        expand=expand,
                        dropout=dropout,
                    )
                    for _ in range(num_layers)
                ]
            )
        elif self.encoder_type == "gru":
            self.sequence_model = _GRUTemporalSequenceModel(
                d_model=d_model,
                dropout=dropout,
                num_layers=num_layers,
            )
        elif self.encoder_type == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=transformer_nhead,
                dim_feedforward=transformer_dim_feedforward or d_model * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.sequence_model = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        elif self.encoder_type == "mean":
            self.sequence_model = nn.Identity()
        else:
            raise ValueError(f"Unsupported sequence encoder type: {encoder_type}")
        self.final_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, sequence_features: torch.Tensor, sequence_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = torch.nan_to_num(sequence_features)
        x = self.input_proj(x)

        if sequence_mask is not None:
            mask = sequence_mask.to(dtype=x.dtype)
            x = x * mask.unsqueeze(-1)
        else:
            mask = None

        if self.encoder_type == "mamba":
            for block in self.blocks:
                x = block(x)
        elif self.encoder_type == "transformer":
            key_padding_mask = None
            if mask is not None:
                key_padding_mask = mask <= 0
            x = self.sequence_model(x, src_key_padding_mask=key_padding_mask)
        else:
            x = self.sequence_model(x)

        x = self.final_norm(x)

        if mask is not None:
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            pooled = (x * mask.unsqueeze(-1)).sum(dim=1) / denom
        else:
            pooled = x.mean(dim=1)

        return self.output_proj(pooled)


class FluxTransferModel(nn.Module):
    def __init__(
        self,
        tabular_input_dim: int,
        metadata_input_dim: int,
        tabular_hidden_dims: Optional[List[int]] = None,
        metadata_hidden_dims: Optional[List[int]] = None,
        fusion_hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        use_image_branch: bool = False,
        image_patch_dim: int = 0,
        image_channels: int = 0,
        image_encoder_type: str = "resnet18",
        image_resnet_variant: str = "resnet18",
        image_resnet_pretrained: bool = False,
        image_norm_layer: str = "batchnorm",
        image_groupnorm_num_groups: int = 32,
        image_branch_trainable: bool = True,
        image_branch_hidden_dims: Optional[List[int]] = None,
        image_embedding_dim: int = 128,
        use_sequence_branch: bool = False,
        sequence_encoder_type: str = "mamba",
        sequence_input_dim: int = 0,
        sequence_embedding_dim: int = 128,
        mamba_d_model: int = 128,
        mamba_layers: int = 2,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        transformer_nhead: int = 4,
        transformer_dim_feedforward: Optional[int] = None,
        fusion_mode: str = "concat",
        image_gate_init_bias: float = -2.0,
        output_dim: int = 1,
        **kwargs,
    ):
        super().__init__()
        self.use_image_branch = use_image_branch
        self.use_sequence_branch = use_sequence_branch
        self.sequence_encoder_type = sequence_encoder_type
        self.image_patch_dim = image_patch_dim
        self.image_channels = image_channels
        self.image_encoder_type = image_encoder_type
        self.fusion_mode = str(fusion_mode or "concat").lower()

        tabular_hidden_dims = tabular_hidden_dims or [256, 128]
        metadata_hidden_dims = metadata_hidden_dims or [32]
        fusion_hidden_dims = fusion_hidden_dims or [128, 64]
        image_branch_hidden_dims = image_branch_hidden_dims or [32, 64, 128]

        self.tabular_encoder = _make_mlp(tabular_input_dim, tabular_hidden_dims, dropout)
        tabular_out_dim = tabular_hidden_dims[-1] if tabular_hidden_dims else tabular_input_dim

        self.metadata_encoder = _make_mlp(metadata_input_dim, metadata_hidden_dims, dropout)
        metadata_out_dim = metadata_hidden_dims[-1] if metadata_hidden_dims else metadata_input_dim

        self.sequence_encoder = None
        sequence_out_dim = 0
        if self.use_sequence_branch:
            if sequence_input_dim <= 0:
                raise ValueError("Sequence branch is enabled but `sequence_input_dim` is not positive.")
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
            sequence_out_dim = sequence_embedding_dim

        self.image_encoder = None
        image_out_dim = 0
        if self.use_image_branch:
            if image_encoder_type in {"resnet18", "resnet50"}:
                if image_channels <= 0:
                    raise ValueError(
                        f"image_encoder_type='{image_encoder_type}' requires `image_channels` > 0."
                    )
                self.image_encoder = ResNetImageEncoder(
                    in_channels=image_channels,
                    variant=image_resnet_variant,
                    pretrained=image_resnet_pretrained,
                    out_dim=image_embedding_dim,
                    dropout=dropout,
                    trainable=image_branch_trainable,
                    norm_layer=image_norm_layer,
                    groupnorm_num_groups=image_groupnorm_num_groups,
                )
                image_out_dim = image_embedding_dim
            elif image_encoder_type == "light_cnn":
                if image_channels <= 0:
                    raise ValueError("image_encoder_type='light_cnn' requires `image_channels` > 0.")
                self.image_encoder = ImagePatchCNN(
                    in_channels=image_channels,
                    conv_channels=image_branch_hidden_dims,
                    out_dim=image_embedding_dim,
                    dropout=dropout,
                )
                image_out_dim = image_embedding_dim
            elif image_patch_dim > 0:
                self.image_encoder = _make_mlp(image_patch_dim, image_branch_hidden_dims, dropout)
                image_out_dim = image_branch_hidden_dims[-1] if image_branch_hidden_dims else image_patch_dim
            else:
                raise ValueError(
                    "Image branch is enabled but the configured encoder could not infer a valid image input path."
                )

        non_image_input_dim = tabular_out_dim + metadata_out_dim + sequence_out_dim
        fusion_input_dim = non_image_input_dim + image_out_dim
        if self.fusion_mode == "gated_residual_image":
            if not self.use_image_branch or image_out_dim <= 0:
                raise ValueError("fusion_mode='gated_residual_image' requires an enabled image branch.")
            self.fusion_head = None
            self.regression_head = None
            self.base_head = nn.Sequential(
                _make_mlp(non_image_input_dim, fusion_hidden_dims, dropout),
                nn.Linear(fusion_hidden_dims[-1] if fusion_hidden_dims else non_image_input_dim, output_dim),
            )
            self.correction_head = nn.Sequential(
                _make_mlp(fusion_input_dim, fusion_hidden_dims, dropout),
                nn.Linear(fusion_hidden_dims[-1] if fusion_hidden_dims else fusion_input_dim, output_dim),
            )
            gate_hidden_dim = fusion_hidden_dims[0] if fusion_hidden_dims else fusion_input_dim
            self.gate_head = nn.Sequential(
                nn.Linear(fusion_input_dim, gate_hidden_dim),
                nn.LayerNorm(gate_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(gate_hidden_dim, output_dim),
                nn.Sigmoid(),
            )
            with torch.no_grad():
                gate_linear = self.gate_head[-2]
                if isinstance(gate_linear, nn.Linear) and gate_linear.bias is not None:
                    gate_linear.bias.fill_(float(image_gate_init_bias))
        elif self.fusion_mode == "concat":
            self.fusion_head = _make_mlp(fusion_input_dim, fusion_hidden_dims, dropout)
            fusion_out_dim = fusion_hidden_dims[-1] if fusion_hidden_dims else fusion_input_dim
            self.regression_head = nn.Linear(fusion_out_dim, output_dim)
            self.base_head = None
            self.correction_head = None
            self.gate_head = None
        else:
            raise ValueError(f"Unsupported fusion_mode: {fusion_mode}")

    def print_trainable_parameters_summary(self):
        logger.info("-" * 60 + "\nModel Trainable Parameters Summary:")
        total_params = 0
        trainable_params = 0
        parts = {
            "Tabular Encoder": self.tabular_encoder,
            "Metadata Encoder": self.metadata_encoder,
            "Sequence Encoder": self.sequence_encoder,
            "Image Encoder": self.image_encoder,
            "Fusion Head": self.fusion_head,
            "Regression Head": self.regression_head,
            "Base Head": self.base_head,
            "Correction Head": self.correction_head,
            "Gate Head": self.gate_head,
        }
        for name, part in parts.items():
            if part is None:
                continue
            part_total = sum(p.numel() for p in part.parameters())
            part_trainable = sum(p.numel() for p in part.parameters() if p.requires_grad)
            total_params += part_total
            trainable_params += part_trainable
            logger.info(f"  {name:<25}: Total={part_total / 1e6:.2f}M, Trainable={part_trainable / 1e6:.2f}M")
        logger.info(f"  {'Total Model':<25}: Total={total_params / 1e6:.2f}M, Trainable={trainable_params / 1e6:.2f}M\n" + "-" * 60)

    def forward(self, batch: Dict[str, Any]) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        tabular_features = batch['tabular_features']
        metadata_features = batch['metadata_features']

        encoded_parts = [
            self.tabular_encoder(tabular_features),
            self.metadata_encoder(metadata_features),
        ]
        non_image_parts = list(encoded_parts)

        if self.use_sequence_branch:
            sequence_features = batch.get('sequence_features')
            sequence_mask = batch.get('sequence_mask')
            if sequence_features is None or sequence_features.numel() == 0:
                raise ValueError("Sequence branch is enabled but no sequence_features were provided.")
            if sequence_features.ndim == 2:
                sequence_features = sequence_features.unsqueeze(0)
            if sequence_mask is not None and sequence_mask.ndim == 1:
                sequence_mask = sequence_mask.unsqueeze(0)
            sequence_context = self.sequence_encoder(sequence_features, sequence_mask)
            encoded_parts.append(sequence_context)
            non_image_parts.append(sequence_context)

        if self.use_image_branch:
            patch_tensor = batch.get('patch_tensor')
            if patch_tensor is None or patch_tensor.numel() == 0:
                raise ValueError("Image branch is enabled but no patch_tensor was provided.")
            patch_tensor = torch.nan_to_num(patch_tensor)
            if patch_tensor.ndim == 3:
                patch_tensor = patch_tensor.unsqueeze(0)
            if patch_tensor.ndim == 5:
                batch_size, num_patches, channels, height, width = patch_tensor.shape
                flat_patches = patch_tensor.reshape(batch_size * num_patches, channels, height, width)
                flat_encoded = self.image_encoder(flat_patches)
                patch_encoded = flat_encoded.reshape(batch_size, num_patches, -1)
                patch_mask = batch.get('patch_mask')
                if patch_mask is None:
                    patch_mask = patch_tensor.new_ones((batch_size, num_patches))
                patch_mask = patch_mask.to(device=patch_encoded.device, dtype=patch_encoded.dtype)
                denom = patch_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
                image_context = (patch_encoded * patch_mask.unsqueeze(-1)).sum(dim=1) / denom
                encoded_parts.append(image_context)
            else:
                encoded_parts.append(self.image_encoder(patch_tensor))

        fused = torch.cat(encoded_parts, dim=-1)
        if self.fusion_mode == "gated_residual_image":
            non_image_fused = torch.cat(non_image_parts, dim=-1)
            base = self.base_head(non_image_fused)
            correction = self.correction_head(fused)
            gate = self.gate_head(fused)
            predictions = (base + gate * correction).squeeze(-1)
        else:
            fused = self.fusion_head(fused)
            predictions = self.regression_head(fused).squeeze(-1)
        aux_loss = predictions.new_zeros(())

        if self.training:
            return predictions, aux_loss
        return predictions
