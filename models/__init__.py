# models/__init__.py

from .flux_transfer_model import FluxTransferModel
from .flux_hierarchical_moe_model import FluxHierarchicalMoEModel
from .attention_blocks import (
    ChannelAttentionCBAM,
    SpatialAttentionEnhanced,
    BasicConvBlock,
    DepthwiseSeparableConvBlock,
    BottleneckBlock,
    BottleneckWithDWSCBlock,
    FEDAB_Enhanced_DWSC,
)

ADVANCED_FUSION_IMPORT_ERROR = None


def _load_advanced_fusion_symbols():
    global ADVANCED_FUSION_IMPORT_ERROR

    try:
        from .advanced_fusion_model import AdvancedFusionModel, MLPHead
        return AdvancedFusionModel, MLPHead
    except Exception as exc:
        ADVANCED_FUSION_IMPORT_ERROR = exc
        raise


def __getattr__(name):
    if name in {"AdvancedFusionModel", "MLPHead"}:
        advanced_fusion_model, mlp_head = _load_advanced_fusion_symbols()
        return advanced_fusion_model if name == "AdvancedFusionModel" else mlp_head
    if name == "ADVANCED_FUSION_IMPORT_ERROR":
        return ADVANCED_FUSION_IMPORT_ERROR
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "AdvancedFusionModel",
    "MLPHead",
    "FluxTransferModel",
    "FluxHierarchicalMoEModel",
    "ChannelAttentionCBAM",
    "SpatialAttentionEnhanced",
    "BasicConvBlock",
    "DepthwiseSeparableConvBlock",
    "BottleneckBlock",
    "BottleneckWithDWSCBlock",
    "FEDAB_Enhanced_DWSC",
    "ADVANCED_FUSION_IMPORT_ERROR",
]
