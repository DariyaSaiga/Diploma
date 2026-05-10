from old_arch.models.late_fusion_cnn_bilstm import LateFusionCNNBiLSTM
from old_arch.models.mult_cross_attention import MulTCrossAttention
from old_arch.models.bottleneck_fusion import BottleneckFusion

MODEL_REGISTRY = {
    "late_fusion": LateFusionCNNBiLSTM,
    "mult": MulTCrossAttention,
    "bottleneck": BottleneckFusion,
}


def build_model(name: str, **kwargs):
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Choose from: {list(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[name](**kwargs)


__all__ = [
    "LateFusionCNNBiLSTM",
    "MulTCrossAttention",
    "BottleneckFusion",
    "MODEL_REGISTRY",
    "build_model",
]
