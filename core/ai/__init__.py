"""
PulseMind AI — IronPulse Intelligence Engine
=============================================
PyTorch-based neural networks for exercise recommendation.

Architecture: MLP, ResNet, AttentionNet, Ensemble
Training:     AdamW + cosine LR + FGSM adversarial + Mixup
Explainability: SHAP + Integrated Gradients + gradient saliency
"""
from .engine import (
    N_FEATURES,
    FEATURE_NAMES,
    build_model,
    save_model,
    load_model,
    compute_feature_importance,
    PulseMindMLP,
    PulseMindResNet,
    PulseMindAttention,
    PulseMindEnsemble,
    MODEL_REGISTRY,
    TORCH_AVAILABLE,
)
from .trainer import (
    PulseMindTrainer,
    prepare_workout_data,
    train_and_evaluate,
    fgsm_attack,
    mixup_batch,
    compute_shap_values,
    compute_integrated_gradients,
)

__all__ = [
    # Engine
    'N_FEATURES', 'FEATURE_NAMES',
    'build_model', 'save_model', 'load_model',
    'compute_feature_importance',
    'PulseMindMLP', 'PulseMindResNet', 'PulseMindAttention', 'PulseMindEnsemble',
    'MODEL_REGISTRY', 'TORCH_AVAILABLE',
    # Trainer
    'PulseMindTrainer', 'prepare_workout_data', 'train_and_evaluate',
    # Adversarial
    'fgsm_attack', 'mixup_batch',
    # Explainability
    'compute_shap_values', 'compute_integrated_gradients',
]
