"""
IronPulse — PulseMind AI Module
=================================
Neural network architectures, optimisers, and training utilities
built from scratch with NumPy.

Available architectures:
  PulseMindMLP          — Deep Multi-Layer Perceptron
  PulseMindResNet       — Residual Network (skip-connections)
  PulseMindAttentionNet — Transformer-inspired (attention-based)
  PulseMindEnsemble     — Soft-voting ensemble

Training:
  PulseMindTrainer — Universal trainer (Adam/SGD, LR scheduling, W&B, early stopping)

Quick usage:
    from core.ai.engine  import build_model
    from core.ai.trainer import PulseMindTrainer, prepare_workout_data

    model   = build_model('resnet', input_size=18, output_size=10)
    trainer = PulseMindTrainer(model, lr=1e-3, use_wandb=True)
    history = trainer.fit(X_train, y_train, X_val, y_val, epochs=200)
"""

from .engine import (
    PulseMindMLP, PulseMindResNet, PulseMindAttentionNet, PulseMindEnsemble,
    PulseMindClassifier, PulseMindRegressor, build_model,
)
from .trainer import (
    PulseMindTrainer,
    FEATURE_NAMES, N_FEATURES,
    Adam, SGDMomentum,
    StepLR, CosineAnnealingLR,
    EarlyStopping,
    prepare_workout_data, prepare_regression_data,
)

__all__ = [
    # Architectures
    'PulseMindMLP', 'PulseMindResNet', 'PulseMindAttentionNet', 'PulseMindEnsemble',
    # Factories
    'PulseMindClassifier', 'PulseMindRegressor', 'build_model',
    # Training
    'PulseMindTrainer', 'Adam', 'SGDMomentum',
    'StepLR', 'CosineAnnealingLR', 'EarlyStopping',
    # Data
    'prepare_workout_data', 'prepare_regression_data',
    # Constants
    'FEATURE_NAMES', 'N_FEATURES',
]
