"""
IronPulse — PulseMind AI Engine (PyTorch)
==========================================
Production-grade neural network library for exercise recommendation
and workout plan generation.

Architectures
─────────────
  • PulseMindMLP       — Deep feed-forward with BatchNorm, Dropout, GELU
  • PulseMindResNet    — Residual blocks with skip connections
  • PulseMindAttention — Multi-head self-attention over feature tokens
  • PulseMindEnsemble  — Soft-voting ensemble of sub-models

All models produce:
  Classification  →  softmax probability over exercises
  Regression      →  scalar target weight prediction

Feature Vector (18-dim)
───────────────────────
  [0]  session_volume      Total weight × reps this session
  [1]  total_sets          Number of sets performed
  [2]  unique_exercises    Distinct exercises count
  [3]  avg_weight          Mean weight per set
  [4]  max_weight          Heaviest set weight
  [5]  avg_reps            Mean reps per set
  [6]  max_reps            Highest reps in a set
  [7]  avg_1rm             Mean estimated 1RM across sets
  [8]  best_1rm            Best estimated 1RM
  [9]  compound_ratio      Fraction of compound movements
  [10] chest_pct           Chest volume share
  [11] back_pct            Back volume share
  [12] legs_pct            Legs volume share
  [13] shoulder_pct        Shoulder volume share
  [14] arms_pct            Arms volume share
  [15] core_pct            Core volume share
  [16] rest_days           Days since last session
  [17] monthly_frequency   Sessions in the last 30 days
"""

import logging
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

N_FEATURES = 18
FEATURE_NAMES = [
    'Session Volume', 'Total Sets', 'Unique Exercises',
    'Avg Weight', 'Max Weight', 'Avg Reps', 'Max Reps',
    'Avg 1RM', 'Best 1RM', 'Compound Ratio',
    'Chest %', 'Back %', 'Legs %', 'Shoulder %', 'Arms %', 'Core %',
    'Rest Days', 'Sessions/Month',
]

WEIGHTS_DIR = Path(__file__).parent / 'weights'
WEIGHTS_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Architecture 1: Deep MLP
# ══════════════════════════════════════════════════════════════════════════════

class PulseMindMLP(nn.Module):
    """
    Deep Multi-Layer Perceptron.

    Architecture:
        Input (18) → [Linear → BatchNorm → GELU → Dropout] × N → Output

    Design choices:
        - GELU activation (smoother than ReLU, used in BERT/GPT)
        - BatchNorm for training stability
        - Dropout for regularisation
        - He (Kaiming) initialisation
    """

    def __init__(self, input_size: int = N_FEATURES, output_size: int = 60,
                 hidden_sizes: List[int] = None, dropout: float = 0.3,
                 mode: str = 'classification'):
        super().__init__()
        self.mode = mode
        self.architecture = 'MLP'

        if hidden_sizes is None:
            hidden_sizes = [256, 128, 64]

        layers = []
        prev_size = input_size
        for h in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, h),
                nn.BatchNorm1d(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_size = h

        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev_size, output_size)

        # Kaiming init
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: Input → Hidden layers → Head

        For classification: returns log-softmax probabilities
        For regression: returns raw scalar predictions
        """
        h = self.backbone(x)
        out = self.head(h)
        if self.mode == 'classification':
            return F.log_softmax(out, dim=-1)
        return out

    def get_architecture_info(self) -> Dict:
        """Return human-readable architecture description."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        layers_info = []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.BatchNorm1d, nn.Dropout)):
                info = {'name': name, 'type': type(module).__name__}
                if isinstance(module, nn.Linear):
                    info['in'] = module.in_features
                    info['out'] = module.out_features
                    info['params'] = module.weight.numel() + (module.bias.numel() if module.bias is not None else 0)
                elif isinstance(module, nn.BatchNorm1d):
                    info['features'] = module.num_features
                elif isinstance(module, nn.Dropout):
                    info['rate'] = module.p
                layers_info.append(info)
        return {
            'architecture': self.architecture,
            'total_params': total_params,
            'trainable_params': trainable,
            'layers': layers_info,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Architecture 2: ResNet (Residual Network)
# ══════════════════════════════════════════════════════════════════════════════

class ResidualBlock(nn.Module):
    """
    Residual block: x + F(x)
    Where F = Linear → BN → GELU → Dropout → Linear → BN

    Skip connection prevents vanishing gradients in deep networks.
    """

    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class PulseMindResNet(nn.Module):
    """
    Residual Network for tabular data.

    Architecture:
        Input (18) → Projection (→ hidden_dim)
        → [ResidualBlock] × n_blocks
        → Output Head

    Key insight: Skip connections let gradients flow directly,
    enabling much deeper networks without degradation.
    """

    def __init__(self, input_size: int = N_FEATURES, output_size: int = 60,
                 hidden_dim: int = 128, n_blocks: int = 4,
                 dropout: float = 0.2, mode: str = 'classification'):
        super().__init__()
        self.mode = mode
        self.architecture = 'ResNet'

        self.proj = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )

        self.blocks = nn.Sequential(*[
            ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)
        ])

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_size),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        h = self.blocks(h)
        out = self.head(h)
        if self.mode == 'classification':
            return F.log_softmax(out, dim=-1)
        return out

    def get_architecture_info(self) -> Dict:
        total_params = sum(p.numel() for p in self.parameters())
        return {
            'architecture': self.architecture,
            'total_params': total_params,
            'trainable_params': total_params,
            'projection_dim': self.proj[0].out_features,
            'n_blocks': len(self.blocks),
            'layers': [
                {'name': 'projection', 'type': 'Linear+BN+GELU',
                 'in': self.proj[0].in_features, 'out': self.proj[0].out_features},
            ] + [
                {'name': f'res_block_{i}', 'type': 'ResidualBlock',
                 'dim': self.proj[0].out_features}
                for i in range(len(self.blocks))
            ] + [
                {'name': 'head', 'type': 'Linear',
                 'in': self.proj[0].out_features,
                 'out': self.head[-1].out_features}
            ],
        }


# ══════════════════════════════════════════════════════════════════════════════
# Architecture 3: Attention Network
# ══════════════════════════════════════════════════════════════════════════════

class FeatureAttention(nn.Module):
    """
    Multi-head attention over input features.

    Each of the 18 features is treated as a "token".
    Attention learns which features are most important for each prediction.

    Q, K, V are learned projections of each feature scalar → d_model vector.
    """

    def __init__(self, d_model: int = 64, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, n_features, d_model)
        Returns:
            output: (batch, n_features, d_model)
            attn_weights: (batch, n_heads, n_features, n_features)
        """
        B, N, D = x.shape

        Q = self.W_q(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(B, N, D)
        out = self.W_o(context)

        return self.norm(x + out), attn


class PulseMindAttention(nn.Module):
    """
    Transformer-inspired Attention Network.

    Architecture:
        Input (18 scalars)
        → Feature Embedding (each scalar → d_model vector)
        → [Multi-Head Self-Attention + FFN] × n_layers
        → Global Average Pooling
        → Classification/Regression Head

    Interpretability: Attention weights show which features
    the model considers most important for each prediction.
    """

    def __init__(self, input_size: int = N_FEATURES, output_size: int = 60,
                 d_model: int = 64, n_heads: int = 4, n_layers: int = 2,
                 dropout: float = 0.2, mode: str = 'classification'):
        super().__init__()
        self.mode = mode
        self.architecture = 'Attention'
        self.input_size = input_size

        # Embed each feature scalar into d_model dimensions
        self.feature_embed = nn.Linear(1, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, input_size, d_model) * 0.02)

        self.attention_layers = nn.ModuleList([
            FeatureAttention(d_model, n_heads, dropout) for _ in range(n_layers)
        ])

        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 4, d_model),
                nn.LayerNorm(d_model),
            ) for _ in range(n_layers)
        ])

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_size),
        )

        self.last_attn_weights = None
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:  (batch, 18)
        Output: (batch, n_classes) log-probabilities
        """
        B, N = x.shape
        # Reshape: (B, 18) → (B, 18, 1) → embed → (B, 18, d_model)
        tokens = self.feature_embed(x.unsqueeze(-1)) + self.pos_embed

        for attn_layer, ffn in zip(self.attention_layers, self.ffn_layers):
            tokens, attn_w = attn_layer(tokens)
            tokens = tokens + ffn(tokens)
            self.last_attn_weights = attn_w

        # Global average pooling: (B, 18, d_model) → (B, d_model)
        pooled = tokens.mean(dim=1)
        out = self.head(pooled)

        if self.mode == 'classification':
            return F.log_softmax(out, dim=-1)
        return out

    def get_attention_weights(self) -> Optional[np.ndarray]:
        """Get last attention weights for interpretability."""
        if self.last_attn_weights is not None:
            return self.last_attn_weights.detach().cpu().numpy()
        return None

    def get_feature_importance(self) -> np.ndarray:
        """Derive feature importance from average attention weights."""
        if self.last_attn_weights is None:
            return np.ones(self.input_size) / self.input_size
        # Average across batch and heads: (n_features, n_features)
        attn = self.last_attn_weights.detach().mean(dim=(0, 1)).cpu().numpy()
        # Column-sum = how much attention each feature receives
        importance = attn.sum(axis=0)
        importance = importance / importance.sum()
        return importance

    def get_architecture_info(self) -> Dict:
        total_params = sum(p.numel() for p in self.parameters())
        return {
            'architecture': self.architecture,
            'total_params': total_params,
            'trainable_params': total_params,
            'd_model': self.feature_embed.out_features,
            'n_heads': self.attention_layers[0].n_heads,
            'n_layers': len(self.attention_layers),
            'layers': [
                {'name': 'feature_embed', 'type': 'Linear(1→d_model)',
                 'out': self.feature_embed.out_features},
                {'name': 'positional_embed', 'type': 'Learned',
                 'shape': f'{self.input_size}×{self.feature_embed.out_features}'},
            ] + [
                {'name': f'attention_{i}', 'type': 'MultiHeadAttention+FFN',
                 'd_model': self.feature_embed.out_features}
                for i in range(len(self.attention_layers))
            ] + [
                {'name': 'pool', 'type': 'GlobalAveragePool'},
                {'name': 'head', 'type': 'Linear→GELU→Linear',
                 'out': self.head[-1].out_features},
            ],
        }


# ══════════════════════════════════════════════════════════════════════════════
# Architecture 4: Ensemble
# ══════════════════════════════════════════════════════════════════════════════

class PulseMindEnsemble(nn.Module):
    """
    Ensemble: combines MLP + ResNet via learned soft-voting.

    Architecture:
        Input → MLP  → logits_1
              → ResNet → logits_2
        → Learned weighted combination → Output

    Typically achieves highest accuracy by reducing variance.
    """

    def __init__(self, input_size: int = N_FEATURES, output_size: int = 60,
                 dropout: float = 0.2, mode: str = 'classification'):
        super().__init__()
        self.mode = mode
        self.architecture = 'Ensemble'

        self.mlp = PulseMindMLP(
            input_size, output_size,
            hidden_sizes=[128, 64], dropout=dropout, mode='regression'
        )
        self.resnet = PulseMindResNet(
            input_size, output_size,
            hidden_dim=96, n_blocks=3, dropout=dropout, mode='regression'
        )

        # Learned combination weights
        self.weight_mlp = nn.Parameter(torch.tensor(0.5))
        self.weight_res = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = F.softmax(torch.stack([self.weight_mlp, self.weight_res]), dim=0)
        logits = w[0] * self.mlp(x) + w[1] * self.resnet(x)
        if self.mode == 'classification':
            return F.log_softmax(logits, dim=-1)
        return logits

    def get_architecture_info(self) -> Dict:
        total_params = sum(p.numel() for p in self.parameters())
        w = F.softmax(torch.stack([self.weight_mlp, self.weight_res]), dim=0)
        return {
            'architecture': self.architecture,
            'total_params': total_params,
            'trainable_params': total_params,
            'sub_models': ['MLP', 'ResNet'],
            'weights': [f'{w[0].item():.2%}', f'{w[1].item():.2%}'],
            'layers': [
                {'name': 'sub_mlp', 'type': 'PulseMindMLP',
                 'params': sum(p.numel() for p in self.mlp.parameters())},
                {'name': 'sub_resnet', 'type': 'PulseMindResNet',
                 'params': sum(p.numel() for p in self.resnet.parameters())},
                {'name': 'voting', 'type': 'LearnedSoftVoting'},
            ],
        }


# ══════════════════════════════════════════════════════════════════════════════
# Model Registry & Factory
# ══════════════════════════════════════════════════════════════════════════════

MODEL_REGISTRY = {
    'mlp':       PulseMindMLP,
    'resnet':    PulseMindResNet,
    'attention': PulseMindAttention,
    'ensemble':  PulseMindEnsemble,
}


def build_model(arch: str, input_size: int = N_FEATURES,
                output_size: int = 60, mode: str = 'classification',
                **kwargs) -> nn.Module:
    """
    Factory function to build a PulseMind model.

    Parameters
    ----------
    arch : str
        One of 'mlp', 'resnet', 'attention', 'ensemble'
    input_size : int
        Number of input features (default 18)
    output_size : int
        Number of output classes/targets
    mode : str
        'classification' or 'regression'
    **kwargs : dict
        Architecture-specific arguments (hidden_sizes, n_blocks, etc.)

    Returns
    -------
    nn.Module
        Initialised model ready for training
    """
    arch = arch.lower()
    if arch not in MODEL_REGISTRY:
        raise ValueError(f"Unknown architecture '{arch}'. Choose from: {list(MODEL_REGISTRY.keys())}")

    model_cls = MODEL_REGISTRY[arch]
    return model_cls(input_size=input_size, output_size=output_size, mode=mode, **kwargs)


def save_model(model: nn.Module, name: str = 'pulsemind') -> Path:
    """Save model weights + architecture info to disk."""
    path = WEIGHTS_DIR / f'{name}.pt'
    torch.save({
        'architecture': model.architecture,
        'state_dict': model.state_dict(),
        'arch_info': model.get_architecture_info(),
    }, path)
    logger.info(f"Model saved to {path}")
    return path


def load_model(name: str = 'pulsemind', input_size: int = N_FEATURES,
               output_size: int = 60, mode: str = 'classification') -> nn.Module:
    """Load model from disk."""
    path = WEIGHTS_DIR / f'{name}.pt'
    if not path.exists():
        raise FileNotFoundError(f"No saved model at {path}")

    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    arch = checkpoint['architecture'].lower()
    model = build_model(arch, input_size=input_size, output_size=output_size, mode=mode)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    logger.info(f"Loaded {arch} model from {path}")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Feature Importance (gradient-based saliency)
# ══════════════════════════════════════════════════════════════════════════════

def compute_feature_importance(model: nn.Module, X: torch.Tensor) -> np.ndarray:
    """
    Compute feature importance via gradient-based saliency.

    Method: For each input, compute |∂output/∂input| and average across
    the batch. Features with large gradients have the most influence.
    """
    model.eval()
    X = X.clone().detach().requires_grad_(True)

    output = model(X)
    if output.dim() > 1 and output.shape[1] > 1:
        # Classification: sum of max-class log-probs
        target = output.max(dim=1).values.sum()
    else:
        target = output.sum()

    target.backward()

    importance = X.grad.abs().mean(dim=0).detach().cpu().numpy()
    importance = importance / (importance.sum() + 1e-9)
    return importance
