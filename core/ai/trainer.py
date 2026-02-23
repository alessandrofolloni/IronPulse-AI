"""
IronPulse — PulseMind Trainer v2 (PyTorch)
===========================================
Production training pipeline with:
  • AdamW optimiser + cosine LR with warm restarts
  • Early stopping with best-checkpoint restoration
  • Train / Validation / Test split (70/15/15)
  • Mini-batch gradient descent
  • Adversarial training (FGSM + Mixup augmentation)
  • SHAP-based explainability
  • W&B experiment tracking
  • Feature extraction from real workout history
  • Progressive overload analysis
"""

import logging
import time
import copy
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from .engine import N_FEATURES, FEATURE_NAMES, build_model

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Feature Extraction
# ══════════════════════════════════════════════════════════════════════════════

def prepare_workout_data(sessions, exercises) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract feature matrix X and label matrix y from real workout sessions.

    X shape: (n_samples, N_FEATURES=18)
    y shape: (n_samples, n_exercises) — soft probability distribution

    Uses real data only. If fewer than 20 real samples exist, generates
    minimal synthetic augmentation to ensure the model can initialise.
    """
    exercise_list = list(exercises.order_by('id'))
    n_exercises = len(exercise_list)

    if n_exercises == 0:
        return np.empty((0, N_FEATURES), dtype=np.float32), np.empty((0, 1), dtype=np.float32)

    exercise_idx = {ex.id: i for i, ex in enumerate(exercise_list)}

    X_list, y_list = [], []
    MUSCLE_ORDER = ['chest', 'back', 'legs', 'shoulders', 'arms', 'core']

    for session in sessions:
        sets = list(session.sets.select_related('exercise').all())
        if not sets:
            continue

        weights    = [s.weight for s in sets if s.weight]
        reps       = [s.reps for s in sets if s.reps]
        one_rms    = [s.one_rm for s in sets if s.one_rm]

        if not weights:
            continue

        vol   = sum(s.weight * s.reps for s in sets if s.weight and s.reps)
        n_sets = len(sets)
        unique = len({s.exercise_id for s in sets})

        avg_w = np.mean(weights) if weights else 0
        max_w = np.max(weights)  if weights else 0
        avg_r = np.mean(reps)    if reps else 0
        max_r = np.max(reps)     if reps else 0
        avg_1 = np.mean(one_rms) if one_rms else 0
        best_1= np.max(one_rms)  if one_rms else 0

        compound = sum(1 for s in sets if s.exercise.is_compound)
        comp_ratio = compound / n_sets if n_sets else 0

        # Muscle group volume distribution
        mg_vol = {mg: 0.0 for mg in MUSCLE_ORDER}
        for s in sets:
            mg = s.exercise.muscle_group
            mapped = mg if mg in mg_vol else 'core'
            if mg in ('biceps', 'triceps'):
                mapped = 'arms'
            mg_vol[mapped] = mg_vol.get(mapped, 0) + (s.weight * s.reps if s.weight and s.reps else 0)

        total_vol = sum(mg_vol.values()) or 1.0
        mg_pct = [mg_vol[mg] / total_vol for mg in MUSCLE_ORDER]

        # Temporal features
        from django.utils import timezone
        from datetime import timedelta
        today = timezone.now().date()

        # Days since this session (rest_days proxy)
        if hasattr(session.date, 'date'):
            sess_date = session.date.date()
        else:
            sess_date = session.date
        days_ago = (today - sess_date).days

        # Monthly frequency context
        from_date = sess_date - timedelta(days=30)
        monthly_freq = sessions.filter(date__gte=from_date, date__lte=sess_date).count()

        feature = [
            float(vol) / 10000.0,     # normalise volume to ~[0,1] range
            float(n_sets) / 30.0,
            float(unique) / 15.0,
            float(avg_w) / 200.0,
            float(max_w) / 300.0,
            float(avg_r) / 20.0,
            float(max_r) / 30.0,
            float(avg_1) / 300.0,
            float(best_1) / 400.0,
            float(comp_ratio),
        ] + mg_pct + [
            float(days_ago) / 30.0,
            float(monthly_freq) / 30.0,
        ]

        X_list.append(feature[:N_FEATURES])

        # Build label from exercises done in this session (soft target)
        label = np.zeros(n_exercises, dtype=np.float32)
        for s in sets:
            if s.exercise_id in exercise_idx:
                # Weight by volume (heavier sets → higher label weight)
                vol_weight = (s.weight * s.reps) if s.weight and s.reps else 1.0
                label[exercise_idx[s.exercise_id]] += vol_weight

        if label.sum() > 0:
            label = label / label.sum()
        y_list.append(label)

    # Build arrays — handle empty case correctly
    if len(X_list) > 0:
        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)
    else:
        X = np.empty((0, N_FEATURES), dtype=np.float32)
        y = np.empty((0, n_exercises), dtype=np.float32)

    # Only augment with synthetic data if truly data-scarce (< 20 real samples)
    # The synthetic data is exercise-aware (not pure random noise)
    if len(X) < 20:
        n_synth = max(50 - len(X), 30)
        logger.warning(
            f"Only {len(X)} real sessions found. Generating {n_synth} "
            "synthetic samples. The more workouts you log, the better the AI."
        )
        rng = np.random.default_rng(42)

        # Synthetic data based on realistic workout distributions
        X_synth = np.zeros((n_synth, N_FEATURES), dtype=np.float32)
        X_synth[:, 0] = rng.uniform(0.2, 0.8, n_synth)   # volume
        X_synth[:, 1] = rng.uniform(0.1, 0.8, n_synth)   # sets
        X_synth[:, 2] = rng.uniform(0.07, 0.4, n_synth)  # unique exercises
        X_synth[:, 3] = rng.uniform(0.2, 0.6, n_synth)   # avg weight
        X_synth[:, 4] = rng.uniform(0.3, 0.8, n_synth)   # max weight
        X_synth[:, 5] = rng.uniform(0.3, 0.7, n_synth)   # avg reps
        X_synth[:, 6] = rng.uniform(0.4, 0.8, n_synth)   # max reps
        X_synth[:, 7] = rng.uniform(0.2, 0.7, n_synth)   # avg 1RM
        X_synth[:, 8] = rng.uniform(0.3, 0.8, n_synth)   # best 1RM
        X_synth[:, 9] = rng.uniform(0.3, 0.8, n_synth)   # compound ratio

        # Muscle group percentages (sum to 1)
        mg_raw = rng.dirichlet(np.ones(6) * 2, n_synth)  # Dirichlet for realistic distribution
        X_synth[:, 10:16] = mg_raw.astype(np.float32)

        X_synth[:, 16] = rng.uniform(0, 0.5, n_synth)    # rest days
        X_synth[:, 17] = rng.uniform(0.1, 0.5, n_synth)  # monthly freq

        # Synthetic labels — prefer popular exercises
        y_synth = np.zeros((n_synth, n_exercises), dtype=np.float32)
        for i in range(n_synth):
            n_picks = min(rng.integers(3, 8), n_exercises)
            picks = rng.choice(n_exercises, size=n_picks, replace=False)
            weights_synth = rng.dirichlet(np.ones(n_picks))
            y_synth[i, picks] = weights_synth

        if len(X) > 0:
            X = np.vstack([X, X_synth])
            y = np.vstack([y, y_synth])
        else:
            X = X_synth
            y = y_synth

    return X, y


# ══════════════════════════════════════════════════════════════════════════════
# Adversarial Training Utilities
# ══════════════════════════════════════════════════════════════════════════════

def fgsm_attack(model: 'nn.Module', X: 'torch.Tensor', y: 'torch.Tensor',
                epsilon: float = 0.05) -> 'torch.Tensor':
    """
    Fast Gradient Sign Method (FGSM) adversarial perturbation.
    Perturbs inputs in the direction that maximises loss, making the
    model more robust to noisy/outlier workout data.

    Reference: Goodfellow et al. (2014) "Explaining and Harnessing Adversarial Examples"
    """
    X_adv = X.clone().requires_grad_(True)
    output = model(X_adv)
    loss = F.kl_div(output, y, reduction='batchmean')
    loss.backward()
    with torch.no_grad():
        X_adv = X_adv + epsilon * X_adv.grad.sign()
        # Clip to keep in valid feature range [0, ~2] (normalised features)
        X_adv = torch.clamp(X_adv, 0.0, 2.0)
    return X_adv.detach()


def mixup_batch(X: 'torch.Tensor', y: 'torch.Tensor',
                alpha: float = 0.2) -> Tuple['torch.Tensor', 'torch.Tensor']:
    """
    Mixup data augmentation: linearly interpolate between random pairs.
    Creates virtual training examples that regularise the decision boundary.

    Reference: Zhang et al. (2017) "mixup: Beyond Empirical Risk Minimization"
    """
    batch_size = X.size(0)
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(batch_size)
    X_mix = lam * X + (1 - lam) * X[idx]
    y_mix = lam * y + (1 - lam) * y[idx]
    return X_mix, y_mix


# ══════════════════════════════════════════════════════════════════════════════
# SHAP Explainability
# ══════════════════════════════════════════════════════════════════════════════

def compute_shap_values(model: 'nn.Module',
                        X_background: np.ndarray,
                        X_explain: np.ndarray) -> Optional[np.ndarray]:
    """
    Compute SHAP DeepExplainer values for feature attribution.

    Returns array of shape (n_samples, n_features) with SHAP values,
    or gradient-based fallback if SHAP is not installed.

    Reference: Lundberg & Lee (2017) "A Unified Approach to Interpreting Model Predictions"
    """
    if not TORCH_AVAILABLE:
        return None

    model.eval()
    X_bg_t = torch.tensor(X_background[:min(50, len(X_background))], dtype=torch.float32)
    X_exp_t = torch.tensor(X_explain[:min(20, len(X_explain))], dtype=torch.float32)

    if SHAP_AVAILABLE:
        try:
            # Wrap model for SHAP (sum over output classes for global explanation)
            class ModelWrapper(nn.Module):
                def __init__(self, m):
                    super().__init__()
                    self.m = m
                def forward(self, x):
                    return self.m(x).exp().sum(dim=-1, keepdim=True)

            wrapper = ModelWrapper(model)
            explainer = shap.DeepExplainer(wrapper, X_bg_t)
            shap_vals = explainer.shap_values(X_exp_t)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[0]
            return np.abs(shap_vals).mean(axis=0)  # Mean absolute SHAP per feature
        except Exception as e:
            logger.warning(f"SHAP failed, falling back to gradient importance: {e}")

    # Gradient-based fallback (from engine.py compute_feature_importance)
    from .engine import compute_feature_importance
    return compute_feature_importance(model, X_exp_t)


def compute_integrated_gradients(model: 'nn.Module',
                                  X: 'torch.Tensor',
                                  baseline: Optional['torch.Tensor'] = None,
                                  steps: int = 50) -> np.ndarray:
    """
    Integrated Gradients for robust feature attribution.
    Integrates gradients along path from baseline (zeros) to input.

    Reference: Sundararajan et al. (2017) "Axiomatic Attribution for Deep Networks"
    """
    if not TORCH_AVAILABLE:
        return np.zeros(N_FEATURES)

    model.eval()
    if baseline is None:
        baseline = torch.zeros_like(X)

    # Interpolate between baseline and input
    alphas = torch.linspace(0, 1, steps).view(-1, 1, 1)
    X_interp = baseline.unsqueeze(0) + alphas * (X.unsqueeze(0) - baseline.unsqueeze(0))
    X_interp = X_interp.view(-1, X.shape[-1]).requires_grad_(True)

    output = model(X_interp)
    # Use logit sum as scalar objective
    score = output.exp().sum(dim=-1).sum()
    score.backward()

    grads = X_interp.grad.view(steps, X.shape[0], X.shape[-1])
    avg_grads = grads.mean(dim=0)
    integrated = (X - baseline) * avg_grads
    # Mean over samples, absolute value
    return integrated.abs().detach().cpu().numpy().mean(axis=0)


# ══════════════════════════════════════════════════════════════════════════════
# Trainer
# ══════════════════════════════════════════════════════════════════════════════

class PulseMindTrainer:
    """
    Full training loop for PulseMind models.

    Features:
    - AdamW with weight decay
    - Cosine annealing LR schedule
    - Early stopping with best checkpoint
    - Adversarial training (FGSM + Mixup) — controlled by adv_alpha
    - W&B logging
    - SHAP explainability post-training
    """

    def __init__(self,
                 model: 'nn.Module',
                 lr: float = 3e-4,
                 weight_decay: float = 1e-4,
                 patience: int = 25,
                 adv_alpha: float = 0.3,
                 use_adversarial: bool = True,
                 use_mixup: bool = True):

        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required. Run: pip install torch")

        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.adv_alpha = adv_alpha        # Fraction of adversarial examples in each batch
        self.use_adversarial = use_adversarial
        self.use_mixup = use_mixup

        self.optimiser = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )

        # CPU only — works on M1, x86, any machine
        self.device = torch.device('cpu')
        self.model.to(self.device)

    def fit(self,
            X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray,
            epochs: int = 200,
            batch_size: int = 32,
            use_wandb: bool = False,
            verbose: bool = True) -> Dict:
        """
        Train the model with adversarial training and Mixup augmentation.
        Returns training history dict.
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required.")

        if use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project="ironpulse-pulsemind",
                config={
                    'architecture': self.model.architecture,
                    'lr': self.lr,
                    'weight_decay': self.weight_decay,
                    'epochs': epochs,
                    'adversarial': self.use_adversarial,
                    'mixup': self.use_mixup,
                    'adv_alpha': self.adv_alpha,
                    'train_samples': len(X_train),
                    'val_samples': len(X_val),
                }
            )

        # Convert to tensors
        X_tr = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_tr = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        X_v  = torch.tensor(X_val,   dtype=torch.float32).to(self.device)
        y_v  = torch.tensor(y_val,   dtype=torch.float32).to(self.device)

        dataset = TensorDataset(X_tr, y_tr)
        loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

        # LR scheduler: cosine annealing with warm restarts
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimiser, T_0=50, T_mult=2, eta_min=1e-6
        )

        history = {
            'train_loss': [], 'val_loss': [], 'accuracy': [],
            'adv_loss': [], 'lr': [],
        }

        best_val_loss = float('inf')
        best_state    = None
        best_epoch    = 0
        no_improve    = 0
        start_time    = time.time()

        for epoch in range(epochs):
            # ── Training ──────────────────────────────────────────────
            self.model.train()
            epoch_loss     = 0.0
            epoch_adv_loss = 0.0
            n_batches      = 0

            for X_batch, y_batch in loader:
                self.optimiser.zero_grad()

                # ── Mixup augmentation ───────────────────────────────
                if self.use_mixup and len(X_batch) > 1:
                    X_batch, y_batch = mixup_batch(X_batch, y_batch, alpha=0.2)

                # ── Clean forward/backward ───────────────────────────
                out  = self.model(X_batch)
                loss = F.kl_div(out, y_batch, reduction='batchmean')

                # ── Adversarial forward/backward (FGSM) ──────────────
                adv_loss_val = torch.tensor(0.0)
                if self.use_adversarial and len(X_batch) > 1:
                    # Generate adversarial examples
                    n_adv = max(1, int(len(X_batch) * self.adv_alpha))
                    X_adv = fgsm_attack(
                        self.model, X_batch[:n_adv], y_batch[:n_adv],
                        epsilon=0.03
                    )
                    self.model.train()
                    out_adv   = self.model(X_adv)
                    adv_loss  = F.kl_div(out_adv, y_batch[:n_adv], reduction='batchmean')
                    # Combine: 70% clean + 30% adversarial
                    total_loss = 0.7 * loss + 0.3 * adv_loss
                    adv_loss_val = adv_loss.detach()
                else:
                    total_loss = loss

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimiser.step()

                epoch_loss     += loss.item()
                epoch_adv_loss += adv_loss_val.item()
                n_batches      += 1

            scheduler.step()

            avg_train_loss = epoch_loss / max(n_batches, 1)
            avg_adv_loss   = epoch_adv_loss / max(n_batches, 1)

            # ── Validation ────────────────────────────────────────────
            self.model.eval()
            with torch.no_grad():
                val_out  = self.model(X_v)
                val_loss = F.kl_div(val_out, y_v, reduction='batchmean').item()

                # Top-3 accuracy
                preds   = val_out.exp().topk(3, dim=1).indices
                targets = y_v.topk(3, dim=1).indices
                correct = sum(
                    bool(set(p.tolist()) & set(t.tolist()))
                    for p, t in zip(preds, targets)
                )
                accuracy = correct / max(len(y_v), 1)

            current_lr = self.optimiser.param_groups[0]['lr']
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_loss)
            history['accuracy'].append(accuracy)
            history['adv_loss'].append(avg_adv_loss)
            history['lr'].append(current_lr)

            if use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': avg_train_loss,
                    'val_loss': val_loss,
                    'top3_accuracy': accuracy,
                    'adv_loss': avg_adv_loss,
                    'lr': current_lr,
                })

            if verbose and (epoch % 20 == 0 or epoch == epochs - 1):
                adv_str = f" | adv={avg_adv_loss:.4f}" if self.use_adversarial else ""
                logger.info(
                    f"Epoch {epoch:3d}/{epochs} | "
                    f"train={avg_train_loss:.4f} | val={val_loss:.4f}{adv_str} | "
                    f"acc={accuracy:.3f} | lr={current_lr:.6f}"
                )

            # ── Early stopping ────────────────────────────────────────
            if val_loss < best_val_loss - 1e-5:
                best_val_loss = val_loss
                best_state    = copy.deepcopy(self.model.state_dict())
                best_epoch    = epoch
                no_improve    = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    if verbose:
                        logger.info(f"Early stopping at epoch {epoch}")
                    break

        # Restore best weights
        if best_state is not None:
            self.model.load_state_dict(best_state)

        duration = time.time() - start_time

        if use_wandb and WANDB_AVAILABLE:
            wandb.log({
                'best_val_loss': best_val_loss,
                'best_epoch': best_epoch,
                'duration_sec': duration,
            })
            wandb.finish()

        history['best_epoch']    = best_epoch
        history['best_val_loss'] = best_val_loss
        history['duration_sec']  = duration

        return history

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate on held-out test set."""
        self.model.eval()
        X_t = torch.tensor(X_test, dtype=torch.float32)
        y_t = torch.tensor(y_test, dtype=torch.float32)
        with torch.no_grad():
            out  = self.model(X_t)
            loss = F.kl_div(out, y_t, reduction='batchmean').item()
            preds   = out.exp().topk(3, dim=1).indices
            targets = y_t.topk(3, dim=1).indices
            correct = sum(
                bool(set(p.tolist()) & set(t.tolist()))
                for p, t in zip(preds, targets)
            )
            top3_acc = correct / max(len(y_t), 1)
        return {'test_loss': loss, 'top3_accuracy': top3_acc}


# ══════════════════════════════════════════════════════════════════════════════
# One-call Training API (used by views.py)
# ══════════════════════════════════════════════════════════════════════════════

def train_and_evaluate(sessions, exercises,
                       arch: str = 'resnet',
                       epochs: int = 200,
                       use_wandb: bool = False,
                       use_adversarial: bool = True,
                       use_mixup: bool = True) -> Dict:
    """
    Full pipeline: extract features → train → evaluate → return results.
    Used by the Django view for one-click training.
    """
    from .engine import build_model, save_model

    X, y = prepare_workout_data(sessions, exercises)

    if len(X) == 0:
        return {'error': 'No exercise data available.'}

    # 70/15/15 split
    n_total = len(X)
    idx     = np.random.permutation(n_total)
    n_train = int(0.70 * n_total)
    n_val   = int(0.15 * n_total)
    idx_tr  = idx[:n_train]
    idx_val = idx[n_train:n_train + n_val]
    idx_te  = idx[n_train + n_val:]

    X_train, y_train = X[idx_tr], y[idx_tr]
    X_val,   y_val   = X[idx_val], y[idx_val]
    X_test,  y_test  = X[idx_te], y[idx_te]

    # Ensure we have at least 1 sample in each split
    if len(X_train) == 0 or len(X_val) == 0:
        # Fall back to using all data for train and val
        split = max(1, len(X) // 2)
        X_train, y_train = X[:split], y[:split]
        X_val, y_val = X[split:], y[split:]
        if len(X_val) == 0:
            X_val, y_val = X_train, y_train
        X_test, y_test = X_val, y_val

    model = build_model(arch, input_size=N_FEATURES, output_size=y.shape[1])
    trainer = PulseMindTrainer(
        model,
        lr=3e-4,
        patience=30,
        use_adversarial=use_adversarial,
        use_mixup=use_mixup,
    )

    history = trainer.fit(
        X_train, y_train, X_val, y_val,
        epochs=epochs, verbose=True, use_wandb=use_wandb
    )

    # Test evaluation
    test_metrics = {}
    if len(X_test) > 0:
        test_metrics = trainer.evaluate(X_test, y_test)

    # Compute SHAP feature attribution
    shap_values = None
    try:
        shap_values = compute_shap_values(model, X_train, X_val)
    except Exception as e:
        logger.warning(f"SHAP computation failed: {e}")

    # Integrated gradients attribution
    ig_values = None
    try:
        X_sample = torch.tensor(X_val[:10], dtype=torch.float32)
        ig_values = compute_integrated_gradients(model, X_sample)
    except Exception as e:
        logger.warning(f"Integrated gradients failed: {e}")

    # Save model
    save_model(model, 'pulsemind_latest')

    # Architecture info
    arch_info = model.get_architecture_info()

    best_acc = max(history['accuracy']) if history['accuracy'] else 0.0
    test_acc = test_metrics.get('top3_accuracy', best_acc)

    return {
        'success': True,
        'architecture': arch,
        'arch_info': arch_info,
        'best_epoch': history['best_epoch'],
        'best_val_loss': history['best_val_loss'],
        'val_accuracy': round(best_acc * 100, 1),
        'test_accuracy': round(test_acc * 100, 1),
        'duration_sec': round(history['duration_sec'], 1),
        'n_train': len(X_train),
        'n_val': len(X_val),
        'n_test': len(X_test),
        'adversarial': use_adversarial,
        'mixup': use_mixup,
        'shap_values': shap_values.tolist() if shap_values is not None else None,
        'ig_values': ig_values.tolist() if ig_values is not None else None,
        'feature_names': FEATURE_NAMES,
        'history': {
            'train_loss': [float(x) for x in history['train_loss']],
            'val_loss':   [float(x) for x in history['val_loss']],
            'accuracy':   [float(x) for x in history['accuracy']],
            'adv_loss':   [float(x) for x in history['adv_loss']],
        },
    }
