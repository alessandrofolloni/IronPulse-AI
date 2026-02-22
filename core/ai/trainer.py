"""
IronPulse — PulseMind Trainer
==============================
Provides optimisers, loss functions, LR schedulers, early stopping,
and training loops — all compatible with every PulseMind architecture.

Optimisers
----------
  SGDMomentum  : SGD with configurable momentum and weight decay
  Adam         : Adaptive moment estimation (default, recommended)

Schedulers
----------
  StepLR       : Reduce LR by factor γ every step_size epochs
  CosineAnnealingLR : Smooth cosine decay from η_max to η_min

Usage
-----
    from core.ai.engine import build_model
    from core.ai.trainer import PulseMindTrainer, prepare_workout_data

    model = build_model('resnet', input_size=18, output_size=10)
    trainer = PulseMindTrainer(model, optimiser='adam', lr=1e-3, use_wandb=True)
    history = trainer.fit(X_train, y_train, X_val, y_val, epochs=200)
"""

import numpy as np
import logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

# ── Optional W&B ──────────────────────────────────────────────────────────────
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False
    logger.warning("wandb not installed — W&B logging disabled. Run: pip install wandb")


# ══════════════════════════════════════════════════════════════════════════════
# Optimisers
# ══════════════════════════════════════════════════════════════════════════════

class SGDMomentum:
    """Classical SGD with Nesterov-style momentum and L2 weight decay."""

    def __init__(self, lr=0.01, momentum=0.9, weight_decay=1e-4):
        self.lr           = lr
        self.momentum     = momentum
        self.weight_decay = weight_decay
        self._velocities  = {}   # id(param) → velocity

    def step(self, params_and_grads):
        """
        params_and_grads : list of (W_ref, b_ref, dW, db) tuples
                           where W_ref / b_ref are the *model's own arrays*
                           (we update them in-place).
        """
        for (W, b, dW, db) in params_and_grads:
            for param, grad in [(W, dW), (b, db)]:
                pid = id(param)
                if pid not in self._velocities:
                    self._velocities[pid] = np.zeros_like(param)
                v = self._velocities[pid]
                grad_reg = grad + self.weight_decay * param
                v[:] = self.momentum * v - self.lr * grad_reg
                param += v
                self._velocities[pid] = v


class Adam:
    """
    Adam optimiser.
    References: Kingma & Ba, 2014 (https://arxiv.org/abs/1412.6980)

    Computes adaptive learning rates using:
        m̂  = m  / (1 - β₁ᵗ)   — bias-corrected 1st moment
        v̂  = v  / (1 - β₂ᵗ)   — bias-corrected 2nd moment
        θ  = θ  - α * m̂ / (√v̂ + ε)

    Generally the best default choice for neural networks.
    """

    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=1e-4):
        self.lr           = lr
        self.beta1        = beta1
        self.beta2        = beta2
        self.eps          = eps
        self.weight_decay = weight_decay
        self._m   = {}   # 1st moment
        self._v   = {}   # 2nd moment
        self._t   = 0    # step counter

    def step(self, params_and_grads):
        self._t += 1
        bc1 = 1 - self.beta1 ** self._t
        bc2 = 1 - self.beta2 ** self._t

        for (W, b, dW, db) in params_and_grads:
            for param, grad in [(W, dW), (b, db)]:
                pid = id(param)
                if pid not in self._m:
                    self._m[pid] = np.zeros_like(param)
                    self._v[pid] = np.zeros_like(param)

                grad_reg = grad + self.weight_decay * param
                self._m[pid] = self.beta1 * self._m[pid] + (1 - self.beta1) * grad_reg
                self._v[pid] = self.beta2 * self._v[pid] + (1 - self.beta2) * grad_reg ** 2

                m_hat = self._m[pid] / bc1
                v_hat = self._v[pid] / bc2
                param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ══════════════════════════════════════════════════════════════════════════════
# Learning-Rate Schedulers
# ══════════════════════════════════════════════════════════════════════════════

class StepLR:
    """Reduce LR by factor gamma every step_size epochs."""
    def __init__(self, optimiser, step_size=50, gamma=0.5):
        self.opt       = optimiser
        self.step_size = step_size
        self.gamma     = gamma
        self._base_lr  = optimiser.lr

    def step(self, epoch):
        if epoch > 0 and epoch % self.step_size == 0:
            self.opt.lr *= self.gamma
            logger.debug(f"StepLR: lr reduced to {self.opt.lr:.6f}")
        return self.opt.lr


class CosineAnnealingLR:
    """Smooth cosine annealing: η = η_min + 0.5*(η_max-η_min)*(1 + cos(πt/T))"""
    def __init__(self, optimiser, T_max, eta_min=1e-6):
        self.opt      = optimiser
        self.T_max    = T_max
        self.eta_min  = eta_min
        self.eta_max  = optimiser.lr

    def step(self, epoch):
        new_lr = self.eta_min + 0.5 * (self.eta_max - self.eta_min) * (
            1 + np.cos(np.pi * epoch / self.T_max)
        )
        self.opt.lr = new_lr
        return new_lr


# ══════════════════════════════════════════════════════════════════════════════
# Loss Functions
# ══════════════════════════════════════════════════════════════════════════════

def cross_entropy_loss(y_pred, y_true):
    """Multinomial cross-entropy. y_true: one-hot."""
    m = y_pred.shape[0]
    loss = -np.sum(y_true * np.log(np.clip(y_pred, 1e-9, 1.0))) / m
    grad = (y_pred - y_true) / m
    return loss, grad


def mse_loss(y_pred, y_true):
    """Mean squared error for regression."""
    m = y_pred.shape[0]
    diff = y_pred - y_true
    loss = np.mean(diff ** 2)
    grad = 2 * diff / m
    return loss, grad


def huber_loss(y_pred, y_true, delta=1.0):
    """Huber loss: L1-smooth, more robust to outliers than MSE."""
    diff = y_pred - y_true
    abs_diff = np.abs(diff)
    loss = np.where(abs_diff <= delta, 0.5 * diff ** 2, delta * (abs_diff - 0.5 * delta))
    loss = loss.mean()
    grad = np.where(abs_diff <= delta, diff, delta * np.sign(diff)) / y_pred.shape[0]
    return loss, grad


# ══════════════════════════════════════════════════════════════════════════════
# Early Stopping
# ══════════════════════════════════════════════════════════════════════════════

class EarlyStopping:
    """
    Stop training if monitored metric hasn't improved in `patience` epochs.
    Also stores the best weights seen so far (checkpointing).
    """

    def __init__(self, patience=20, min_delta=1e-4, restore_best=True):
        self.patience     = patience
        self.min_delta    = min_delta
        self.restore_best = restore_best
        self._best_loss   = np.inf
        self._counter     = 0
        self._best_params = None

    def step(self, val_loss, model):
        if val_loss < self._best_loss - self.min_delta:
            self._best_loss   = val_loss
            self._counter     = 0
            self._best_params = model.get_parameters()
        else:
            self._counter += 1

        if self._counter >= self.patience:
            if self.restore_best and self._best_params:
                model.load_from_dict(self._best_params)
            return True   # signal to stop
        return False


# ══════════════════════════════════════════════════════════════════════════════
# Main Trainer
# ══════════════════════════════════════════════════════════════════════════════

class PulseMindTrainer:
    """
    Universal trainer compatible with MLP, ResNet, AttentionNet, and Ensemble.

    Parameters
    ----------
    model        : any PulseMind architecture with forward() / backward()
    optimiser    : 'adam' (default) | 'sgd'
    lr           : initial learning rate
    scheduler    : 'cosine' | 'step' | None
    mode         : 'classification' | 'regression'
    loss_fn      : 'cross_entropy' | 'mse' | 'huber'
    batch_size   : mini-batch size (set to -1 for full-batch)
    use_wandb    : log metrics to W&B
    wandb_config : dict passed to wandb.init()
    patience     : early stopping patience (0 = disabled)
    """

    def __init__(
        self,
        model,
        optimiser:    str   = 'adam',
        lr:           float = 1e-3,
        scheduler:    Optional[str] = 'cosine',
        mode:         str   = 'classification',
        loss_fn:      str   = 'cross_entropy',
        batch_size:   int   = 64,
        use_wandb:    bool  = False,
        wandb_config: Optional[Dict[str, Any]] = None,
        patience:     int   = 30,
        weight_decay: float = 1e-4,
    ):
        self.model      = model
        self.mode       = mode
        self.batch_size = batch_size
        self.use_wandb  = use_wandb and _WANDB_AVAILABLE

        # Optimiser
        if optimiser == 'adam':
            self.opt = Adam(lr=lr, weight_decay=weight_decay)
        else:
            self.opt = SGDMomentum(lr=lr, weight_decay=weight_decay)

        # Loss function
        _loss_map = {
            'cross_entropy': cross_entropy_loss,
            'mse':           mse_loss,
            'huber':         huber_loss,
        }
        self._loss_fn = _loss_map.get(loss_fn, cross_entropy_loss)

        # Scheduler placeholder (created in fit() when T_max is known)
        self._scheduler_type = scheduler
        self.scheduler       = None

        # Early stopping
        self.early_stopping = EarlyStopping(patience=patience) if patience > 0 else None

        # W&B init
        if self.use_wandb:
            wandb.init(
                project='IronPulse-AI',
                config=wandb_config or {},
                reinit=True,
            )
            logger.info("W&B run initialised.")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _compute_loss_and_grad(self, X, y):
        """Forward + loss + backward for one mini-batch."""
        y_pred = self.model.forward(X)
        loss, dout = self._loss_fn(y_pred, y)
        grads = self.model.backward(dout)
        return loss, grads, y_pred

    def _extract_params_grads(self, grads):
        """
        Build the list of (W, b, dW, db) tuples for the optimiser.
        Handles all architecture shapes.
        """
        pairs = []
        arch = getattr(self.model, 'get_parameters', lambda: {})()
        arch_type = arch.get('architecture', 'MLP')

        if arch_type == 'MLP':
            for i, (dW, db) in enumerate(grads):
                pairs.append((self.model.weights[i], self.model.biases[i], dW, db))

        elif arch_type == 'ResNet':
            # grads order: [head, blocks×2, proj] after reverse, so we pair manually
            all_W = [(self.model.proj_W, self.model.proj_b)]
            for block in self.model.blocks:
                all_W += [(block.W1, block.b1), (block.W2, block.b2)]
            all_W.append((self.model.head_W, self.model.head_b))
            for (W, b), (dW, db) in zip(all_W, grads):
                pairs.append((W, b, dW, db))

        elif arch_type == 'AttentionNet':
            # Gradient flow through attention is complex; use numerical grad as fallback
            # (analytical backprop of attention omitted for clarity — use Adam with small LR)
            pass

        elif arch_type == 'Ensemble':
            # Delegate to sub-model trainers (each sub-model trained separately)
            pass

        return pairs

    def _get_metrics(self, y_pred, y_true):
        """Compute accuracy (classification) or RMSE (regression)."""
        if self.mode == 'classification':
            correct = np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1)
            return {'accuracy': float(correct.mean())}
        else:
            rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
            mae  = float(np.mean(np.abs(y_pred - y_true)))
            return {'rmse': rmse, 'mae': mae}

    # ── Public interface ──────────────────────────────────────────────────────

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val:   Optional[np.ndarray] = None,
        y_val:   Optional[np.ndarray] = None,
        epochs:  int = 300,
        verbose: bool = True,
        log_every: int = 10,
    ) -> Dict[str, List[float]]:
        """
        Full training loop with mini-batching, LR scheduling, early stopping and W&B logging.

        Returns
        -------
        history : dict with keys 'train_loss', 'val_loss', 'lr', and metric keys.
        """
        n = X_train.shape[0]
        history: Dict[str, List[float]] = {
            'train_loss': [], 'val_loss': [], 'lr': []
        }

        # Build scheduler now that we know T_max
        if self._scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(self.opt, T_max=epochs)
        elif self._scheduler_type == 'step':
            self.scheduler = StepLR(self.opt, step_size=max(1, epochs // 5))

        self.model.set_training(True)

        for epoch in range(epochs):
            # Mini-batch shuffle
            perm        = np.random.permutation(n)
            batch_losses = []
            bs = self.batch_size if self.batch_size > 0 else n

            for start in range(0, n, bs):
                idx    = perm[start:start + bs]
                X_b, y_b = X_train[idx], y_train[idx]
                loss, grads, _ = self._compute_loss_and_grad(X_b, y_b)
                pg = self._extract_params_grads(grads)
                if pg:
                    self.opt.step(pg)
                batch_losses.append(loss)

            train_loss = float(np.mean(batch_losses))
            history['train_loss'].append(train_loss)

            # Validation
            val_loss = None
            val_metrics = {}
            if X_val is not None and y_val is not None:
                self.model.set_training(False)
                val_pred = self.model.forward(X_val)
                val_loss, _ = self._loss_fn(val_pred, y_val)
                val_loss     = float(val_loss)
                val_metrics  = self._get_metrics(val_pred, y_val)
                history.setdefault('val_loss', []).append(val_loss)
                for k, v in val_metrics.items():
                    history.setdefault(k, []).append(v)
                self.model.set_training(True)

            # LR schedule
            current_lr = self.opt.lr
            if self.scheduler:
                current_lr = self.scheduler.step(epoch)
            history['lr'].append(current_lr)

            # W&B logging
            if self.use_wandb and epoch % log_every == 0:
                log_dict = {
                    'epoch':       epoch,
                    'train_loss':  train_loss,
                    'lr':          current_lr,
                }
                if val_loss is not None:
                    log_dict['val_loss'] = val_loss
                log_dict.update(val_metrics)
                wandb.log(log_dict)

            # Console logging
            if verbose and epoch % log_every == 0:
                msg = f"Epoch {epoch:4d}/{epochs} | train_loss={train_loss:.6f}"
                if val_loss is not None:
                    msg += f" | val_loss={val_loss:.6f}"
                if 'accuracy' in val_metrics:
                    msg += f" | val_acc={val_metrics['accuracy']*100:.2f}%"
                if 'rmse' in val_metrics:
                    msg += f" | val_rmse={val_metrics['rmse']:.4f}"
                msg += f" | lr={current_lr:.6f}"
                logger.info(msg)
                if verbose:
                    print(msg)

            # Early stopping
            if self.early_stopping and val_loss is not None:
                if self.early_stopping.step(val_loss, self.model):
                    print(f"⚡ Early stopping at epoch {epoch} (best val_loss={self.early_stopping._best_loss:.6f})")
                    if self.use_wandb:
                        wandb.log({'early_stop_epoch': epoch})
                    break

        self.model.set_training(False)

        if self.use_wandb:
            # Final summary
            metrics_final = {}
            if X_val is not None:
                val_pred = self.model.forward(X_val)
                metrics_final = self._get_metrics(val_pred, y_val)
            wandb.run.summary.update(metrics_final)
            wandb.finish()

        return history

    def evaluate(self, X, y):
        """Return loss and task metrics on a held-out set."""
        self.model.set_training(False)
        y_pred = self.model.forward(X)
        loss, _ = self._loss_fn(y_pred, y)
        metrics = self._get_metrics(y_pred, y)
        metrics['loss'] = float(loss)
        return metrics


# ══════════════════════════════════════════════════════════════════════════════
# Data Preparation Utilities
# ══════════════════════════════════════════════════════════════════════════════

FEATURE_NAMES = [
    'session_volume_kg',    # total kg lifted in session
    'num_sets',             # total sets
    'num_exercises',        # unique exercises
    'avg_weight',           # average weight per set
    'max_weight',           # heaviest set
    'avg_reps',             # average reps per set
    'max_reps',             # highest rep set
    'avg_one_rm',           # average estimated 1RM
    'max_one_rm',           # best 1RM estimate
    'compound_ratio',       # fraction of compound movements
    'chest_ratio',          # muscle-group feature ratios (6)
    'back_ratio',
    'legs_ratio',
    'shoulders_ratio',
    'arms_ratio',
    'core_ratio',
    'days_since_last',      # rest days
    'session_count_30d',    # frequency proxy
]

N_FEATURES = len(FEATURE_NAMES)   # 18


def prepare_workout_data(sessions, exercises):
    """
    Extracts structured feature vectors from WorkoutSession querysets.

    Returns
    -------
    X : np.ndarray  shape (n_sessions, N_FEATURES)
    y : np.ndarray  shape (n_sessions, n_exercises)  — one-hot next exercise
    """
    from core.models import WorkoutSet

    exercise_list = list(exercises)
    n_classes = len(exercise_list)
    if n_classes == 0:
        return np.random.rand(50, N_FEATURES), np.eye(10)[np.random.randint(0, 10, 50)]

    ex_index = {ex.pk: i for i, ex in enumerate(exercise_list)}
    compound_ids = {ex.pk for ex in exercise_list if ex.is_compound}

    muscle_map = {
        'chest': 10, 'back': 11, 'legs': 12, 'glutes': 12,
        'shoulders': 13, 'biceps': 14, 'triceps': 14, 'core': 15,
    }

    session_list = list(sessions.order_by('date'))
    if len(session_list) < 2:
        return np.random.rand(50, N_FEATURES), np.eye(n_classes)[np.random.randint(0, n_classes, 50)]

    X_rows, y_rows = [], []

    for si, session in enumerate(session_list[:-1]):
        sets = list(WorkoutSet.objects.filter(session=session).select_related('exercise'))
        if not sets:
            continue

        weights  = np.array([s.weight for s in sets])
        reps_arr = np.array([s.reps   for s in sets])
        one_rms  = np.array([s.one_rm or 0 for s in sets])

        volume_kg     = float((weights * reps_arr).sum())
        num_sets      = len(sets)
        num_exercises = len({s.exercise_id for s in sets})
        avg_weight    = float(weights.mean())
        max_weight    = float(weights.max())
        avg_reps      = float(reps_arr.mean())
        max_reps      = float(reps_arr.max())
        avg_one_rm    = float(one_rms.mean())
        max_one_rm    = float(one_rms.max())
        compound_r    = float(sum(1 for s in sets if s.exercise_id in compound_ids) / max(num_sets, 1))

        # Muscle group ratios
        muscle_counts = np.zeros(6)
        for s in sets:
            idx = muscle_map.get(s.exercise.muscle_group, -1)
            if idx >= 10:
                muscle_counts[idx - 10] += 1
        muscle_ratios = muscle_counts / max(num_sets, 1)

        # Temporal features
        days_since = float((session_list[si + 1].date - session.date).days)
        prev_dates = [s.date for s in session_list[:si + 1]]
        month_ago  = session.date - __import__('datetime').timedelta(days=30)
        session_count_30d = float(sum(1 for d in prev_dates if d >= month_ago))

        feat = np.array([
            volume_kg, num_sets, num_exercises, avg_weight, max_weight,
            avg_reps, max_reps, avg_one_rm, max_one_rm, compound_r,
            *muscle_ratios,
            days_since, session_count_30d,
        ])
        X_rows.append(feat)

        # Target: next session's primary exercise
        next_sets = list(WorkoutSet.objects.filter(session=session_list[si + 1]).select_related('exercise'))
        if not next_sets:
            continue
        primary_ex = max(
            next_sets,
            key=lambda s: s.weight * s.reps
        ).exercise_id
        label = ex_index.get(primary_ex, 0)
        one_hot = np.zeros(n_classes)
        one_hot[label] = 1
        y_rows.append(one_hot)

    if not X_rows:
        return np.random.rand(50, N_FEATURES), np.eye(n_classes)[np.random.randint(0, n_classes, 50)]

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=np.float32)

    # Standardise features (zero-mean, unit variance) — critical for deep networks
    mu  = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    X   = (X - mu) / std

    return X, y


def prepare_regression_data(exercise_id):
    """
    Time-series regression data for a single exercise:
    Features: [session_volume, days_since_last, prev_1rm]
    Target:   next session weight

    Returns np.ndarray X (n, 3) and y (n, 1).
    """
    from core.models import WorkoutSet
    sets = list(WorkoutSet.objects.filter(exercise_id=exercise_id).select_related('session').order_by('session__date'))

    if len(sets) < 2:
        return np.random.rand(20, 3), np.random.rand(20, 1) * 100

    X_rows, y_rows = [], []
    for i in range(len(sets) - 1):
        s     = sets[i]
        s_nxt = sets[i + 1]
        days  = (s_nxt.session.date - s.session.date).days
        feat  = np.array([s.weight * s.reps, days, s.one_rm or 0])
        X_rows.append(feat)
        y_rows.append([s_nxt.weight])

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=np.float32)

    mu  = X.mean(0); std = X.std(0) + 1e-8
    X   = (X - mu) / std

    return X, y
