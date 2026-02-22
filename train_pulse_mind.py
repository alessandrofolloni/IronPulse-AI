"""
IronPulse — PulseMind End-to-End Training Pipeline
====================================================
Supports multiple sophisticated architectures, W&B logging,
pre-training with synthetic data, and easy CLI invocation.

Quick start (easy training procedure):
---------------------------------------
    # 1. Basic training with the best architecture
    python train_pulse_mind.py

    # 2. Full W&B experiment tracking
    python train_pulse_mind.py --wandb

    # 3. Choose architecture: mlp | resnet | attention | ensemble
    python train_pulse_mind.py --arch resnet --wandb

    # 4. Quick smoke test (small epochs, no W&B)
    python train_pulse_mind.py --arch mlp --epochs 50 --quick

    # 5. Hyperparameter sweep across all architectures
    python train_pulse_mind.py --sweep

    # 6. Run regression task (predict next-session weight)
    python train_pulse_mind.py --task regression --arch mlp

Environment
-----------
    DJANGO_SETTINGS_MODULE=gymapp.settings  (auto-set)
    WANDB_PROJECT=IronPulse-AI              (default)
    WANDB_API_KEY=<your_key>               (from .env or env var)
"""

import os
import sys
import json
import logging
import argparse
import datetime
import numpy as np

# ── Initialise Django ──────────────────────────────────────────────────────────
sys.path.insert(0, os.getcwd())
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'gymapp.settings')

import django
django.setup()

from core.models import WorkoutSession, Exercise, AIModelMetadata
from core.ai.engine import build_model
from core.ai.trainer import PulseMindTrainer, prepare_workout_data, prepare_regression_data, FEATURE_NAMES, N_FEATURES

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('pulsemind')

WEIGHT_DIR = 'core/ai/weights'


# ══════════════════════════════════════════════════════════════════════════════
# Pre-training with Synthetic Data
# ══════════════════════════════════════════════════════════════════════════════

def generate_synthetic_data(n_samples=2000, n_features=N_FEATURES, n_classes=10, seed=42):
    """
    Generate structured synthetic gym data for pre-training.

    Synthetic rules:
      - High compound ratio → more likely to predict compound exercises
      - High volume → likely to recommend rest or lower-intensity day
      - Short rest → more recovery-focused exercise selection

    This gives the model a beneficial starting point ("transfer learning lite")
    before it sees real user data.
    """
    np.random.seed(seed)
    X = np.random.randn(n_samples, n_features).astype(np.float32)

    # Structured label generation (not pure random)
    y = np.zeros((n_samples, n_classes), dtype=np.float32)
    for i in range(n_samples):
        volume_feat  = X[i, 0]     # session_volume signal
        compound_feat = X[i, 9]    # compound ratio signal
        if compound_feat > 0.5:
            label = np.random.randint(0, n_classes // 3)       # compound exercises
        elif volume_feat > 1.0:
            label = np.random.randint(n_classes // 3, 2 * n_classes // 3)  # accessory
        else:
            label = np.random.randint(2 * n_classes // 3, n_classes)       # recovery
        y[i, label] = 1.0

    return X, y


def pretrain_model(model, n_classes, use_wandb=False):
    """
    Pre-train model on synthetic data.
    Converges faster and avoids cold-start when real data is sparse.
    """
    print("\n🔥 Phase 1: Pre-training on synthetic data...")
    X_syn, y_syn = generate_synthetic_data(n_samples=3000, n_classes=n_classes)

    n = X_syn.shape[0]
    split = int(0.85 * n)
    X_tr, y_tr = X_syn[:split], y_syn[:split]
    X_vl, y_vl = X_syn[split:], y_syn[split:]

    config = {
        'phase': 'pretrain',
        'data': 'synthetic',
        'samples': n,
    }
    trainer = PulseMindTrainer(
        model,
        optimiser    = 'adam',
        lr           = 5e-3,
        scheduler    = 'cosine',
        mode         = 'classification',
        batch_size   = 128,
        patience     = 25,
        use_wandb    = use_wandb,
        wandb_config = config,
    )
    history = trainer.fit(X_tr, y_tr, X_vl, y_vl, epochs=100, verbose=True, log_every=10)
    best_val = min(history.get('val_loss', [999]))
    print(f"✅ Pre-training complete. Best val_loss={best_val:.4f}")
    return history


# ══════════════════════════════════════════════════════════════════════════════
# Classification Pipeline (Exercise Recommendation)
# ══════════════════════════════════════════════════════════════════════════════

def train_classification(args):
    arch       = args.arch
    epochs     = args.epochs
    use_wandb  = args.wandb
    do_pretrain = args.pretrain

    print(f"\n🚀 PulseMind Classification Pipeline")
    print(f"   Architecture  : {arch.upper()}")
    print(f"   Epochs        : {epochs}")
    print(f"   W&B logging   : {'ON' if use_wandb else 'OFF'}")
    print(f"   Pre-training  : {'ON' if do_pretrain else 'OFF'}")
    print("─" * 52)

    # ── Data ──────────────────────────────────────────────────
    sessions  = WorkoutSession.objects.all()
    exercises = Exercise.objects.all()
    n_classes = exercises.count() or 10

    if sessions.count() < 5:
        print("⚠️  Insufficient real data — using synthetic data for full training.")
        X, y = generate_synthetic_data(n_samples=1500, n_classes=n_classes)
    else:
        print(f"📦 Extracting features from {sessions.count()} sessions...")
        X, y = prepare_workout_data(sessions, exercises)

    n = X.shape[0]
    print(f"   Dataset shape : X={X.shape}  y={y.shape}")

    # ── Train/Val/Test split 80/10/10 ──────────────────────────
    perm = np.random.permutation(n)
    t1, t2 = int(0.8 * n), int(0.9 * n)
    X_train, X_val, X_test = X[perm[:t1]], X[perm[t1:t2]], X[perm[t2:]]
    y_train, y_val, y_test = y[perm[:t1]], y[perm[t1:t2]], y[perm[t2:]]

    # ── Hyperparameter configs per architecture ─────────────────
    arch_configs = {
        'mlp': {
            'hidden_layers': [512, 256, 128, 64],
            'activation':    'leaky_relu',
            'dropout_rate':  0.3,
            'lr':            1e-3,
        },
        'resnet': {
            'hidden_size':   256,
            'n_blocks':      5,
            'activation':    'elu',
            'dropout_rate':  0.2,
            'lr':            8e-4,
        },
        'attention': {
            'd_model':       64,
            'n_layers':      4,
            'dropout_rate':  0.1,
            'lr':            5e-4,
        },
        'ensemble': {
            'sub_archs':     ['mlp', 'resnet'],
            'hidden_layers': [256, 128, 64],
            'hidden_size':   128,
            'n_blocks':      3,
            'dropout_rate':  0.2,
            'lr':            8e-4,
        },
    }
    cfg = arch_configs.get(arch, arch_configs['mlp'])

    wandb_config = {
        'architecture':  arch,
        'epochs':        epochs,
        'n_features':    X.shape[1],
        'n_classes':     n_classes,
        'n_train':       len(X_train),
        **cfg,
    }

    # ── Build model ────────────────────────────────────────────
    model = build_model(arch, input_size=X.shape[1], output_size=n_classes,
                         mode='classification', **cfg)

    # ── Optional pre-training ──────────────────────────────────
    if do_pretrain:
        pretrain_model(model, n_classes, use_wandb=use_wandb)
        # Fine-tune with lower LR
        cfg['lr'] = cfg['lr'] * 0.3

    # ── Fine-tuning / full training ────────────────────────────
    print(f"\n🧠 Phase {'2' if do_pretrain else '1'}: Training on {'real' if sessions.count() >= 5 else 'synthetic'} data...")
    trainer = PulseMindTrainer(
        model,
        optimiser    = 'adam',
        lr           = cfg.get('lr', 1e-3),
        scheduler    = 'cosine',
        mode         = 'classification',
        batch_size   = 64,
        patience     = 40,
        use_wandb    = use_wandb,
        wandb_config = wandb_config,
    )
    history = trainer.fit(X_train, y_train, X_val, y_val, epochs=epochs, verbose=True, log_every=20)

    # ── Test evaluation ────────────────────────────────────────
    test_metrics = trainer.evaluate(X_test, y_test)
    accuracy = test_metrics.get('accuracy', 0.0)
    print(f"\n📊 Test Results:")
    print(f"   Accuracy  : {accuracy * 100:.2f}%")
    print(f"   Test Loss : {test_metrics['loss']:.6f}")

    # ── Explainability ─────────────────────────────────────────
    print(f"\n🔍 Feature Importances ({arch.upper()}):")
    importances = model.feature_importance(X_test[:min(50, len(X_test))])
    feat_table = sorted(zip(FEATURE_NAMES, importances), key=lambda x: -x[1])
    for name, imp in feat_table[:10]:
        bar = '█' * int(imp * 200)
        print(f"   {name:<25s} {imp:.4f}  {bar}")

    # ── Export weights ─────────────────────────────────────────
    params = model.get_parameters()
    os.makedirs(WEIGHT_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    weight_file = os.path.join(WEIGHT_DIR, f'pulsemind_{arch}_{ts}.json')
    latest_file = os.path.join(WEIGHT_DIR, 'pulsemind_latest.json')

    with open(weight_file, 'w') as f:
        json.dump(params, f)
    with open(latest_file, 'w') as f:
        json.dump(params, f)

    print(f"\n💾 Weights saved: {weight_file}")

    # ── Update DB metadata ─────────────────────────────────────
    metadata, _ = AIModelMetadata.objects.get_or_create(pk=1)
    metadata.model_name             = f"PulseMind-{arch.capitalize()}"
    metadata.last_trained           = datetime.datetime.now()
    metadata.accuracy               = round(accuracy * 100, 2)
    metadata.total_training_samples = len(X_train)
    metadata.weights_info           = params
    metadata.save()

    print(f"✅ DB metadata updated. Model: {metadata.model_name}  Acc: {metadata.accuracy}%")
    return history, accuracy


# ══════════════════════════════════════════════════════════════════════════════
# Regression Pipeline (Strength Prediction)
# ══════════════════════════════════════════════════════════════════════════════

def train_regression(args):
    arch      = args.arch
    epochs    = args.epochs
    use_wandb = args.wandb

    print(f"\n🏋️ PulseMind Regression Pipeline (Strength Prediction)")
    print(f"   Architecture : {arch.upper()}")
    print("─" * 52)

    exercises = list(Exercise.objects.all())
    if not exercises:
        print("❌ No exercises in DB. Run: python seed.py first.")
        return

    # Use first exercise as demo (in a real app, iterate over all)
    ex = exercises[0]
    print(f"   Training for : {ex.name}")

    X, y = prepare_regression_data(ex.pk)
    print(f"   Dataset shape: X={X.shape}  y={y.shape}")

    n = X.shape[0]
    split = int(0.8 * n)
    X_tr, y_tr = X[:split], y[:split]
    X_vl, y_vl = X[split:], y[split:]

    model = build_model(
        arch if arch != 'attention' else 'mlp',   # attention needs seq, use mlp for regression
        input_size  = X.shape[1],
        output_size = 1,
        mode        = 'regression',
        hidden_layers = [128, 64, 32],
        dropout_rate  = 0.2,
    )

    trainer = PulseMindTrainer(
        model,
        optimiser    = 'adam',
        lr           = 1e-3,
        scheduler    = 'cosine',
        mode         = 'regression',
        loss_fn      = 'huber',
        batch_size   = 32,
        patience     = 30,
        use_wandb    = use_wandb,
        wandb_config = {'task': 'regression', 'exercise': ex.name},
    )

    history = trainer.fit(X_tr, y_tr, X_vl, y_vl, epochs=epochs, verbose=True, log_every=20)
    test_metrics = trainer.evaluate(X_vl, y_vl)
    print(f"\n📊 Regression Results: RMSE={test_metrics['rmse']:.2f} kg  MAE={test_metrics['mae']:.2f} kg")
    return history


# ══════════════════════════════════════════════════════════════════════════════
# Architecture Sweep
# ══════════════════════════════════════════════════════════════════════════════

def run_sweep(args):
    """Train all architectures and compare performance."""
    print("\n🔬 Architecture Sweep Mode")
    print("=" * 52)
    results = {}
    for arch in ['mlp', 'resnet', 'attention', 'ensemble']:
        print(f"\n▶️  Architecture: {arch.upper()}")
        args.arch     = arch
        args.epochs   = min(args.epochs, 100)
        args.pretrain = False
        try:
            _, acc = train_classification(args)
            results[arch] = acc
        except Exception as e:
            logger.error(f"Sweep failed for {arch}: {e}")
            results[arch] = 0.0

    print("\n🏆 Sweep Summary:")
    print("─" * 35)
    for arch, acc in sorted(results.items(), key=lambda x: -x[1]):
        bar = '█' * int(acc * 20)
        print(f"  {arch:<12} {acc*100:6.2f}%  {bar}")
    best = max(results, key=results.get)
    print(f"\n✅ Best architecture: {best.upper()} ({results[best]*100:.2f}%)")


# ══════════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description='PulseMind AI Training Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_pulse_mind.py
  python train_pulse_mind.py --arch resnet --epochs 300 --wandb
  python train_pulse_mind.py --arch attention --pretrain --wandb
  python train_pulse_mind.py --sweep
  python train_pulse_mind.py --task regression --arch mlp
        """
    )
    p.add_argument('--arch',     choices=['mlp', 'resnet', 'attention', 'ensemble'],
                   default='resnet', help='Neural network architecture (default: resnet)')
    p.add_argument('--epochs',   type=int, default=200,
                   help='Training epochs (default: 200)')
    p.add_argument('--task',     choices=['classification', 'regression'],
                   default='classification', help='Training task (default: classification)')
    p.add_argument('--wandb',    action='store_true',
                   help='Enable Weights & Biases logging')
    p.add_argument('--pretrain', action='store_true',
                   help='Pre-train on synthetic data before real data')
    p.add_argument('--sweep',    action='store_true',
                   help='Run all architectures and compare performance')
    p.add_argument('--quick',    action='store_true',
                   help='Quick smoke test: 50 epochs, no W&B')
    p.add_argument('--seed',     type=int, default=42,
                   help='Random seed for reproducibility (default: 42)')
    return p.parse_args()


def main():
    args = parse_args()

    # Quick mode overrides
    if args.quick:
        args.epochs  = 50
        args.wandb   = False
        args.pretrain = False

    # Reproducibility
    np.random.seed(args.seed)

    print("=" * 52)
    print("  ⚡ IronPulse — PulseMind AI Training")
    print("=" * 52)

    if args.sweep:
        run_sweep(args)
    elif args.task == 'regression':
        train_regression(args)
    else:
        train_classification(args)


if __name__ == '__main__':
    main()
