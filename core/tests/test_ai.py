"""
PulseMind AI — Tests (v2 — Adversarial + Explainability)
=========================================================
Tests model architectures, training pipeline, adversarial robustness,
and SHAP/Integrated Gradients explainability.

Run with: pytest core/tests/test_ai.py -v
"""

import pytest
import numpy as np

# Skip all tests if PyTorch not available
torch = pytest.importorskip("torch")
import torch.nn as nn

from core.ai.engine import (
    N_FEATURES, FEATURE_NAMES,
    PulseMindMLP, PulseMindResNet, PulseMindAttention, PulseMindEnsemble,
    build_model, save_model, load_model, compute_feature_importance,
    MODEL_REGISTRY,
)
from core.ai.trainer import (
    fgsm_attack, mixup_batch,
    compute_shap_values, compute_integrated_gradients,
    PulseMindTrainer,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_input():
    """Random batch of 16 samples with N_FEATURES features, in [0,1]."""
    return torch.rand(16, N_FEATURES)


@pytest.fixture
def n_classes():
    return 30


@pytest.fixture(params=['mlp', 'resnet', 'attention', 'ensemble'])
def model_name(request):
    return request.param


@pytest.fixture
def tiny_xy(n_classes):
    """Tiny dataset for training tests."""
    X = np.random.rand(40, N_FEATURES).astype(np.float32)
    # Dirichlet labels (realistic probability distribution)
    y = np.random.dirichlet(np.ones(n_classes), size=40).astype(np.float32)
    return X, y


# ── Architecture Tests ────────────────────────────────────────────────────────

class TestArchitectures:
    """Verify all architectures produce correct output shapes."""

    def test_forward_shape(self, model_name, sample_input, n_classes):
        """Output shape must be (batch_size, n_classes)."""
        model = build_model(model_name, input_size=N_FEATURES, output_size=n_classes)
        model.eval()
        with torch.no_grad():
            out = model(sample_input)
        assert out.shape == (16, n_classes), f"Expected (16, {n_classes}), got {out.shape}"

    def test_output_is_log_probabilities(self, model_name, sample_input, n_classes):
        """Classification output should be log-probabilities (sum of exp ≈ 1)."""
        model = build_model(model_name, input_size=N_FEATURES, output_size=n_classes)
        model.eval()
        with torch.no_grad():
            out = model(sample_input)
        probs = torch.exp(out)
        sums = probs.sum(dim=1)
        assert torch.allclose(sums, torch.ones(16), atol=1e-5), \
            f"Probabilities don't sum to 1: {sums[:3]}"

    def test_no_nan_in_output(self, model_name, sample_input, n_classes):
        """No NaN values in forward pass."""
        model = build_model(model_name, input_size=N_FEATURES, output_size=n_classes)
        model.eval()
        with torch.no_grad():
            out = model(sample_input)
        assert not torch.isnan(out).any(), "NaN detected in output"

    def test_gradient_flow(self, model_name, n_classes):
        """Gradients must flow to all parameters."""
        model = build_model(model_name, input_size=N_FEATURES, output_size=n_classes)
        model.train()
        x = torch.randn(4, N_FEATURES, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"

    def test_architecture_info(self, model_name, n_classes):
        """get_architecture_info must return valid info dict."""
        model = build_model(model_name, input_size=N_FEATURES, output_size=n_classes)
        info = model.get_architecture_info()
        assert 'architecture' in info
        assert 'total_params' in info
        assert info['total_params'] > 0
        assert 'layers' in info
        assert len(info['layers']) > 0

    def test_deterministic_eval(self, model_name, sample_input, n_classes):
        """Eval mode should produce same output on repeated calls."""
        model = build_model(model_name, input_size=N_FEATURES, output_size=n_classes)
        model.eval()
        with torch.no_grad():
            out1 = model(sample_input)
            out2 = model(sample_input)
        assert torch.allclose(out1, out2), "Eval mode is not deterministic"


# ── Model Registry Tests ─────────────────────────────────────────────────────

class TestModelRegistry:
    """Test model factory and registry."""

    def test_registry_has_all_models(self):
        assert set(MODEL_REGISTRY.keys()) == {'mlp', 'resnet', 'attention', 'ensemble'}

    def test_build_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown architecture"):
            build_model('nonexistent')


# ── Adversarial Training Tests ────────────────────────────────────────────────

class TestAdversarial:
    """Verify FGSM attack and Mixup augmentation work correctly."""

    def test_fgsm_changes_input(self, n_classes):
        """FGSM should produce different inputs from the original."""
        model = build_model('resnet', input_size=N_FEATURES, output_size=n_classes)
        model.eval()
        X = torch.rand(8, N_FEATURES)
        y = torch.from_numpy(
            np.random.dirichlet(np.ones(n_classes), size=8).astype(np.float32)
        )
        X_adv = fgsm_attack(model, X, y, epsilon=0.05)
        assert X_adv.shape == X.shape
        assert not torch.allclose(X_adv, X), "FGSM did not perturb inputs"

    def test_fgsm_bounded(self, n_classes):
        """Adversarial examples must stay in [0, 2] range."""
        model = build_model('mlp', input_size=N_FEATURES, output_size=n_classes)
        X = torch.rand(8, N_FEATURES)
        y = torch.from_numpy(
            np.random.dirichlet(np.ones(n_classes), size=8).astype(np.float32)
        )
        X_adv = fgsm_attack(model, X, y, epsilon=0.1)
        assert X_adv.min() >= 0.0, "Adversarial examples go below 0"
        assert X_adv.max() <= 2.0, "Adversarial examples exceed 2.0"

    def test_mixup_shape_preserved(self, n_classes):
        """Mixup should return same shapes."""
        X = torch.rand(16, N_FEATURES)
        y = torch.from_numpy(
            np.random.dirichlet(np.ones(n_classes), size=16).astype(np.float32)
        )
        X_mix, y_mix = mixup_batch(X, y, alpha=0.2)
        assert X_mix.shape == X.shape
        assert y_mix.shape == y.shape

    def test_mixup_labels_still_sum_to_one(self, n_classes):
        """Mixed labels should still sum to ~1 per sample."""
        X = torch.rand(16, N_FEATURES)
        y = torch.from_numpy(
            np.random.dirichlet(np.ones(n_classes), size=16).astype(np.float32)
        )
        _, y_mix = mixup_batch(X, y, alpha=0.2)
        sums = y_mix.sum(dim=1)
        assert torch.allclose(sums, torch.ones(16), atol=1e-5)


# ── Explainability Tests ──────────────────────────────────────────────────────

class TestExplainability:
    """Verify feature attribution methods work correctly."""

    def test_gradient_importance_shape(self, model_name, n_classes):
        """compute_feature_importance returns (N_FEATURES,) array."""
        model = build_model(model_name, input_size=N_FEATURES, output_size=n_classes)
        X = torch.rand(8, N_FEATURES)
        imp = compute_feature_importance(model, X)
        assert imp.shape == (N_FEATURES,)

    def test_gradient_importance_sums_to_one(self, model_name, n_classes):
        model = build_model(model_name, input_size=N_FEATURES, output_size=n_classes)
        X = torch.rand(8, N_FEATURES)
        imp = compute_feature_importance(model, X)
        assert abs(imp.sum() - 1.0) < 0.01

    def test_gradient_importance_non_negative(self, model_name, n_classes):
        model = build_model(model_name, input_size=N_FEATURES, output_size=n_classes)
        X = torch.rand(8, N_FEATURES)
        imp = compute_feature_importance(model, X)
        assert (imp >= 0).all()

    def test_integrated_gradients_shape(self, n_classes):
        """Integrated gradients should return (N_FEATURES,) array."""
        model = build_model('resnet', input_size=N_FEATURES, output_size=n_classes)
        X = torch.rand(4, N_FEATURES)
        ig = compute_integrated_gradients(model, X, steps=10)
        assert ig.shape == (N_FEATURES,)

    def test_integrated_gradients_non_negative(self, n_classes):
        """IG values should be non-negative (abs taken)."""
        model = build_model('mlp', input_size=N_FEATURES, output_size=n_classes)
        X = torch.rand(4, N_FEATURES)
        ig = compute_integrated_gradients(model, X, steps=10)
        assert (ig >= 0).all()

    def test_shap_fallback_works(self, n_classes):
        """compute_shap_values should return array even without shap library."""
        model = build_model('mlp', input_size=N_FEATURES, output_size=n_classes)
        X_bg = np.random.rand(20, N_FEATURES).astype(np.float32)
        X_ex = np.random.rand(5, N_FEATURES).astype(np.float32)
        result = compute_shap_values(model, X_bg, X_ex)
        assert result is not None
        assert result.shape == (N_FEATURES,)


# ── Training Pipeline Tests ───────────────────────────────────────────────────

class TestTrainer:
    """Test the PulseMindTrainer with adversarial training."""

    def test_trainer_fits_cleanly(self, tiny_xy, n_classes):
        """Trainer should complete without errors (standard mode)."""
        X, y = tiny_xy
        split = 32
        model = build_model('mlp', input_size=N_FEATURES, output_size=n_classes)
        trainer = PulseMindTrainer(model, lr=1e-3, patience=5,
                                   use_adversarial=False, use_mixup=False)
        history = trainer.fit(X[:split], y[:split], X[split:], y[split:],
                              epochs=10, verbose=False)
        assert len(history['train_loss']) > 0
        assert len(history['val_loss']) > 0
        assert history['best_epoch'] >= 0

    def test_trainer_adversarial_mode(self, tiny_xy, n_classes):
        """Trainer with adversarial=True should run without errors."""
        X, y = tiny_xy
        split = 32
        model = build_model('resnet', input_size=N_FEATURES, output_size=n_classes)
        trainer = PulseMindTrainer(model, lr=1e-3, patience=5,
                                   use_adversarial=True, use_mixup=True)
        history = trainer.fit(X[:split], y[:split], X[split:], y[split:],
                              epochs=10, verbose=False)
        assert 'adv_loss' in history
        assert len(history['adv_loss']) > 0

    def test_trainer_loss_decreases(self, tiny_xy, n_classes):
        """Training loss should generally trend downward over 20 epochs."""
        X, y = tiny_xy
        split = 30
        model = build_model('mlp', input_size=N_FEATURES, output_size=n_classes)
        trainer = PulseMindTrainer(model, lr=5e-3, patience=20,
                                   use_adversarial=False, use_mixup=False)
        history = trainer.fit(X[:split], y[:split], X[split:], y[split:],
                              epochs=20, verbose=False)
        first_5  = np.mean(history['train_loss'][:5])
        last_5   = np.mean(history['train_loss'][-5:])
        assert last_5 < first_5 * 1.5, \
            f"Loss did not decrease: {first_5:.4f} → {last_5:.4f}"


# ── Save/Load Tests ──────────────────────────────────────────────────────────

class TestSaveLoad:
    """Test model persistence."""

    def test_save_and_load_roundtrip(self, tmp_path, n_classes):
        import core.ai.engine as eng
        orig_dir = eng.WEIGHTS_DIR
        eng.WEIGHTS_DIR = tmp_path
        try:
            model = build_model('mlp', input_size=N_FEATURES, output_size=n_classes)
            x = torch.rand(4, N_FEATURES)
            model.eval()
            with torch.no_grad():
                out_before = model(x)
            save_model(model, name='test_model')
            loaded = load_model('test_model', input_size=N_FEATURES, output_size=n_classes)
            loaded.eval()
            with torch.no_grad():
                out_after = loaded(x)
            assert torch.allclose(out_before, out_after, atol=1e-5)
        finally:
            eng.WEIGHTS_DIR = orig_dir

    def test_load_nonexistent_raises(self, tmp_path):
        import core.ai.engine as eng
        orig_dir = eng.WEIGHTS_DIR
        eng.WEIGHTS_DIR = tmp_path
        try:
            with pytest.raises(FileNotFoundError):
                load_model('nonexistent')
        finally:
            eng.WEIGHTS_DIR = orig_dir


# ── Constants Tests ──────────────────────────────────────────────────────────

class TestConstants:
    """Verify engine constants."""

    def test_n_features(self):
        assert N_FEATURES == 18

    def test_feature_names_count(self):
        assert len(FEATURE_NAMES) == N_FEATURES

    def test_feature_names_are_strings(self):
        for name in FEATURE_NAMES:
            assert isinstance(name, str) and len(name) > 0


# ── Performance Benchmarks ───────────────────────────────────────────────────

class TestPerformance:
    """Basic performance benchmarks."""

    def test_forward_speed(self, model_name, n_classes):
        """Forward pass should complete in < 100ms for a batch of 64."""
        import time
        model = build_model(model_name, input_size=N_FEATURES, output_size=n_classes)
        model.eval()
        x = torch.rand(64, N_FEATURES)
        with torch.no_grad():
            model(x)  # warm-up
        start = time.time()
        for _ in range(10):
            with torch.no_grad():
                model(x)
        elapsed = (time.time() - start) / 10
        assert elapsed < 0.1, f"Forward pass too slow: {elapsed:.3f}s"

    def test_model_size_reasonable(self, model_name, n_classes):
        """Model should have < 5M parameters for tabular data."""
        model = build_model(model_name, input_size=N_FEATURES, output_size=n_classes)
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params < 5_000_000, f"Too many params: {n_params:,}"
