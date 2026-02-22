"""
Tests for the PulseMind AI engine and trainer.
Run with: pytest core/tests/test_ai.py -v

Tests cover:
- All 4 architectures (MLP, ResNet, AttentionNet, Ensemble)
- Forward pass shape correctness
- Backward pass / gradient flow
- Serialisation round-trip (save → load → inference)
- Feature importance (explainability)
- Data preparation utilities
- Optimisers (Adam, SGD-Momentum)
- Loss functions
- Early stopping
"""
import numpy as np
import pytest

# ── Import engine & trainer ───────────────────────────────────────
from core.ai.engine import (
    PulseMindMLP, PulseMindResNet, PulseMindAttentionNet, PulseMindEnsemble,
    PulseMindClassifier, PulseMindRegressor, build_model,
    softmax, relu, leaky_relu, elu,
    LayerNorm, Dropout,
    N_FEATURES, FEATURE_NAMES,
)
from core.ai.trainer import (
    PulseMindTrainer,
    Adam, SGDMomentum,
    StepLR, CosineAnnealingLR,
    EarlyStopping,
    cross_entropy_loss, mse_loss, huber_loss,
)

BATCH = 16
N_IN  = 18
N_OUT = 10

np.random.seed(0)


# ── Fixtures ─────────────────────────────────────────────────────

@pytest.fixture
def Xy_clf():
    X = np.random.randn(BATCH, N_IN).astype(np.float32)
    y = np.zeros((BATCH, N_OUT), dtype=np.float32)
    y[range(BATCH), np.random.randint(0, N_OUT, BATCH)] = 1.0
    return X, y


@pytest.fixture
def Xy_reg():
    X = np.random.randn(BATCH, 3).astype(np.float32)
    y = np.random.rand(BATCH, 1).astype(np.float32) * 100
    return X, y


# ══════════════════════════════════════════════════════════════════
# Activation functions
# ══════════════════════════════════════════════════════════════════

class TestActivations:
    def test_relu_non_negative(self):
        x = np.array([-2.0, 0.0, 3.0])
        assert np.all(relu(x) >= 0)

    def test_softmax_sums_to_one(self):
        z = np.random.randn(5, N_OUT).astype(np.float32)
        out = softmax(z)
        np.testing.assert_allclose(out.sum(axis=1), np.ones(5), atol=1e-5)

    def test_softmax_no_nan(self):
        # Large values shouldn't produce NaN (numerical stability)
        z = np.array([[1000.0, 1001.0, 999.0]])
        out = softmax(z)
        assert not np.any(np.isnan(out))

    def test_leaky_relu_negative(self):
        x = np.array([-1.0, -2.0, 1.0])
        out = leaky_relu(x, alpha=0.01)
        assert out[0] == pytest.approx(-0.01, abs=1e-5)
        assert out[2] == 1.0


# ══════════════════════════════════════════════════════════════════
# MLP Architecture
# ══════════════════════════════════════════════════════════════════

class TestMLP:
    def test_forward_shape(self, Xy_clf):
        X, y = Xy_clf
        model = PulseMindMLP([N_IN, 64, 32, N_OUT], mode='classification')
        out = model.forward(X)
        assert out.shape == (BATCH, N_OUT)

    def test_forward_probabilities(self, Xy_clf):
        X, y = Xy_clf
        model = PulseMindMLP([N_IN, 64, N_OUT], mode='classification')
        out = model.forward(X)
        np.testing.assert_allclose(out.sum(axis=1), np.ones(BATCH), atol=1e-5)
        assert np.all(out >= 0)

    def test_regression_output(self, Xy_reg):
        X, y = Xy_reg
        model = PulseMindMLP([3, 32, 1], mode='regression')
        out = model.forward(X)
        assert out.shape == (BATCH, 1)

    def test_backward_returns_grads(self, Xy_clf):
        X, y = Xy_clf
        model = PulseMindMLP([N_IN, 32, N_OUT], mode='classification')
        out = model.forward(X)
        _, dout = cross_entropy_loss(out, y)
        grads = model.backward(dout)
        assert len(grads) == len(model.weights)
        for (dW, db), W, b in zip(grads, model.weights, model.biases):
            assert dW.shape == W.shape
            assert db.shape == b.shape

    def test_get_and_load_parameters(self, Xy_clf):
        X, _ = Xy_clf
        model = PulseMindMLP([N_IN, 32, N_OUT])
        out1 = model.forward(X)
        params = model.get_parameters()
        assert params['architecture'] == 'MLP'

        # New model, load, should reproduce same output
        model2 = PulseMindMLP([N_IN, 32, N_OUT])
        model2.load_from_dict(params)
        out2 = model2.forward(X)
        np.testing.assert_allclose(out1, out2, atol=1e-5)

    def test_feature_importance_sums_to_one(self, Xy_clf):
        X, _ = Xy_clf
        model = PulseMindMLP([N_IN, 32, N_OUT])
        imp = model.feature_importance(X)
        assert imp.shape == (N_IN,)
        assert abs(imp.sum() - 1.0) < 1e-4

    def test_dropout_disabled_at_eval(self, Xy_clf):
        X, _ = Xy_clf
        model = PulseMindMLP([N_IN, 64, N_OUT], dropout_rate=0.5)
        model.set_training(False)
        out1 = model.forward(X)
        out2 = model.forward(X)
        np.testing.assert_allclose(out1, out2)

    def test_factory_classifier(self, Xy_clf):
        X, _ = Xy_clf
        model = PulseMindClassifier(N_IN, [64, 32], N_OUT)
        out = model.forward(X)
        assert out.shape == (BATCH, N_OUT)

    def test_factory_regressor(self, Xy_reg):
        X, _ = Xy_reg
        model = PulseMindRegressor(3, [32, 16])
        out = model.forward(X)
        assert out.shape == (BATCH, 1)


# ══════════════════════════════════════════════════════════════════
# ResNet Architecture
# ══════════════════════════════════════════════════════════════════

class TestResNet:
    def test_forward_shape(self, Xy_clf):
        X, _ = Xy_clf
        model = PulseMindResNet(N_IN, hidden_size=32, n_blocks=2, output_size=N_OUT)
        out = model.forward(X)
        assert out.shape == (BATCH, N_OUT)

    def test_forward_probabilities(self, Xy_clf):
        X, _ = Xy_clf
        model = PulseMindResNet(N_IN, 32, 2, N_OUT, mode='classification')
        out = model.forward(X)
        np.testing.assert_allclose(out.sum(axis=1), np.ones(BATCH), atol=1e-5)

    def test_serialise_round_trip(self, Xy_clf):
        X, _ = Xy_clf
        model = PulseMindResNet(N_IN, 32, 2, N_OUT)
        out1 = model.forward(X)
        params = model.get_parameters()
        assert params['architecture'] == 'ResNet'
        model2 = PulseMindResNet(N_IN, 32, 2, N_OUT)
        model2.load_from_dict(params)
        out2 = model2.forward(X)
        np.testing.assert_allclose(out1, out2, atol=1e-5)

    def test_build_model_factory(self, Xy_clf):
        X, _ = Xy_clf
        model = build_model('resnet', N_IN, N_OUT, mode='classification',
                             hidden_size=32, n_blocks=2)
        out = model.forward(X)
        assert out.shape == (BATCH, N_OUT)


# ══════════════════════════════════════════════════════════════════
# AttentionNet Architecture
# ══════════════════════════════════════════════════════════════════

class TestAttentionNet:
    def test_forward_shape(self, Xy_clf):
        X, _ = Xy_clf
        model = PulseMindAttentionNet(N_IN, d_model=16, n_layers=2, output_size=N_OUT)
        out = model.forward(X)
        assert out.shape == (BATCH, N_OUT)

    def test_attention_weights_stored(self, Xy_clf):
        X, _ = Xy_clf
        model = PulseMindAttentionNet(N_IN, d_model=16, n_layers=1, output_size=N_OUT)
        model.forward(X)
        # After forward, attention weights should be stored for interpretability
        assert model.attn_layers[0].attn_weights is not None

    def test_feature_importance_shape(self, Xy_clf):
        X, _ = Xy_clf
        model = PulseMindAttentionNet(N_IN, d_model=16, n_layers=2, output_size=N_OUT)
        imp = model.feature_importance(X)
        assert imp.shape == (N_IN,)

    def test_serialise_round_trip(self, Xy_clf):
        X, _ = Xy_clf
        model = PulseMindAttentionNet(N_IN, d_model=8, n_layers=1, output_size=N_OUT)
        out1 = model.forward(X)
        params = model.get_parameters()
        assert params['architecture'] == 'AttentionNet'
        model2 = PulseMindAttentionNet(N_IN, d_model=8, n_layers=1, output_size=N_OUT)
        model2.load_from_dict(params)
        out2 = model2.forward(X)
        np.testing.assert_allclose(out1, out2, atol=1e-5)


# ══════════════════════════════════════════════════════════════════
# Ensemble Architecture
# ══════════════════════════════════════════════════════════════════

class TestEnsemble:
    def test_forward_shape(self, Xy_clf):
        X, _ = Xy_clf
        m1  = build_model('mlp',    N_IN, N_OUT, hidden_layers=[32])
        m2  = build_model('resnet', N_IN, N_OUT, hidden_size=32, n_blocks=1)
        ens = PulseMindEnsemble([m1, m2])
        out = ens.forward(X)
        assert out.shape == (BATCH, N_OUT)

    def test_ensemble_probabilities_sum_to_one(self, Xy_clf):
        X, _ = Xy_clf
        m1 = build_model('mlp', N_IN, N_OUT, hidden_layers=[32])
        m2 = build_model('mlp', N_IN, N_OUT, hidden_layers=[16])
        ens = PulseMindEnsemble([m1, m2])
        out = ens.forward(X)
        np.testing.assert_allclose(out.sum(axis=1), np.ones(BATCH), atol=1e-5)

    def test_feature_importance(self, Xy_clf):
        X, _ = Xy_clf
        m1 = build_model('mlp', N_IN, N_OUT, hidden_layers=[32])
        m2 = build_model('mlp', N_IN, N_OUT, hidden_layers=[16])
        ens = PulseMindEnsemble([m1, m2])
        imp = ens.feature_importance(X)
        assert imp.shape == (N_IN,)


# ══════════════════════════════════════════════════════════════════
# Loss Functions
# ══════════════════════════════════════════════════════════════════

class TestLossFunctions:
    def test_cross_entropy_positive(self, Xy_clf):
        X, y = Xy_clf
        model = PulseMindMLP([N_IN, 32, N_OUT])
        y_pred = model.forward(X)
        loss, grad = cross_entropy_loss(y_pred, y)
        assert loss > 0
        assert grad.shape == y_pred.shape

    def test_mse_positive(self, Xy_reg):
        X, y = Xy_reg
        model = PulseMindMLP([3, 16, 1], mode='regression')
        y_pred = model.forward(X)
        loss, grad = mse_loss(y_pred, y)
        assert loss >= 0
        assert grad.shape == y_pred.shape

    def test_huber_bounded(self, Xy_reg):
        X, y = Xy_reg
        model = PulseMindMLP([3, 16, 1], mode='regression')
        y_pred = model.forward(X)
        loss, _ = huber_loss(y_pred, y)
        assert loss >= 0

    def test_cross_entropy_perfect_prediction_near_zero(self):
        y_pred = np.array([[0.999, 0.001]])
        y_true = np.array([[1.0,   0.0]])
        loss, _ = cross_entropy_loss(y_pred, y_true)
        assert loss < 0.1


# ══════════════════════════════════════════════════════════════════
# Optimisers
# ══════════════════════════════════════════════════════════════════

class TestOptimisers:
    def _step(self, opt, model, X, y):
        out = model.forward(X)
        _, dout = cross_entropy_loss(out, y)
        grads = model.backward(dout)
        pg = [(model.weights[i], model.biases[i], dW, db)
              for i, (dW, db) in enumerate(grads)]
        opt.step(pg)

    def test_adam_reduces_loss(self, Xy_clf):
        X, y = Xy_clf
        model = PulseMindMLP([N_IN, 32, N_OUT])
        opt   = Adam(lr=0.01)
        losses = []
        for _ in range(50):
            out = model.forward(X)
            loss, _ = cross_entropy_loss(out, y)
            losses.append(loss)
            self._step(opt, model, X, y)
        assert losses[-1] < losses[0], "Adam should reduce loss"

    def test_sgd_momentum_reduces_loss(self, Xy_clf):
        X, y = Xy_clf
        model = PulseMindMLP([N_IN, 32, N_OUT])
        opt   = SGDMomentum(lr=0.01)
        losses = []
        for _ in range(50):
            out = model.forward(X)
            loss, _ = cross_entropy_loss(out, y)
            losses.append(loss)
            self._step(opt, model, X, y)
        assert losses[-1] < losses[0]


# ══════════════════════════════════════════════════════════════════
# LR Schedulers
# ══════════════════════════════════════════════════════════════════

class TestSchedulers:
    def test_cosine_decreases_then_increases(self):
        opt = Adam(lr=0.1)
        sch = CosineAnnealingLR(opt, T_max=100, eta_min=0.001)
        lrs = [sch.step(e) for e in range(101)]
        # LR at t=0 should equal eta_max
        assert lrs[0] == pytest.approx(0.1, abs=1e-4)
        # LR at t=100 should equal eta_min
        assert lrs[100] == pytest.approx(0.001, abs=1e-4)
        # LR should be monotonically decreasing for first half
        assert all(lrs[i] >= lrs[i+1] for i in range(50))

    def test_step_lr_halves_lr(self):
        opt = Adam(lr=0.1)
        sch = StepLR(opt, step_size=5, gamma=0.5)
        sch.step(5)   # trigger first decay
        assert opt.lr == pytest.approx(0.05, abs=1e-6)
        sch.step(10)  # trigger second decay
        assert opt.lr == pytest.approx(0.025, abs=1e-6)


# ══════════════════════════════════════════════════════════════════
# Early Stopping
# ══════════════════════════════════════════════════════════════════

class TestEarlyStopping:
    def test_does_not_stop_on_improvement(self, Xy_clf):
        X, _ = Xy_clf
        model = PulseMindMLP([N_IN, 32, N_OUT])
        es = EarlyStopping(patience=5)
        for i in range(6):
            stop = es.step(1.0 - i * 0.1, model)   # improving each time
        assert not stop

    def test_stops_on_plateau(self, Xy_clf):
        X, _ = Xy_clf
        model = PulseMindMLP([N_IN, 32, N_OUT])
        es = EarlyStopping(patience=3)
        for _ in range(4):
            stop = es.step(1.0, model)   # no improvement → count up
        assert stop


# ══════════════════════════════════════════════════════════════════
# Trainer Integration
# ══════════════════════════════════════════════════════════════════

class TestTrainer:
    def test_fit_returns_history(self, Xy_clf):
        X, y = Xy_clf
        model   = PulseMindMLP([N_IN, 32, N_OUT])
        trainer = PulseMindTrainer(model, lr=0.01, batch_size=8, patience=0)
        history = trainer.fit(X, y, epochs=5, verbose=False)
        assert 'train_loss' in history
        assert len(history['train_loss']) == 5

    def test_fit_reduces_loss(self, Xy_clf):
        X, y = Xy_clf
        model   = PulseMindMLP([N_IN, 64, 32, N_OUT])
        trainer = PulseMindTrainer(model, lr=5e-3, batch_size=-1, patience=0)
        history = trainer.fit(X, y, epochs=100, verbose=False)
        losses = history['train_loss']
        assert losses[-1] < losses[0], "Loss should decrease over 100 epochs"

    def test_evaluate_returns_metrics(self, Xy_clf):
        X, y = Xy_clf
        model = PulseMindMLP([N_IN, 32, N_OUT])
        trainer = PulseMindTrainer(model, patience=0)
        trainer.fit(X, y, epochs=5, verbose=False)
        metrics = trainer.evaluate(X, y)
        assert 'loss' in metrics
        assert 'accuracy' in metrics
        assert 0 <= metrics['accuracy'] <= 1

    def test_regression_trainer(self, Xy_reg):
        X, y = Xy_reg
        model   = PulseMindMLP([3, 32, 1], mode='regression')
        trainer = PulseMindTrainer(model, mode='regression', loss_fn='huber',
                                   lr=1e-3, patience=0)
        history = trainer.fit(X, y, epochs=20, verbose=False)
        assert 'train_loss' in history


# ══════════════════════════════════════════════════════════════════
# Build Model Factory
# ══════════════════════════════════════════════════════════════════

class TestBuildModel:
    @pytest.mark.parametrize("arch", ['mlp', 'resnet', 'attention'])
    def test_all_archs_produce_correct_shape(self, arch, Xy_clf):
        X, _ = Xy_clf
        kwargs = {}
        if arch == 'mlp':
            kwargs['hidden_layers'] = [32, 16]
        elif arch == 'resnet':
            kwargs['hidden_size'] = 32
            kwargs['n_blocks'] = 1
        elif arch == 'attention':
            kwargs['d_model'] = 8
            kwargs['n_layers'] = 1
        model = build_model(arch, N_IN, N_OUT, mode='classification', **kwargs)
        out = model.forward(X)
        assert out.shape == (BATCH, N_OUT)

    def test_unknown_arch_raises(self):
        with pytest.raises(ValueError, match="Unknown architecture"):
            build_model('transformer', N_IN, N_OUT)


# ══════════════════════════════════════════════════════════════════
# Constants / Metadata
# ══════════════════════════════════════════════════════════════════

class TestConstants:
    def test_feature_names_length(self):
        assert len(FEATURE_NAMES) == N_FEATURES

    def test_n_features_is_18(self):
        assert N_FEATURES == 18
