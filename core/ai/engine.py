"""
IronPulse — PulseMind AI Engine
================================
From-scratch neural network library using NumPy only.
Supports multiple sophisticated architectures logged to W&B.

Architectures available:
  - PulseMindMLP          : Deep Multi-Layer Perceptron (baseline)
  - PulseMindResNet       : Residual Network (skip connections, avoids vanishing gradient)
  - PulseMindAttentionNet : Self-Attention + FFN (transformer-inspired, explainable)
  - PulseMindEnsemble     : Ensemble of multiple models via soft voting

All models implement:
  - forward(X)            : inference pass
  - get_parameters()      : serialise weights → dict
  - load_from_dict(d)     : restore weights from dict
  - feature_importance(X) : gradient-based saliency for explainability
"""

import numpy as np
import json
import logging

logger = logging.getLogger(__name__)

# ── Activations ───────────────────────────────────────────────────────────────

def relu(z):
    return np.maximum(0, z)

def relu_deriv(z):
    return (z > 0).astype(float)

def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)

def leaky_relu_deriv(z, alpha=0.01):
    return np.where(z > 0, 1.0, alpha)

def elu(z, alpha=1.0):
    return np.where(z > 0, z, alpha * (np.exp(np.clip(z, -500, 0)) - 1))

def elu_deriv(z, alpha=1.0):
    return np.where(z > 0, 1.0, alpha * np.exp(np.clip(z, -500, 0)))

def softmax(z):
    e = np.exp(z - np.max(z, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

ACTIVATIONS = {
    'relu':       (relu,       relu_deriv),
    'leaky_relu': (leaky_relu, leaky_relu_deriv),
    'elu':        (elu,        elu_deriv),
}


# ── Weight Initialisation ─────────────────────────────────────────────────────

def he_init(fan_in, fan_out):
    """He (Kaiming) initialisation — optimal for ReLU family."""
    return np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)

def xavier_init(fan_in, fan_out):
    """Glorot/Xavier initialisation — balanced for tanh / sigmoid."""
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, (fan_in, fan_out))


# ── Layer Normalisation (simplified) ─────────────────────────────────────────

class LayerNorm:
    """
    Normalise across feature dimension (μ=0, σ=1) with learnable scale/shift.
    Improves training stability, especially for deep or attention networks.
    """
    def __init__(self, size, eps=1e-6):
        self.gamma = np.ones((1, size))
        self.beta  = np.zeros((1, size))
        self.eps   = eps
        self._cache = {}

    def forward(self, x):
        mu  = x.mean(axis=1, keepdims=True)
        var = x.var(axis=1,  keepdims=True)
        x_norm = (x - mu) / np.sqrt(var + self.eps)
        self._cache = {'x': x, 'mu': mu, 'var': var, 'x_norm': x_norm}
        return self.gamma * x_norm + self.beta

    def backward(self, dout):
        x_norm = self._cache['x_norm']
        var    = self._cache['var']
        x      = self._cache['x']
        mu     = self._cache['mu']
        m      = x.shape[1]
        dgamma = np.sum(dout * x_norm, axis=0, keepdims=True)
        dbeta  = np.sum(dout, axis=0, keepdims=True)
        dx_norm = dout * self.gamma
        dvar = np.sum(dx_norm * (x - mu) * -0.5 * (var + self.eps) ** -1.5, axis=1, keepdims=True)
        dmu  = np.sum(dx_norm * -1 / np.sqrt(var + self.eps), axis=1, keepdims=True)
        dx   = dx_norm / np.sqrt(var + self.eps) + dvar * 2 * (x - mu) / m + dmu / m
        self.gamma -= 0.001 * dgamma   # micro-update (trainer handles LR)
        self.beta  -= 0.001 * dbeta
        return dx

    def get_parameters(self):
        return {'gamma': self.gamma.tolist(), 'beta': self.beta.tolist()}

    def load_from_dict(self, d):
        self.gamma = np.array(d['gamma'])
        self.beta  = np.array(d['beta'])


# ── Dropout ───────────────────────────────────────────────────────────────────

class Dropout:
    def __init__(self, rate=0.3):
        self.rate = rate
        self._mask = None
        self.training = True

    def forward(self, x):
        if not self.training or self.rate == 0:
            return x
        self._mask = (np.random.rand(*x.shape) > self.rate) / (1 - self.rate)
        return x * self._mask

    def backward(self, dout):
        if self._mask is None:
            return dout
        return dout * self._mask


# ══════════════════════════════════════════════════════════════════════════════
# Architecture 1: PulseMindMLP — Deep Multi-Layer Perceptron
# ══════════════════════════════════════════════════════════════════════════════

class PulseMindMLP:
    """
    Deep MLP with He init, configurable activation, optional BatchNorm & Dropout.
    Acts as both classifier (softmax) and regressor (linear output).

    Parameters
    ----------
    layer_sizes   : list[int]  full layer dimensions [input, h1, …, output]
    activation    : str        'relu' | 'leaky_relu' | 'elu'
    dropout_rate  : float      0 = disabled
    mode          : str        'classification' | 'regression'
    """

    def __init__(self, layer_sizes, activation='relu', dropout_rate=0.0, mode='classification'):
        self.layer_sizes   = layer_sizes
        self.activation    = activation
        self.dropout_rate  = dropout_rate
        self.mode          = mode
        self.act_fn, self.act_deriv = ACTIVATIONS[activation]
        self.training = True

        self.weights   = []
        self.biases    = []
        self.dropouts  = []
        self.layer_norms = []
        self.activations_cache = []

        for i in range(len(layer_sizes) - 1):
            self.weights.append(he_init(layer_sizes[i], layer_sizes[i + 1]))
            self.biases.append(np.zeros((1, layer_sizes[i + 1])))
            self.dropouts.append(Dropout(dropout_rate))
            self.layer_norms.append(LayerNorm(layer_sizes[i + 1]))

    def set_training(self, flag: bool):
        self.training = flag
        for d in self.dropouts:
            d.training = flag

    def forward(self, X):
        self.activations_cache = [X]
        self.z_cache = []
        curr = X
        n_layers = len(self.weights)

        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = curr @ W + b
            self.z_cache.append(z)

            if i < n_layers - 1:
                # Hidden layer: activation → LayerNorm → Dropout
                a = self.act_fn(z)
                a = self.layer_norms[i].forward(a)
                a = self.dropouts[i].forward(a)
            else:
                # Output layer
                if self.mode == 'classification':
                    a = softmax(z)
                else:
                    a = z   # linear output for regression

            self.activations_cache.append(a)
            curr = a

        return curr

    def backward(self, dout):
        """Returns list of (dW, db) tuples from last → first layer."""
        n = len(self.weights)
        grads = []
        curr_d = dout

        for i in reversed(range(n)):
            if i < n - 1:
                curr_d = self.dropouts[i].backward(curr_d)
                curr_d = self.layer_norms[i].backward(curr_d)
                curr_d = curr_d * self.act_deriv(self.z_cache[i])

            dW = self.activations_cache[i].T @ curr_d
            db = curr_d.sum(axis=0, keepdims=True)
            grads.append((dW, db))

            if i > 0:
                curr_d = curr_d @ self.weights[i].T

        grads.reverse()
        return grads

    def get_parameters(self):
        return {
            'architecture': 'MLP',
            'layer_sizes': self.layer_sizes,
            'activation': self.activation,
            'mode': self.mode,
            'weights': [w.tolist() for w in self.weights],
            'biases':  [b.tolist() for b in self.biases],
        }

    def load_from_dict(self, d):
        self.weights = [np.array(w) for w in d['weights']]
        self.biases  = [np.array(b) for b in d['biases']]

    def feature_importance(self, X):
        """Gradient-based saliency: mean |∂output/∂input| per feature."""
        self.set_training(False)
        out = self.forward(X)
        # Upstream gradient: ones (sum over classes)
        dout = np.ones_like(out) / out.shape[1]
        n    = len(self.weights)
        curr_d = dout

        for i in reversed(range(n)):
            if i < n - 1:
                curr_d = curr_d * self.act_deriv(self.z_cache[i])
            if i > 0:
                curr_d = curr_d @ self.weights[i].T

        importance = np.abs(curr_d).mean(axis=0)
        return importance / (importance.sum() + 1e-9)


# ══════════════════════════════════════════════════════════════════════════════
# Architecture 2: PulseMindResNet — Residual Network
# ══════════════════════════════════════════════════════════════════════════════

class ResidualBlock:
    """
    Two-layer residual block with skip connection.
    Forces the sub-network to learn the *residual* Δ, not the full mapping.
    Effectively eliminates vanishing gradients in very deep networks.
    """

    def __init__(self, size, activation='relu', dropout_rate=0.0):
        self.W1 = he_init(size, size)
        self.b1 = np.zeros((1, size))
        self.W2 = he_init(size, size)
        self.b2 = np.zeros((1, size))
        self.act_fn, self.act_deriv = ACTIVATIONS[activation]
        self.dropout = Dropout(dropout_rate)
        self.ln1 = LayerNorm(size)
        self.ln2 = LayerNorm(size)
        self._cache = {}

    def forward(self, x):
        identity = x
        z1 = x @ self.W1 + self.b1
        a1 = self.ln1.forward(self.act_fn(z1))
        a1 = self.dropout.forward(a1)
        z2 = a1 @ self.W2 + self.b2
        a2 = self.ln2.forward(z2)
        out = self.act_fn(a2 + identity)   # skip connection
        self._cache = {'x': x, 'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2}
        return out

    def backward(self, dout):
        cache = self._cache
        # dout flows through both skip path and main path
        d_main = dout * self.act_deriv(cache['a2'] + cache['x'])

        # Layer 2
        d_ln2  = self.ln2.backward(d_main)
        dW2    = cache['a1'].T @ d_ln2
        db2    = d_ln2.sum(0, keepdims=True)
        da1    = d_ln2 @ self.W2.T

        # Layer 1
        da1    = self.dropout.backward(da1)
        da1    = self.ln1.backward(da1)
        d_z1   = da1 * self.act_deriv(cache['z1'])
        dW1    = cache['x'].T @ d_z1
        db1    = d_z1.sum(0, keepdims=True)
        d_skip = dout * self.act_deriv(cache['a2'] + cache['x'])
        dx     = d_z1 @ self.W1.T + d_skip   # skip gradient

        return dx, [(dW1, db1), (dW2, db2)]

    def get_parameters(self):
        return {
            'W1': self.W1.tolist(), 'b1': self.b1.tolist(),
            'W2': self.W2.tolist(), 'b2': self.b2.tolist(),
        }

    def load_from_dict(self, d):
        self.W1 = np.array(d['W1']); self.b1 = np.array(d['b1'])
        self.W2 = np.array(d['W2']); self.b2 = np.array(d['b2'])


class PulseMindResNet:
    """
    Input → Projection → N × ResidualBlocks → Output head.
    Preferred for deeper architectures (>4 layers) where plain MLPs degrade.
    """

    def __init__(self, input_size, hidden_size, n_blocks, output_size,
                 activation='relu', dropout_rate=0.0, mode='classification'):
        self.mode = mode
        self.hidden_size = hidden_size
        self.n_blocks = n_blocks
        self.output_size = output_size

        # Projection: input_size → hidden_size
        self.proj_W = he_init(input_size, hidden_size)
        self.proj_b = np.zeros((1, hidden_size))
        self.proj_ln = LayerNorm(hidden_size)

        self.blocks = [
            ResidualBlock(hidden_size, activation, dropout_rate)
            for _ in range(n_blocks)
        ]

        # Output head
        self.head_W = he_init(hidden_size, output_size)
        self.head_b = np.zeros((1, output_size))

        self._cache = {}

    def set_training(self, flag):
        for block in self.blocks:
            block.dropout.training = flag

    def forward(self, X):
        # Projection
        z_proj = X @ self.proj_W + self.proj_b
        a_proj = self.proj_ln.forward(np.maximum(0, z_proj))
        self._cache['X'] = X
        self._cache['z_proj'] = z_proj
        self._cache['a_proj'] = a_proj

        # Residual blocks
        curr = a_proj
        self._cache['block_inputs'] = []
        for block in self.blocks:
            self._cache['block_inputs'].append(curr)
            curr = block.forward(curr)
        self._cache['after_blocks'] = curr

        # Head
        out = curr @ self.head_W + self.head_b
        if self.mode == 'classification':
            out = softmax(out)
        self._cache['out'] = out
        return out

    def backward(self, dout):
        grads = []
        # Head
        dW_head = self._cache['after_blocks'].T @ dout
        db_head = dout.sum(0, keepdims=True)
        grads.append((dW_head, db_head))
        d_curr = dout @ self.head_W.T

        # Residual blocks (reverse)
        for block in reversed(self.blocks):
            d_curr, block_grads = block.backward(d_curr)
            grads.extend(block_grads)

        # Projection
        d_proj = self.proj_ln.backward(d_curr)
        d_proj *= (self._cache['z_proj'] > 0)
        dW_proj = self._cache['X'].T @ d_proj
        db_proj = d_proj.sum(0, keepdims=True)
        grads.append((dW_proj, db_proj))
        grads.reverse()
        return grads

    def _all_weights(self):
        """Flat list of (W, b) pairs for the gradient update step."""
        pairs = [(self.proj_W, self.proj_b)]
        for block in self.blocks:
            pairs += [(block.W1, block.b1), (block.W2, block.b2)]
        pairs.append((self.head_W, self.head_b))
        return pairs

    def get_parameters(self):
        return {
            'architecture': 'ResNet',
            'hidden_size': self.hidden_size,
            'n_blocks': self.n_blocks,
            'output_size': self.output_size,
            'mode': self.mode,
            'proj_W': self.proj_W.tolist(),
            'proj_b': self.proj_b.tolist(),
            'blocks': [b.get_parameters() for b in self.blocks],
            'head_W': self.head_W.tolist(),
            'head_b': self.head_b.tolist(),
        }

    def load_from_dict(self, d):
        self.proj_W = np.array(d['proj_W'])
        self.proj_b = np.array(d['proj_b'])
        for block, bp in zip(self.blocks, d['blocks']):
            block.load_from_dict(bp)
        self.head_W = np.array(d['head_W'])
        self.head_b = np.array(d['head_b'])

    def feature_importance(self, X):
        """Gradient-based saliency (input → output sensitivity)."""
        self.set_training(False)
        out = self.forward(X)
        dout = np.ones_like(out) / out.shape[1]
        grads = self.backward(dout)
        # Last grad pair is projection input gradient
        dx = X * (grads[0][0].T * np.ones_like(X))
        importance = np.abs(dx).mean(axis=0)
        return importance / (importance.sum() + 1e-9)


# ══════════════════════════════════════════════════════════════════════════════
# Architecture 3: PulseMindAttentionNet — Transformer-Inspired (Explainable)
# ══════════════════════════════════════════════════════════════════════════════

class ScaledDotProductAttention:
    """
    Single-head scaled dot-product attention.
    Attention scores double as explainability weights:
    feature i attending to feature j tells us how much feature i
    depends on feature j for a given prediction.
    """

    def __init__(self, d_model):
        scale = np.sqrt(d_model)
        self.W_Q = he_init(d_model, d_model) / scale
        self.W_K = he_init(d_model, d_model) / scale
        self.W_V = he_init(d_model, d_model) / scale
        self.W_O = he_init(d_model, d_model) / scale
        self._cache = {}
        self.attn_weights = None   # stored for interpretability

    def forward(self, x):
        """x: (batch, seq, d_model) — here seq = feature dimension."""
        Q = x @ self.W_Q
        K = x @ self.W_K
        V = x @ self.W_V
        d = Q.shape[-1]
        scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d)
        attn = softmax(scores.reshape(-1, scores.shape[-1])).reshape(scores.shape)
        self.attn_weights = attn   # (batch, seq, seq) — inspectable
        ctx = attn @ V
        out = ctx @ self.W_O
        self._cache = {'x': x, 'Q': Q, 'K': K, 'V': V, 'attn': attn, 'ctx': ctx}
        return out

    def get_parameters(self):
        return {
            'W_Q': self.W_Q.tolist(), 'W_K': self.W_K.tolist(),
            'W_V': self.W_V.tolist(), 'W_O': self.W_O.tolist(),
        }

    def load_from_dict(self, d):
        self.W_Q = np.array(d['W_Q']); self.W_K = np.array(d['W_K'])
        self.W_V = np.array(d['W_V']); self.W_O = np.array(d['W_O'])


class PulseMindAttentionNet:
    """
    Transformer-inspired architecture:
        Input → Token Embedding → N × (Attention + FFN + LayerNorm) → Pool → Head

    Features as "tokens": each input feature becomes a 1×d_model embedding,
    then attention reveals cross-feature dependencies (fully explainable via attn_weights).

    Best for: understanding WHICH features drive predictions (explainability).
    """

    def __init__(self, input_size, d_model=32, n_heads=1, n_layers=2,
                 output_size=1, dropout_rate=0.0, mode='classification'):
        self.input_size  = input_size
        self.d_model     = d_model
        self.n_layers    = n_layers
        self.output_size = output_size
        self.mode        = mode

        # Feature embedding: each scalar → d_model vector
        self.embed_W = he_init(1, d_model)
        self.embed_b = np.zeros((1, d_model))

        # Transformer layers
        self.attn_layers = [ScaledDotProductAttention(d_model) for _ in range(n_layers)]
        self.ffn_W1 = [he_init(d_model, d_model * 4) for _ in range(n_layers)]
        self.ffn_b1 = [np.zeros((1, d_model * 4)) for _ in range(n_layers)]
        self.ffn_W2 = [he_init(d_model * 4, d_model) for _ in range(n_layers)]
        self.ffn_b2 = [np.zeros((1, d_model)) for _ in range(n_layers)]
        self.lns_pre  = [LayerNorm(d_model) for _ in range(n_layers)]
        self.lns_post = [LayerNorm(d_model) for _ in range(n_layers)]

        # Output head (after mean pooling over features)
        self.head_W = he_init(d_model, output_size)
        self.head_b = np.zeros((1, output_size))

        self._cache = {}

    def set_training(self, flag):
        pass  # no dropout in attention by default

    def forward(self, X):
        """X: (batch, input_size)  →  out: (batch, output_size)"""
        batch = X.shape[0]
        # Embed: (batch, input_size, 1) → (batch, input_size, d_model)
        x_3d = X[:, :, np.newaxis]       # (batch, seq, 1)
        tokens = x_3d @ self.embed_W.T[np.newaxis] + self.embed_b   # (batch, seq, d_model)

        for i in range(self.n_layers):
            # Pre-norm attention
            normed  = np.array([self.lns_pre[i].forward(tokens[b]) for b in range(batch)])
            attn_out = self.attn_layers[i].forward(normed)
            tokens  = tokens + attn_out    # residual

            # FFN
            normed2 = np.array([self.lns_post[i].forward(tokens[b]) for b in range(batch)])
            ff1 = np.maximum(0, normed2 @ self.ffn_W1[i] + self.ffn_b1[i])
            ff2 = ff1 @ self.ffn_W2[i] + self.ffn_b2[i]
            tokens = tokens + ff2          # residual

        # Mean pool over sequence (feature) dimension → (batch, d_model)
        pooled = tokens.mean(axis=1)
        out = pooled @ self.head_W + self.head_b
        if self.mode == 'classification':
            out = softmax(out)
        self._cache = {'X': X, 'tokens': tokens, 'pooled': pooled}
        return out

    def feature_importance(self, X):
        """
        Attention-based importance:
        Mean attention weight received by each feature token across all layers.
        This is directly interpretable: which input features does the model attend to?
        """
        self.forward(X)
        importances = np.zeros(X.shape[1])
        for layer in self.attn_layers:
            if layer.attn_weights is not None:
                # Mean over batch and head: (seq, seq) → sum per column (received attention)
                importances += layer.attn_weights.mean(axis=(0, 1))
        importances = np.abs(importances)
        return importances / (importances.sum() + 1e-9)

    def get_parameters(self):
        return {
            'architecture': 'AttentionNet',
            'input_size':  self.input_size,
            'd_model':     self.d_model,
            'n_layers':    self.n_layers,
            'output_size': self.output_size,
            'mode':        self.mode,
            'embed_W':     self.embed_W.tolist(),
            'embed_b':     self.embed_b.tolist(),
            'attn_layers': [a.get_parameters() for a in self.attn_layers],
            'ffn_W1':      [w.tolist() for w in self.ffn_W1],
            'ffn_b1':      [b.tolist() for b in self.ffn_b1],
            'ffn_W2':      [w.tolist() for w in self.ffn_W2],
            'ffn_b2':      [b.tolist() for b in self.ffn_b2],
            'head_W':      self.head_W.tolist(),
            'head_b':      self.head_b.tolist(),
        }

    def load_from_dict(self, d):
        self.embed_W  = np.array(d['embed_W'])
        self.embed_b  = np.array(d['embed_b'])
        for layer, ld in zip(self.attn_layers, d['attn_layers']):
            layer.load_from_dict(ld)
        self.ffn_W1   = [np.array(w) for w in d['ffn_W1']]
        self.ffn_b1   = [np.array(b) for b in d['ffn_b1']]
        self.ffn_W2   = [np.array(w) for w in d['ffn_W2']]
        self.ffn_b2   = [np.array(b) for b in d['ffn_b2']]
        self.head_W   = np.array(d['head_W'])
        self.head_b   = np.array(d['head_b'])


# ══════════════════════════════════════════════════════════════════════════════
# Architecture 4: PulseMindEnsemble — Soft-Voting Ensemble
# ══════════════════════════════════════════════════════════════════════════════

class PulseMindEnsemble:
    """
    Combines multiple base models via soft-voting (averaged probabilities).
    Dramatically reduces variance vs any single model — best for final predictions.
    Sub-models can be MLP, ResNet, or AttentionNet.
    """

    def __init__(self, models, weights=None):
        self.models  = models
        self.weights = weights or [1.0 / len(models)] * len(models)

    def set_training(self, flag):
        for m in self.models:
            m.set_training(flag)

    def forward(self, X):
        preds = np.stack([m.forward(X) for m in self.models], axis=0)  # (n_models, batch, out)
        w = np.array(self.weights)[:, np.newaxis, np.newaxis]
        return (preds * w).sum(axis=0)

    def feature_importance(self, X):
        importances = np.stack([m.feature_importance(X) for m in self.models])
        return importances.mean(axis=0)

    def get_parameters(self):
        return {
            'architecture': 'Ensemble',
            'weights': self.weights,
            'models':  [m.get_parameters() for m in self.models],
        }

    def load_from_dict(self, d):
        self.weights = d['weights']
        for model, md in zip(self.models, d['models']):
            model.load_from_dict(md)


# ══════════════════════════════════════════════════════════════════════════════
# Legacy Aliases (backwards compatible with old views.py / train_pulse_mind.py)
# ══════════════════════════════════════════════════════════════════════════════

def PulseMindClassifier(input_size, hidden_layers, output_size, activation='relu', dropout_rate=0.0):
    """Factory: builds a PulseMindMLP for classification."""
    sizes = [input_size] + hidden_layers + [output_size]
    return PulseMindMLP(sizes, activation=activation, dropout_rate=dropout_rate, mode='classification')


def PulseMindRegressor(input_size, hidden_layers, activation='relu', dropout_rate=0.0):
    """Factory: builds a PulseMindMLP for regression (single output)."""
    sizes = [input_size] + hidden_layers + [1]
    return PulseMindMLP(sizes, activation=activation, dropout_rate=dropout_rate, mode='regression')


# ── Model Registry ────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    'mlp':          PulseMindMLP,
    'resnet':       PulseMindResNet,
    'attention':    PulseMindAttentionNet,
    'ensemble':     PulseMindEnsemble,
}


def build_model(arch: str, input_size: int, output_size: int, mode: str = 'classification', **kwargs):
    """
    Convenience factory for all architectures.

    Examples
    --------
    >>> model = build_model('mlp', 18, 10, hidden_layers=[256, 128, 64])
    >>> model = build_model('resnet', 18, 10, hidden_size=128, n_blocks=4)
    >>> model = build_model('attention', 18, 10, d_model=32, n_layers=3)
    """
    arch = arch.lower()
    if arch == 'mlp':
        hidden = kwargs.get('hidden_layers', [256, 128, 64])
        sizes  = [input_size] + hidden + [output_size]
        return PulseMindMLP(
            sizes,
            activation    = kwargs.get('activation', 'relu'),
            dropout_rate  = kwargs.get('dropout_rate', 0.2),
            mode          = mode,
        )
    elif arch == 'resnet':
        return PulseMindResNet(
            input_size,
            hidden_size   = kwargs.get('hidden_size', 128),
            n_blocks      = kwargs.get('n_blocks', 4),
            output_size   = output_size,
            activation    = kwargs.get('activation', 'relu'),
            dropout_rate  = kwargs.get('dropout_rate', 0.2),
            mode          = mode,
        )
    elif arch == 'attention':
        return PulseMindAttentionNet(
            input_size,
            d_model       = kwargs.get('d_model', 32),
            n_layers      = kwargs.get('n_layers', 3),
            output_size   = output_size,
            dropout_rate  = kwargs.get('dropout_rate', 0.1),
            mode          = mode,
        )
    elif arch == 'ensemble':
        sub_archs = kwargs.get('sub_archs', ['mlp', 'resnet'])
        sub_models = [build_model(a, input_size, output_size, mode, **kwargs) for a in sub_archs]
        return PulseMindEnsemble(sub_models)
    else:
        raise ValueError(f"Unknown architecture '{arch}'. Choose from: {list(MODEL_REGISTRY)}")
