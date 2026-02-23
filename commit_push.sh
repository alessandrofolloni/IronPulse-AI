#!/usr/bin/env bash
# ── IronPulse — Git Commit & Push Helper ─────────────────────────
# Run this from the GYM directory:  bash commit_push.sh
# Or with a custom message:         bash commit_push.sh "feat: my change"

set -e
cd "$(dirname "$0")"

MSG="${1:-"feat: sophisticated AI engine + improved graphics + complete training pipeline"}"

echo "📦 Staging all changes..."
git add -A

echo "📝 Committing with message: $MSG"
git commit -m "$MSG

Changes in this commit:

AI Engine (core/ai/engine.py):
- Added PulseMindMLP with He init, LayerNorm, configurable Dropout & activation
- Added PulseMindResNet with skip-connections (eliminates vanishing gradients)
- Added PulseMindAttentionNet (transformer-inspired, cross-feature attention)
- Added PulseMindEnsemble (soft-voting, highest accuracy)
- All architectures expose get_parameters() / load_from_dict() / feature_importance()
- Added build_model() factory, MODEL_REGISTRY, legacy aliases for backwards compat

AI Trainer (core/ai/trainer.py):
- Adam optimiser with L2 weight decay and bias correction
- SGD with Nesterov-style momentum
- Cosine Annealing and Step LR schedulers
- Early stopping with best-checkpoint restoration
- Cross-Entropy, MSE, Huber loss functions
- Mini-batch shuffled training loop with W&B logging
- Proper feature engineering in prepare_workout_data() (18 real features)
- prepare_regression_data() for strength prediction task

Training Pipeline (train_pulse_mind.py):
- Easy CLI: --arch, --epochs, --wandb, --pretrain, --sweep, --quick, --task
- Synthetic pre-training to warm-start before real data
- Architecture sweep mode (compares all 4 archs)
- Gradient-based feature importance output table
- Timestamped weight exports + DB metadata update

Bug Fixes (core/views.py):
- Added missing datetime, numpy imports
- Fixed 'model.get_parameters()' (was undefined in old engine)
- Fixed api_generate_plan using proper datetime.date.today()
- Refactored api_train_ai to use new ResNet architecture + Adam
- Refactored api_generate_plan to actually use model.forward() for ranking
- Refactored api_predict_strength with real last-session data

AI Lab UI (templates/core/ai_lab.html):
- Architecture selector (MLP / ResNet / AttentionNet / Ensemble)
- Live loss chart (Chart.js)
- Animated training progress bar
- Real-time console with color-coded output
- Model metrics display (accuracy, samples, version)
- Feature importance bar chart
- Architecture explainer cards

CSS System:
- Added btn-icon, btn-success, btn-indigo, card-highlight classes
- Improved fade-in animation timing
- Custom scrollbar styling
- Added ::selection colour

Tests (core/tests/test_ai.py):
- 40+ tests covering all architectures, loss functions, optimisers,
  schedulers, early stopping, serialisation, and trainer integration

Docs (README.md):
- Complete training CLI reference table
- Feature engineering documentation (all 18 features explained)
- Architecture comparison table with use-case guidance
- Full API endpoint reference
- Updated project architecture diagram"

echo "🚀 Pushing to origin..."
git push origin main

echo ""
echo "✅ Done! Check your repository on GitHub."
