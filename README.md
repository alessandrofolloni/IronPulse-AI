# ⚡ IronPulse — Elite Fitness Management Platform

> *Train smarter. Track with precision. Optimise with AI.*

IronPulse is a dark-luxury gym management web application built with **Django 4.2** and a **100% custom from-scratch neural network engine (PulseMind AI)**. Designed for serious athletes who want scientific strength tracking and AI-powered workout optimisation — zero external ML frameworks.

---

## ✨ Core Features

| Module | Description |
|---|---|
| 🏠 **Dashboard** | Activity heatmaps, rapid stat tiles, goal tracking, nutrition snapshot |
| ⚡ **1RM Intelligence** | 7 formulas (Epley, Brzycki, Lander, Lombardi, Mayhew, O'Conner, Wathan), load tables |
| 🏋️ **Workout Engine** | Session logging, live 1RM estimation per set, auto-PR detection |
| 📋 **Exercise Library** | 60+ pre-seeded movements, muscle-group filtering, instant search |
| 🏆 **PR Auto-Tracker** | Personal records updated automatically on every workout save |
| 📏 **Biometric Tracking** | Body weight/fat trends, BMI calculator, full measurement history |
| 🥗 **Nutrition Log** | Daily macro tracker: calories, protein, carbs, fat |
| 🎯 **Goals** | Visual progress bars for strength, weight-loss, or custom targets |
| 🧠 **PulseMind AI Lab** | Train, evaluate, and explain your custom neural network in the browser |
| 📥 **CSV Export** | Full training history export for external analysis |

---

## 🧠 PulseMind AI Engine

The AI component is built **entirely from scratch with NumPy** — no PyTorch, TensorFlow, or scikit-learn. This ensures full transparency and customisability.

### Architectures Available

| Architecture | Key Idea | Best For |
|---|---|---|
| **PulseMindMLP** | Deep Multi-Layer Perceptron with He init, LayerNorm, Dropout | Fast baseline |
| **PulseMindResNet** | Residual skip-connections (like ResNet-50) | Deep networks, default choice |
| **PulseMindAttentionNet** | Scaled dot-product attention over input features (transformer-inspired) | Explainability, cross-feature dependencies |
| **PulseMindEnsemble** | Soft-voting combination of MLP + ResNet | Best overall accuracy |

### Optimisers

- **Adam** (default) — Adaptive moment estimation with bias correction and L2 weight decay
- **SGD + Momentum** — Classic with Nesterov-style momentum

### LR Schedulers

- **Cosine Annealing** — Smooth η decay from η_max → η_min over T_max epochs
- **Step Decay** — Reduce LR by γ every N epochs

### Explainability

Every architecture exposes `feature_importance(X)` — gradient-based saliency (MLP/ResNet) or attention-weight scores (AttentionNet):

```python
importances = model.feature_importance(X_test)
# → per-feature [0…1] contribution scores
```

### Features Extracted from Training Data

| # | Feature | Description |
|---|---|---|
| 0 | `session_volume_kg` | Total kg lifted × reps |
| 1 | `num_sets` | Total sets logged |
| 2 | `num_exercises` | Unique exercises |
| 3–4 | `avg_weight`, `max_weight` | Weight distribution |
| 5–6 | `avg_reps`, `max_reps` | Volume distribution |
| 7–8 | `avg_one_rm`, `max_one_rm` | Strength estimates |
| 9 | `compound_ratio` | Fraction of compound movements |
| 10–15 | Muscle group ratios | Chest/Back/Legs/Shoulders/Arms/Core |
| 16 | `days_since_last` | Rest days between sessions |
| 17 | `session_count_30d` | Training frequency proxy |

---

## 🚀 Easy Training Procedure

### 1. Basic Training (Best Architecture — ResNet)
```bash
python train_pulse_mind.py
```

### 2. Full W&B Experiment Tracking
```bash
wandb login                            # one-time setup
python train_pulse_mind.py --wandb
```

### 3. Choose Architecture
```bash
python train_pulse_mind.py --arch mlp
python train_pulse_mind.py --arch resnet       # ← default & recommended
python train_pulse_mind.py --arch attention    # best for explainability
python train_pulse_mind.py --arch ensemble     # best accuracy, slower
```

### 4. Pre-training on Synthetic Data (Warm-Start)
```bash
python train_pulse_mind.py --pretrain --wandb
```

### 5. Architecture Sweep (Compare All)
```bash
python train_pulse_mind.py --sweep
```

### 6. Strength Prediction (Regression Task)
```bash
python train_pulse_mind.py --task regression --arch mlp
```

### 7. Quick Smoke Test
```bash
python train_pulse_mind.py --quick
```

### All CLI Options
```
--arch       mlp | resnet | attention | ensemble  (default: resnet)
--epochs     N                                    (default: 200)
--task       classification | regression          (default: classification)
--wandb      Enable Weights & Biases logging
--pretrain   Pre-train on synthetic data first
--sweep      Run all architectures, compare results
--quick      50 epochs, no W&B (for testing)
--seed       Random seed for reproducibility     (default: 42)
```

---

## 🛠️ Installation & Setup

```bash
# 1. Clone & enter directory
git clone https://github.com/alessandrofolloni/GYM.git
cd GYM

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env: set DB_ENGINE=sqlite (default) or mysql

# 5. Database setup
python manage.py makemigrations
python manage.py migrate
python seed.py                         # Seeds 62+ exercises

# 6. Launch dev server
python manage.py runserver
```

Visit [http://localhost:8000](http://localhost:8000)

---

## 🧪 Testing

```bash
# Run all tests
pytest core/tests/ -v

# Run with coverage
pytest core/tests/ --cov=core --cov-report=term-missing
```

### W&B Setup (optional but recommended)
```bash
pip install wandb
wandb login                            # enter your API key once
python train_pulse_mind.py --wandb --arch resnet --epochs 300
```

---

## 🏗️ Project Architecture

```
GYM/
├── gymapp/                  # Django project config
│   ├── settings.py          # DB, static, installed apps
│   └── urls.py              # Root URL routing
│
├── core/                    # Main application
│   ├── models.py            # Data models (Exercise, Session, Set, PR, Goal…)
│   ├── views.py             # View logic + REST API endpoints
│   ├── urls.py              # App URL patterns
│   ├── forms.py             # Django model forms
│   ├── admin.py             # Admin panel config
│   │
│   ├── ai/                  # PulseMind AI module
│   │   ├── engine.py        # Neural network architectures (MLP, ResNet, Attention, Ensemble)
│   │   ├── trainer.py       # Trainer, optimisers (Adam/SGD), schedulers, data prep
│   │   └── weights/         # Exported model weights (JSON, gitignored)
│   │
│   └── tests/               # Test suite
│       ├── test_models.py   # Model unit tests
│       ├── test_views.py    # View/API tests
│       └── test_ai.py       # AI engine tests
│
├── templates/               # Django HTML templates (dark-luxury UI)
│   ├── base.html            # Sidebar, topbar, global JS
│   └── core/                # One template per view
│
├── static/css/              # IronPulse Design System
│   ├── main.css             # Entry point (imports all modules)
│   └── modules/
│       ├── variables.css    # Design tokens (colours, spacing, shadows)
│       ├── base.css         # Reset, typography, animations
│       ├── layout.css       # Sidebar, topbar, grid system
│       ├── components.css   # Cards, buttons, tables, tags, badges
│       ├── forms.css        # Form inputs, validation styles
│       ├── ai.css           # AI Lab, 1RM Calculator specific styles
│       └── utilities.css    # Helpers, alerts, empty states
│
├── train_pulse_mind.py      # Easy training CLI (see above)
├── seed.py                  # Seeds 62+ exercises into the DB
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

---

## 🔬 External Datasets for Pre-training / Scaling

| Dataset | Use Case |
|---|---|
| [OpenPowerlifting](https://openpowerlifting.org/data) | Global 1RM baseline benchmarks |
| [FitRec](https://github.com/houyunxiang33/fitrec-project) | Workout intensity & heart rate sequences |
| [NHANES](https://www.cdc.gov/nchs/nhanes/) | Body composition population norms |

---

## 🌐 Tech Stack

| Layer | Technology |
|---|---|
| Backend | Django 4.2 (Python 3.9+) |
| Database | SQLite (dev) / MySQL (prod) |
| Frontend | Vanilla CSS Design System, Chart.js 4.4 |
| AI Engine | NumPy-only (from scratch) |
| Experiment Tracking | Weights & Biases (W&B) |
| Testing | pytest + pytest-django |

---

## 📡 API Endpoints

| Method | URL | Description |
|---|---|---|
| GET | `/api/1rm/` | Calculate 1RM (all formulas) |
| GET | `/api/train-ai/` | Trigger in-browser AI training |
| GET | `/api/generate-plan/` | Generate AI workout plan |
| GET | `/api/predict/<exercise_id>/` | Predict next-session weight (regression) |
| GET | `/api/exercise-history/<id>/` | Full training history for an exercise |
| GET | `/api/dashboard-stats/` | Activity stats for dashboard charts |
| GET | `/export/workouts/csv/` | Export all workouts as CSV |

---

*Built with precision. Powered by PulseMind AI. **IronPulse.***
