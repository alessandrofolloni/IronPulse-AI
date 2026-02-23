# 🏋️ IronPulse — Professional Gym Management Platform

> An all-in-one fitness tracking app with a production-grade AI recommendation engine, adversarial training, and SHAP explainability — built for daily use.

![Python](https://img.shields.io/badge/Python-3.9+-blue) ![Django](https://img.shields.io/badge/Django-4.2-green) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red) ![License](https://img.shields.io/badge/license-MIT-yellow)

---

## ✨ Features

### 📊 Training & Tracking
- **Workout logging** — sets, reps, weight, RPE, notes
- **1RM Calculator** — Epley, Brzycki, Adams, Lander, O'Conner formulas
- **Personal Records** — auto-detected and stored per exercise
- **Workout streak tracker** — consecutive day counter
- **CSV export** — full workout history download
- **Training Plans** — AI-generated or manual multi-day programmes

### 📏 Body & Nutrition
- **Body measurements** — weight, BMI, body fat %, waist, bicep trends
- **Nutrition log** — daily calories, protein, carbs, fat tracking
- **Body weight trend** — line chart over time

### 🧠 PulseMind AI Engine (PyTorch)
- **4 architectures** — MLP, ResNet, AttentionNet, Ensemble
- **Real exercise data** — 500+ exercises from WGER open-source database
- **Adversarial training** — FGSM (Fast Gradient Sign Method) for robustness
- **Mixup augmentation** — smoother decision boundaries
- **SHAP explainability** — feature attribution via SHAP DeepExplainer
- **Integrated Gradients** — axiomatic attribution baseline
- **W&B integration** — experiment tracking via Weights & Biases
- **One-click training** — from the AI Lab UI or CLI
- **Progressive overload recommendations** — suggested next session weights

### 🎯 Goals & Intelligence
- **Goal tracking** — strength, weight loss, endurance, habits
- **Muscle coverage radar** — 30-day volume distribution by muscle group
- **AI status dashboard** — model accuracy, last trained, version

---

## 🚀 Quick Start

### First-time Setup (run once)
```bash
# Clone and enter project
cd /path/to/GYM

# Activate virtual environment
source venv/bin/activate   # or: python3 -m venv venv && source venv/bin/activate

# Install everything, migrate DB, import 500+ real exercises
bash start.sh setup
```

### Daily Use
```bash
bash start.sh          # → http://localhost:8000
```

---

## 🧠 AI Training

### From UI
1. Go to **AI Lab** (`/ai-lab/`)
2. Select architecture (ResNet recommended)
3. Toggle **Adversarial Training** and **Mixup** (both on by default)
4. Click **Train PulseMind**
5. View SHAP/IG explainability charts after training

### From CLI
```bash
bash start.sh train            # ResNet, 200 epochs, adversarial + mixup
bash start.sh train-full       # 400 epochs
bash start.sh train-wandb      # With W&B logging
bash start.sh sweep            # Compare all 4 architectures

# Advanced
python train_pulse_mind.py --arch ensemble --epochs 300
python train_pulse_mind.py --no-adversarial --no-mixup  # Baseline (no augmentation)
python train_pulse_mind.py --sweep                       # Auto-select best arch
```

### Import Real Exercise Database
```bash
python manage.py import_wger              # Import 500 exercises (default)
python manage.py import_wger --max 1000   # Import up to 1000
python manage.py import_wger --overwrite  # Re-import + update existing
```

---

## 📁 Project Structure

```
GYM/
├── core/
│   ├── models.py               — Django models (Exercise, WorkoutSession, etc.)
│   ├── views.py                — All views + API endpoints
│   ├── urls.py                 — URL routing
│   ├── forms.py                — Django forms
│   ├── ai/
│   │   ├── __init__.py         — Public API for the AI package
│   │   ├── engine.py           — PyTorch model architectures
│   │   ├── trainer.py          — Training pipeline (adversarial, SHAP, IG)
│   │   └── weights/            — Saved model checkpoints (.pt files)
│   ├── management/
│   │   └── commands/
│   │       └── import_wger.py  — WGER exercise database importer
│   └── tests/
│       └── test_ai.py          — Pytest suite (architectures, adversarial, explainability)
├── templates/
│   ├── base.html               — Layout with sidebar navigation
│   └── core/
│       ├── dashboard.html       — Command center with radar + sparklines
│       ├── ai_lab.html          — AI training UI + SHAP charts
│       ├── workout_detail.html  — Per-session detail view
│       └── ...                 — 14 more templates
├── static/
│   └── css/
│       ├── main.css             — CSS entry point
│       └── modules/             — variables, components, layout, etc.
├── PulseMind_Inspection.ipynb  — AI inspection notebook (adversarial + SHAP)
├── train_pulse_mind.py         — CLI training script
├── start.sh                    — One-script launcher
├── requirements.txt            — All Python dependencies
└── seed.py                     — Optional local exercise seeder
```

---

## 🛠️ Commands Reference

| Command | Description |
|---|---|
| `bash start.sh` | Start dev server (auto-migrates) |
| `bash start.sh setup` | First-time: deps + migrate + import exercises |
| `bash start.sh import` | Import 500+ exercises from WGER |
| `bash start.sh train` | Train AI (ResNet, 200 epochs, adversarial) |
| `bash start.sh train-full` | Train AI (400 epochs) |
| `bash start.sh train-wandb` | Train with W&B logging |
| `bash start.sh sweep` | Compare all 4 architectures |
| `bash start.sh test` | Run pytest suite |
| `bash start.sh deps` | Install/update dependencies |
| `python manage.py import_wger` | Import real exercise data |

---

## 🧪 Testing

```bash
bash start.sh test
# or
pytest core/tests/test_ai.py -v

# Coverage includes:
# ✅ All 4 architectures — forward shape, log-probabilities, NaN, gradient flow
# ✅ FGSM attack — perturbation, bounds [0,2]
# ✅ Mixup — shape preserved, labels sum to 1
# ✅ Integrated Gradients — shape, non-negative
# ✅ SHAP fallback — returns array even without shap library
# ✅ Training pipeline — convergence, adversarial mode, early stopping
# ✅ Save/load roundtrip — identical outputs
# ✅ Performance — forward pass < 100ms, params < 5M
```

---

## 🤖 AI Architecture Details

### Feature Vector (18-dimensional)
| # | Feature | Description |
|---|---|---|
| 0 | Session Volume | Total weight × reps (normalised) |
| 1 | Total Sets | Number of sets performed |
| 2 | Unique Exercises | Distinct exercises count |
| 3-4 | Avg/Max Weight | Weight statistics (normalised) |
| 5-6 | Avg/Max Reps | Rep statistics |
| 7-8 | Avg/Best 1RM | Estimated 1RM values |
| 9 | Compound Ratio | Fraction of compound movements |
| 10-15 | Muscle %% | Volume share per muscle group |
| 16 | Rest Days | Days since last session |
| 17 | Monthly Frequency | Sessions in past 30 days |

### Training Pipeline
```
Real workout data → Feature extraction (18-dim) → Normalisation
     ↓
Mixup augmentation (α=0.2) → Clean forward pass
     ↓
FGSM perturbation (ε=0.03) → Adversarial forward pass
     ↓
Combined loss: 70% clean + 30% adversarial
     ↓
AdamW optimiser + Cosine LR with warm restarts → Early stopping
     ↓
SHAP DeepExplainer / Integrated Gradients → Feature attribution charts
```

---

## 🔑 Environment Variables (optional)
```bash
WANDB_API_KEY=your_key    # For W&B experiment tracking
DEBUG=True                # Django debug mode (default in dev)
SECRET_KEY=your_key       # Django secret key (auto-generated if missing)
```

---

## 📦 Dependencies
```
torch>=2.0         PyTorch (CPU, M1/MPS compatible)
django==4.2        Web framework
numpy>=1.24        Numerical computing
pandas>=1.5        Data analysis
shap>=0.44         SHAP explainability (optional but recommended)
scikit-learn>=1.3  Preprocessing utilities
wandb>=0.16        Experiment tracking (optional)
requests>=2.28     WGER API client
pytest>=8.0        Test runner
```

Install: `pip install -r requirements.txt`

---

## 🏗️ Roadmap

- [ ] PWA support — installable on phone home screen
- [ ] Built-in rest timer between sets
- [ ] Progressive overload alerts ("Try 82.5kg today")
- [ ] Weekly digest — auto-generated training summary email
- [ ] User authentication — multi-user support
- [ ] Export to Apple Health / Google Fit

---

*Built with ❤️ for athletes who take their training seriously.*
