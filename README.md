# 🏋️ IronPulse — Smart Fitness Tracker

**IronPulse** is a full-featured fitness tracking web app with a built-in AI engine that learns from your workout history to give you personalised recommendations.

---

## What You Can Do

### 🏠 Dashboard
Your command centre. See at a glance:
- Workouts in the last 30 days
- Total volume lifted (kg)
- Personal records
- Active goals
- 7-day activity chart
- Quick links to log workouts, meals, and measurements

### 🏋️ Workout Tracking
Log every training session with full detail:
- **Session name, date, notes, duration**
- **Sets**: exercise, weight, reps — as many as you need
- Volume and one-rep-max calculated automatically
- Browse, edit, and delete past sessions

### 📋 Exercise Library
A searchable, filterable database of exercises:
- Filter by muscle group (Chest, Back, Legs, Shoulders, etc.)
- Search by name
- See difficulty level, compound/isolation tag
- Add your own custom exercises

### 📏 Body Measurements
Track your body composition over time:
- Weight, body fat %, muscle mass
- Individual measurements: chest, waist, hips, arms, thighs
- Visual progress tracking

### 🥗 Nutrition Logging
Simple daily nutrition tracking:
- Calories, protein, carbs, fat
- Per-meal logging with meal type (breakfast, lunch, dinner, snack)
- Daily totals on the dashboard

### 🎯 Goals
Set and track fitness milestones:
- Target weight, strength, or habit goals
- Track progress status (active, completed, paused)
- See active goals count on dashboard

### ⚡ 1RM Calculator
Estimate your one-rep max for any exercise:
- Uses the Epley formula
- Enter weight and reps performed to get prediction

### 🧠 AI Lab — PulseMind Intelligence
The standout feature. A fully custom neural network (built from scratch with NumPy) that:
- **Learns from your actual workout history** to rank exercises
- **Generates personalised workout plans** — 3-day splits based on model predictions
- **Predicts target weights** for exercises based on your progression
- Supports 4 architectures you can select in the UI:
  - **Deep MLP** — fast, classic neural network
  - **ResNet** — skip-connections for deeper learning (recommended)
  - **AttentionNet** — transformer-inspired, shows which features matter most
  - **Ensemble** — combines models for highest accuracy

#### How it works (simple version):
1. Open the **AI Lab** page
2. Pick an architecture (ResNet is default)
3. Click **Train PulseMind AI** — the app extracts 18 features from your workout history, trains the model, and saves the weights
4. Click **Generate AI Workout Plan** — the model ranks exercises and creates a plan for you
5. See **feature importances** — understand what drives the AI's recommendations

---

## Getting Started

### Prerequisites
- Python 3.9+
- pip

### Installation

```bash
git clone https://github.com/alessandrofolloni/IronPulse-AI.git
cd IronPulse-AI

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# First-time setup (installs deps, migrates DB, seeds 60+ exercises)
bash start.sh setup
```

### Running

```bash
# Activate venv (if not already active)
source venv/bin/activate

# Start the server
bash start.sh
```

Then open **http://localhost:8000** in your browser.

### All Commands

| Command | What it does |
|---------|-------------|
| `bash start.sh` | Start dev server at localhost:8000 |
| `bash start.sh setup` | First-time setup (migrate + seed) |
| `bash start.sh train` | Quick AI training (50 epochs) |
| `bash start.sh train-full` | Full training (300 epochs + synthetic pre-training) |
| `bash start.sh sweep` | Compare all 4 architectures side-by-side |
| `bash start.sh test` | Run the test suite |
| `bash start.sh seed` | Seed the exercise database |
| `bash start.sh migrate` | Apply database migrations |

---

## Project Structure

```
IronPulse-AI/
├── start.sh                 ← one script to run everything
├── manage.py                ← Django management
├── train_pulse_mind.py      ← CLI training script
├── seed.py                  ← seeds 60+ exercises
├── requirements.txt
│
├── gymapp/                  ← Django project settings
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
│
├── core/                    ← main application
│   ├── models.py            ← Exercise, WorkoutSession, WorkoutPlan, etc.
│   ├── views.py             ← all page views + API endpoints
│   ├── urls.py              ← URL routing
│   ├── forms.py             ← Django forms
│   ├── ai/                  ← PulseMind AI engine
│   │   ├── engine.py        ← neural network architectures (NumPy)
│   │   ├── trainer.py       ← training loop, optimisers, schedulers
│   │   └── weights/         ← saved model weights (gitignored)
│   └── tests/
│       └── test_ai.py       ← 40+ tests for AI engine
│
├── templates/core/          ← HTML templates
│   ├── dashboard.html
│   ├── ai_lab.html
│   ├── exercises.html
│   ├── workouts.html
│   └── ...
│
└── static/css/              ← stylesheets
    ├── main.css
    └── modules/
        ├── base.css
        ├── components.css
        ├── ai.css
        └── utilities.css
```

---

## Technical Details (for developers)

### AI Engine Architecture

The PulseMind engine is built **entirely from scratch using NumPy** — no PyTorch, no TensorFlow, no scikit-learn.

| Architecture | Parameters | Key Feature |
|---|---|---|
| `PulseMindMLP` | He init, LayerNorm, Dropout | Fast baseline |
| `PulseMindResNet` | Skip-connections per block | Deep without vanishing gradients |
| `PulseMindAttentionNet` | Scaled dot-product attention | Interpretable attention weights |
| `PulseMindEnsemble` | Soft-voting over sub-models | Highest accuracy |

### 18 Input Features

The model extracts these features from workout history:
1. Session volume (kg) · 2. Total sets · 3. Unique exercises
4. Avg weight · 5. Max weight · 6. Avg reps · 7. Max reps
8. Avg estimated 1RM · 9. Best estimated 1RM
10. Compound exercise ratio
11–16. Muscle group distribution (chest, back, legs, shoulders, arms, core)
17. Days since last session · 18. Monthly session frequency

### Training Pipeline

- **Optimisers**: Adam (bias-corrected, L2 decay) or SGD+Momentum
- **LR Schedulers**: Cosine Annealing, Step Decay
- **Early Stopping**: patience-based with best-checkpoint restoration
- **Loss Functions**: Cross-Entropy (classification), MSE, Huber (regression)
- **Data**: mini-batch shuffled, train/val split

### Weights & Biases Integration

For experiment tracking:
```bash
export WANDB_API_KEY=your_key_here
bash start.sh train-wandb
```

### API Endpoints

| Method | URL | Description |
|--------|-----|-------------|
| GET | `/api/train/` | Train the AI model |
| GET | `/api/generate-plan/` | Generate workout plan from trained model |
| GET | `/api/predict-strength/` | Predict target weight for an exercise |

### Running Tests

```bash
bash start.sh test
```

Covers: all 4 architectures, forward/backward pass shapes, serialisation round-trips, loss functions, optimisers, LR schedulers, early stopping, feature importance, and trainer integration.

---

## Dependencies

- Django 4.2
- NumPy (< 2.0)
- wandb (optional, for experiment tracking)
- crispy-forms (form styling)

---

## License

MIT

---

Built with 🧡 by Alessandro Folloni
