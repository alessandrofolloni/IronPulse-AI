# 🏋️ IronPulse — Elite Fitness Management Platform

IronPulse is a high-performance, dark-luxury gym management web application built with **Django** and **MySQL**. It's designed for serious athletes who want precise tracking of their training volume, scientific strength metrics, and body composition.

## ✨ Premium Features

- **🏠 Command Center (Dashboard)** — Real-time activity heatmaps, rapid stat summaries, and interactive goal tracking.
- **⚡ Scientific 1RM Intelligence** — Calculate your peak strength using 7 industry-standard scientific formulas (Epley, Brzycki, Lander, etc.) with automated percentage load tables.
- **🏋️ Elite Workout Engine** — Advanced session tracking with live 1RM estimation as you log your sets.
- **🔍 Smart Exercise Library** — Pre-seeded with 60+ foundational movements, featuring instant search and muscle-group filtering.
- **🏆 PR Auto-Tracker** — Your personal records are automatically identified and archived from your training logs.
- **📏 Biometric Tracking** — High-fidelity weight trend charts with automated BMI calculation and body fat tracking.
- **🥗 Macro & Nutrition Intel** — Daily snapshot of your caloric and protein intake to fuel performance.
- **🎯 Dynamic Goals** — Visual progress tracking for strength, weight loss, or custom milestones.
- **📥 Data Portability** — Export your entire training history to CSV for external analysis.

## 🚀 Velocity Stack

- **Backend**: Django 4.2 (Python 3.9)
- **Database**: MySQL (Production) / SQLite (Development)
- **Frontend**: Custom CSS Design System (Inter & Outfit Typography), Chart.js
- **Testing**: Pytest-Django (Core coverage: Models, Views, API)
- **Architecture**: Optimized with DB indexes and pre-fetched queries for maximum scalability.

## 🛠️ Installation & Setup

```bash
# 1. Clone & Initialize
git clone https://github.com/alessandrofolloni/GYM.git
cd GYM

# 2. Virtual Environment
python3 -m venv venv
source venv/bin/activate

# 3. Dependencies
pip install -r requirements.txt
pip install pytest pytest-django  # For testing

# 4. Global Configuration
cp .env.example .env
# Configure your DB_ENGINE (mysql/sqlite) and credentials in .env

# 5. Database Deployment
python manage.py makemigrations
python manage.py migrate
python seed.py          # Rapidly seed 60+ exercises

# 6. Ignite the Server
python manage.py runserver
```

## 🧪 Quality Assurance

We use `pytest` to ensure structural integrity across models and views.
```bash
PYTHONPATH=. pytest core/tests/ -c core/pytest.ini
```

## 📁 Architecture Overview

```
GYM/
├── gymapp/         # Core System Configuration
├── core/           # Engine (Models, Logic, API)
│   ├── tests/      # Automated Test Suite
│   ├── models.py   # Normalized Data Structures
│   └── views.py    # High-Performance Logic
├── templates/      # Dark Luxury UI (Semantic HTML)
├── static/css/     # IronPulse Design System (main.css)
└── seed.py         # Automated Data Seeder
```

---
*Built for performance. Managed with precision. **IronPulse.***
