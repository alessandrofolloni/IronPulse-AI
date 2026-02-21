# 🏋️ IronPulse — Personal Gym Management Platform

A full-featured gym management web application built with **Django** (Python) and **MySQL**, featuring:

## ✨ Features

- **🏠 Dashboard** — Activity charts, stats, recent workouts, top PRs, goals overview, nutrition summary
- **⚡ 1RM Calculator** — 7 scientific formulas (Epley, Brzycki, Lander, Lombardi, Mayhew, O'Connor, Wathan), load percentage table, and Chart.js bar chart
- **🏋️ Workout Logger** — Session-based workout tracking with live 1RM preview per set as you enter weight/reps
- **📋 Exercise Library** — 60 pre-seeded exercises with muscle-group filter tabs
- **🏆 Personal Records** — Auto-tracked from workout sessions
- **📏 Body Measurements** — Weight trend chart with BMI calculation
- **🥗 Nutrition Log** — Daily macro and calorie tracking with summary
- **🎯 Goals** — Progress tracking with progress bars and deadlines

## 🚀 Quick Start (Local)

```bash
# 1. Clone the repo
git clone https://github.com/alessandrofolloni/GYM.git
cd GYM

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your settings

# 5. Migrate database (SQLite by default, MySQL in production)
python manage.py migrate

# 6. Seed exercises
python seed.py

# 7. Create superuser (for admin panel)
python manage.py createsuperuser

# 8. Run server
python manage.py runserver
```

Visit http://127.0.0.1:8000/

## 🔧 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DEBUG` | `True` | Debug mode |
| `DB_ENGINE` | `sqlite` | Use `mysql` for MySQL |
| `DB_NAME` | `gymapp` | Database name |
| `DB_USER` | `root` | Database user |
| `DB_PASSWORD` | `` | Database password |
| `DB_HOST` | `localhost` | Database host |
| `DB_PORT` | `3306` | MySQL port |

## 🛠️ Tech Stack

- **Backend**: Django 4.2, Python 3.9
- **Database**: MySQL (production) / SQLite (development)
- **Frontend**: Vanilla HTML/CSS/JS, Chart.js
- **Design**: Dark luxury gym aesthetic, custom CSS design system

## 📁 Project Structure

```
GYM/
├── gymapp/         # Django project config
├── core/           # Main app (models, views, urls)
│   ├── models.py   # Exercise, Workout, Measurement, Nutrition, Goals
│   ├── views.py    # All views + 1RM API
│   └── forms.py
├── templates/      # HTML templates
│   ├── base.html   # Sidebar + topbar layout
│   └── core/       # Feature templates
├── static/css/     # main.css (full design system)
├── seed.py         # Exercise seeder (60 exercises)
└── requirements.txt
```
