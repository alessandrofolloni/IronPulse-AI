#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
#  IronPulse — One Script To Rule Them All
# ═══════════════════════════════════════════════════════════════
#
#  bash start.sh           → start dev server  (auto-migrates first)
#  bash start.sh train     → quick AI training  (50 epochs)
#  bash start.sh train-full→ full training      (300 epochs + pretrain)
#  bash start.sh sweep     → compare all 4 architectures
#  bash start.sh test      → run test suite
#  bash start.sh seed      → seed exercises
#  bash start.sh setup     → first-time: migrate + seed
#
# ═══════════════════════════════════════════════════════════════

set -e
REPO="$(cd "$(dirname "$0")" && pwd)"

# ── Find Python ───────────────────────────────────────────────
for p in "$REPO/venv/bin/python3" "$REPO/venv/bin/python"; do
  [ -x "$p" ] && PYTHON="$p" && break
done
[ -z "$PYTHON" ] && [ -n "$VIRTUAL_ENV" ] && PYTHON="$(command -v python3 2>/dev/null || command -v python 2>/dev/null)"
[ -z "$PYTHON" ] && PYTHON="$(command -v python3 2>/dev/null)"
[ -z "$PYTHON" ] && { echo "❌ Python not found. Run:  source venv/bin/activate"; exit 1; }

echo "🐍  $($PYTHON --version 2>&1)"

# ── Helpers ───────────────────────────────────────────────────
header() { echo ""; echo "═══════════════════════════════════════"; echo "  ⚡ IronPulse — $1"; echo "═══════════════════════════════════════"; }

ensure_deps() {
  "$PYTHON" -m pip install -q -r "$REPO/requirements.txt" 2>&1 | grep -vE "already satisfied|WARNING|notice|DEPRECATION" || true
  # Auto-fix broken numpy (common on macOS mixed-arch setups)
  "$PYTHON" -c "import numpy" 2>/dev/null || {
    echo "⚠️  Fixing numpy…"
    "$PYTHON" -m pip install --force-reinstall "numpy>=1.24,<2.0" -q 2>&1 | tail -2
  }
}

auto_migrate() {
  echo "⚙️   Checking database…"
  "$PYTHON" "$REPO/manage.py" migrate --run-syncdb 2>&1 | grep -vE "^  Applying|^Operations|^Running" | head -5 || true
}

# ── Route ─────────────────────────────────────────────────────
MODE="${1:-server}"

case "$MODE" in

  server|"")
    header "Starting Dev Server"
    ensure_deps
    auto_migrate
    echo ""
    echo "🌐  http://localhost:8000"
    echo "    Ctrl+C to stop"
    echo ""
    "$PYTHON" "$REPO/manage.py" runserver
    ;;

  train)
    header "Quick AI Training (50 epochs)"
    ensure_deps
    auto_migrate
    "$PYTHON" "$REPO/train_pulse_mind.py" --quick
    ;;

  train-full)
    header "Full AI Training (300 epochs + pretrain)"
    ensure_deps
    auto_migrate
    "$PYTHON" "$REPO/train_pulse_mind.py" --arch resnet --epochs 300 --pretrain
    ;;

  train-wandb)
    header "Training with W&B"
    ensure_deps
    auto_migrate
    "$PYTHON" "$REPO/train_pulse_mind.py" --arch resnet --epochs 300 --pretrain --wandb
    ;;

  sweep)
    header "Architecture Sweep"
    ensure_deps
    auto_migrate
    "$PYTHON" "$REPO/train_pulse_mind.py" --sweep --epochs 80
    ;;

  test)
    header "Test Suite"
    ensure_deps
    "$PYTHON" -m pip install -q pytest pytest-django 2>/dev/null || true
    "$PYTHON" -m pytest "$REPO/core/tests/" -v --tb=short
    ;;

  seed)
    header "Seeding Database"
    auto_migrate
    "$PYTHON" "$REPO/seed.py"
    echo "✅  Done"
    ;;

  migrate)
    header "Migrations"
    "$PYTHON" "$REPO/manage.py" makemigrations
    "$PYTHON" "$REPO/manage.py" migrate
    echo "✅  Done"
    ;;

  setup)
    header "First-Time Setup"
    ensure_deps
    "$PYTHON" "$REPO/manage.py" migrate
    "$PYTHON" "$REPO/seed.py"
    echo ""
    echo "✅  Ready! Run:  bash start.sh"
    ;;

  *)
    echo "Usage: bash start.sh [server|train|train-full|sweep|test|seed|setup]"
    exit 1
    ;;
esac
