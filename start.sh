#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
#  IronPulse — One Script To Rule Them All
# ═══════════════════════════════════════════════════════════════
#
#  bash start.sh              → start dev server (auto-migrates)
#  bash start.sh setup        → first-time: migrate + seed + import exercises
#  bash start.sh import       → import 500+ real exercises from WGER
#  bash start.sh train        → quick AI training (200 epochs, adversarial)
#  bash start.sh train-full   → full training (400 epochs)
#  bash start.sh train-wandb  → training with W&B logging
#  bash start.sh sweep        → compare all 4 architectures
#  bash start.sh test         → run pytest suite
#  bash start.sh deps         → install all Python dependencies
#
# ═══════════════════════════════════════════════════════════════

set -e

PYTHON=${PYTHON:-python3}
MANAGE="$PYTHON manage.py"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'

step() { echo -e "${CYAN}▶ $1${NC}"; }
ok()   { echo -e "${GREEN}✅ $1${NC}"; }
warn() { echo -e "${YELLOW}⚠  $1${NC}"; }
err()  { echo -e "${RED}❌ $1${NC}"; exit 1; }

CMD="${1:-serve}"

# ── Dependencies ─────────────────────────────────────────────
install_deps() {
    step "Installing Python dependencies..."
    pip install -r requirements.txt
    ok "Dependencies installed"
}

# ── Migrate ──────────────────────────────────────────────────
migrate() {
    step "Running migrations..."
    $MANAGE migrate --no-input
    ok "Database up to date"
}

# ── Seed exercises (fallback if WGER import wasn't run) ──────
seed() {
    step "Seeding base exercises..."
    $MANAGE loaddata core/fixtures/exercises.json 2>/dev/null || \
        $PYTHON seed.py 2>/dev/null || \
        warn "No seed file found — run 'bash start.sh import' to get real exercises"
}

# ── Import real exercises from WGER ─────────────────────────
import_exercises() {
    step "Importing exercises from WGER open-source database..."
    $MANAGE import_wger --max 500
    ok "Exercise library imported"
}

# ── AI Training ──────────────────────────────────────────────
train() {
    local epochs="${1:-200}"
    local wandb="${2:-false}"
    step "Training PulseMind AI (${epochs} epochs, adversarial=true, mixup=true)..."
    $PYTHON train_pulse_mind.py --arch resnet --epochs "$epochs" \
        $([ "$wandb" = "true" ] && echo "--wandb") \
        --adversarial --mixup
    ok "Training complete"
}

sweep() {
    step "Architecture sweep: MLP vs ResNet vs AttentionNet vs Ensemble..."
    for arch in mlp resnet attention ensemble; do
        echo -e "\n${YELLOW}── $arch ──${NC}"
        $PYTHON train_pulse_mind.py --arch "$arch" --epochs 100 --adversarial --mixup
    done
    ok "Sweep complete"
}

# ── Main ─────────────────────────────────────────────────────
case "$CMD" in
    serve|"")
        migrate
        step "Starting dev server → http://localhost:8000"
        $MANAGE runserver
        ;;
    setup)
        install_deps
        migrate
        import_exercises
        ok "Setup complete! Run 'bash start.sh' to start."
        ;;
    import)
        import_exercises
        ;;
    deps)
        install_deps
        ;;
    train)
        train 200 false
        ;;
    train-full)
        train 400 false
        ;;
    train-wandb)
        train 300 true
        ;;
    sweep)
        sweep
        ;;
    test)
        step "Running test suite..."
        pytest core/tests/ -v --tb=short
        ;;
    migrate)
        migrate
        ;;
    *)
        err "Unknown command: $CMD. Valid: serve, setup, import, train, train-full, train-wandb, sweep, test, deps"
        ;;
esac
