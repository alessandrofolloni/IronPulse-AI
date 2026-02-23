#!/usr/bin/env python3
"""
IronPulse — PulseMind Training Script (PyTorch)
================================================
Trains the PulseMind AI model from the command line.
Uses real WGER exercise data + adversarial training + SHAP explainability.

Usage:
    python train_pulse_mind.py                       # ResNet, 200 epochs, adversarial
    python train_pulse_mind.py --arch mlp            # MLP architecture
    python train_pulse_mind.py --arch ensemble       # Ensemble (most robust)
    python train_pulse_mind.py --epochs 400          # Longer training
    python train_pulse_mind.py --no-adversarial      # Disable adversarial training
    python train_pulse_mind.py --wandb               # Enable W&B logging
    python train_pulse_mind.py --sweep               # Compare all architectures
"""

import os, sys, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'gymapp.settings')

import django
django.setup()

import numpy as np
from core.ai.trainer import prepare_workout_data, train_and_evaluate, FEATURE_NAMES
from core.models import WorkoutSession, Exercise, AIModelMetadata
from django.utils import timezone

RED = '\033[0;31m'; GREEN = '\033[0;32m'; YELLOW = '\033[1;33m'
CYAN = '\033[0;36m'; BOLD = '\033[1m'; NC = '\033[0m'

def banner():
    print(f"\n{BOLD}{CYAN}{'═'*55}")
    print(f"  🧠 PulseMind AI — Training Script")
    print(f"  IronPulse Fitness Intelligence Engine")
    print(f"{'═'*55}{NC}\n")

def print_results(results, arch):
    if not results.get('success'):
        print(f"{RED}Training failed: {results.get('error')}{NC}")
        return

    print(f"\n{GREEN}{'═'*55}{NC}")
    print(f"{BOLD}  ✅ Training Complete — {arch.upper()}{NC}")
    print(f"{'─'*55}")
    print(f"  Val Accuracy:   {CYAN}{results['val_accuracy']}%{NC}")
    print(f"  Test Accuracy:  {GREEN}{results['test_accuracy']}%{NC}")
    print(f"  Best Epoch:     {results['best_epoch']}")
    print(f"  Duration:       {results['duration_sec']:.1f}s")
    print(f"  Train samples:  {results['n_train']}")
    print(f"  Adversarial:    {'✅' if results['adversarial'] else '❌'}")
    print(f"  Mixup:          {'✅' if results['mixup'] else '❌'}")

    shap = results.get('shap_values') or results.get('ig_values')
    if shap:
        method = 'SHAP' if results.get('shap_values') else 'Integrated Gradients'
        print(f"\n{BOLD}  🔍 Top Feature Attribution ({method}):{NC}")
        paired = sorted(zip(FEATURE_NAMES, shap), key=lambda x: -abs(x[1]))
        for name, val in paired[:5]:
            bar = '█' * int(abs(val) * 30)
            print(f"    {name:20s}  {val:+.4f}  {CYAN}{bar}{NC}")

    print(f"{GREEN}{'═'*55}{NC}\n")

def save_metadata(results, arch, exercises_count):
    metadata, _ = AIModelMetadata.objects.get_or_create(pk=1)
    metadata.model_name  = f'PulseMind-{arch.upper()}'
    metadata.last_trained = timezone.now()
    metadata.accuracy    = results['test_accuracy']
    metadata.total_training_samples = results['n_train'] + results['n_val'] + results['n_test']
    metadata.version     = round((metadata.version or 1.0) + 0.1, 1)
    metadata.weights_info = {
        'architecture':  arch,
        'arch_info':     results.get('arch_info'),
        'shap_values':   results.get('shap_values'),
        'ig_values':     results.get('ig_values'),
        'feature_names': FEATURE_NAMES,
        'test_accuracy': results['test_accuracy'],
        'adversarial':   results['adversarial'],
        'mixup':         results['mixup'],
        'n_exercises':   exercises_count,
    }
    metadata.save()
    print(f"  {GREEN}Metadata saved to DB{NC}")

def main():
    parser = argparse.ArgumentParser(description='Train PulseMind AI')
    parser.add_argument('--arch', default='resnet',
                        choices=['mlp', 'resnet', 'attention', 'ensemble'],
                        help='Model architecture')
    parser.add_argument('--epochs', type=int, default=200, help='Training epochs')
    parser.add_argument('--no-adversarial', action='store_true',
                        help='Disable adversarial training (FGSM)')
    parser.add_argument('--no-mixup', action='store_true',
                        help='Disable Mixup augmentation')
    parser.add_argument('--wandb', action='store_true', help='Enable W&B logging')
    parser.add_argument('--sweep', action='store_true', help='Compare all architectures')
    args = parser.parse_args()

    banner()

    sessions  = WorkoutSession.objects.all()
    exercises = Exercise.objects.all()

    n_sess = sessions.count()
    n_ex   = exercises.count()
    print(f"  Sessions logged:   {CYAN}{n_sess}{NC}")
    print(f"  Exercises in DB:   {CYAN}{n_ex}{NC}")
    if n_sess < 20:
        print(f"  {YELLOW}⚠ Fewer than 20 real sessions — AI will use synthetic augmentation.{NC}")
        print(f"  {YELLOW}  Log more workouts for better personalization.{NC}")

    if n_ex == 0:
        print(f"\n{RED}No exercises found. Run:{NC}")
        print(f"  python manage.py import_wger\n")
        return

    use_adv   = not args.no_adversarial
    use_mixup = not args.no_mixup

    if args.sweep:
        print(f"\n{BOLD}  🔬 Architecture Sweep{NC}")
        best_acc = 0; best_arch = None
        for arch in ['mlp', 'resnet', 'attention', 'ensemble']:
            print(f"\n  Training {arch.upper()}...")
            r = train_and_evaluate(
                sessions, exercises, arch=arch, epochs=min(args.epochs, 100),
                use_wandb=args.wandb, use_adversarial=use_adv, use_mixup=use_mixup,
            )
            print_results(r, arch)
            if r.get('success') and r['test_accuracy'] > best_acc:
                best_acc = r['test_accuracy']; best_arch = arch
        print(f"\n{GREEN}  Best architecture: {BOLD}{best_arch}{NC}{GREEN} ({best_acc}%){NC}")
        if best_arch:
            r_best = train_and_evaluate(
                sessions, exercises, arch=best_arch, epochs=args.epochs,
                use_wandb=args.wandb, use_adversarial=use_adv, use_mixup=use_mixup,
            )
            save_metadata(r_best, best_arch, n_ex)
    else:
        print(f"\n  Training {args.arch.upper()} | {args.epochs} epochs | adv={use_adv} | mixup={use_mixup}\n")
        r = train_and_evaluate(
            sessions, exercises, arch=args.arch, epochs=args.epochs,
            use_wandb=args.wandb, use_adversarial=use_adv, use_mixup=use_mixup,
        )
        print_results(r, args.arch)
        if r.get('success'):
            save_metadata(r, args.arch, n_ex)

if __name__ == '__main__':
    main()
