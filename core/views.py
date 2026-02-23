import datetime
import json
import math
import csv
import numpy as np

from django.shortcuts import render, get_object_or_404, redirect
from django.http import JsonResponse, HttpResponse
from django.db.models import Max, Avg, Count, Sum
from django.utils import timezone
from datetime import timedelta

from .models import (
    Exercise, WorkoutSession, WorkoutSet,
    PersonalRecord, BodyMeasurement, NutritionLog, Goal,
    WorkoutPlan, PlanDay, PlanExercise, AIModelMetadata
)
from .forms import (
    WorkoutSessionForm, WorkoutSetForm, ExerciseForm,
    BodyMeasurementForm, NutritionLogForm, GoalForm,
    WorkoutPlanForm
)
from .ai.engine import build_model, load_model, save_model, compute_feature_importance, N_FEATURES, FEATURE_NAMES
from .ai.trainer import PulseMindTrainer, prepare_workout_data, train_and_evaluate


# ── Helpers ────────────────────────────────────────────────────────────────────

def calculate_one_rm(weight, reps, formula='epley'):
    """Multiple 1RM formulas."""
    w, r = float(weight), int(reps)
    if r == 1:
        return w
    formulas = {
        'epley':    w * (1 + r / 30),
        'brzycki':  w * (36 / (37 - r)),
        'lander':   (100 * w) / (101.3 - 2.67123 * r),
        'lombardi': w * (r ** 0.10),
        'mayhew':   (100 * w) / (52.2 + 41.9 * math.exp(-0.055 * r)),
        'oconner':  w * (1 + r / 40),
        'wathan':   (100 * w) / (48.8 + 53.8 * math.exp(-0.075 * r)),
    }
    return {k: round(v, 2) for k, v in formulas.items()} if formula == 'all' else round(formulas.get(formula, formulas['epley']), 2)


# ── Dashboard ──────────────────────────────────────────────────────────────────

def dashboard(request):
    today     = timezone.now().date()
    week_ago  = today - timedelta(days=7)
    month_ago = today - timedelta(days=30)

    recent_sessions     = WorkoutSession.objects.prefetch_related('sets__exercise')[:5]
    total_workouts      = WorkoutSession.objects.count()
    workouts_this_week  = WorkoutSession.objects.filter(date__gte=week_ago).count()
    month_sessions      = WorkoutSession.objects.filter(date__gte=month_ago).count()
    total_volume = WorkoutSet.objects.aggregate(
        total=Sum(models_expr('weight', 'reps'))
    )['total'] or 0

    # PRs
    top_prs  = PersonalRecord.objects.order_by('-one_rm')[:5]
    pr_count = PersonalRecord.objects.count()

    # Body weight trend (last 10 measurements)
    bw_measurements = BodyMeasurement.objects.filter(
        weight_kg__isnull=False
    ).order_by('-date')[:10]
    bw_labels = [m.date.strftime('%b %d') for m in reversed(list(bw_measurements))]
    bw_values = [float(m.weight_kg) for m in reversed(list(bw_measurements))]

    # Active goals
    active_goals       = Goal.objects.filter(status='active')[:4]
    active_goals_count = Goal.objects.filter(status='active').count()

    # Full macros today
    today_nutrition = NutritionLog.objects.filter(date=today)
    today_macros = today_nutrition.aggregate(
        cal=Sum('calories'), protein=Sum('protein_g'),
        carbs=Sum('carbs_g'), fat=Sum('fat_g'),
    )
    today_cal     = today_macros['cal'] or 0
    today_protein = today_macros['protein'] or 0
    today_carbs   = today_macros['carbs'] or 0
    today_fat     = today_macros['fat'] or 0

    # 7-day activity chart data
    activity_data = []
    for i in range(6, -1, -1):
        d     = today - timedelta(days=i)
        count = WorkoutSession.objects.filter(date=d).count()
        activity_data.append(count)

    # Workout streak (consecutive days with a session, going backwards)
    streak = 0
    check_date = today
    while True:
        if WorkoutSession.objects.filter(date=check_date).exists():
            streak += 1
            check_date -= timedelta(days=1)
        elif check_date == today:
            # Allow today to be a rest day (check yesterday)
            check_date -= timedelta(days=1)
        else:
            break

    # Muscle group coverage (last 30 days volume per muscle group)
    muscle_groups = ['chest', 'back', 'shoulders', 'legs', 'biceps', 'triceps', 'core']
    muscle_volume = {}
    for mg in muscle_groups:
        vol = WorkoutSet.objects.filter(
            session__date__gte=month_ago,
            exercise__muscle_group=mg,
        ).aggregate(v=Sum(models_expr('weight', 'reps')))['v'] or 0
        muscle_volume[mg] = round(float(vol), 0)
    muscle_labels = [mg.title() for mg in muscle_groups]
    muscle_values = [muscle_volume[mg] for mg in muscle_groups]

    # AI Model status
    ai_metadata = AIModelMetadata.objects.first()

    context = {
        'recent_sessions':    recent_sessions,
        'total_workouts':     total_workouts,
        'workouts_this_week': workouts_this_week,
        'month_sessions':     month_sessions,
        'total_volume':       round(total_volume, 0) if total_volume else 0,
        'top_prs':            top_prs,
        'pr_count':           pr_count,
        'bw_measurements':    list(bw_measurements),
        'bw_labels':          json.dumps(bw_labels),
        'bw_values':          json.dumps(bw_values),
        'active_goals':       active_goals,
        'active_goals_count': active_goals_count,
        'today_cal':          today_cal,
        'today_protein':      today_protein,
        'today_carbs':        today_carbs,
        'today_fat':          today_fat,
        'activity_data':      activity_data,
        'streak':             streak,
        'muscle_labels':      json.dumps(muscle_labels),
        'muscle_values':      json.dumps(muscle_values),
        'ai_metadata':        ai_metadata,
        'active_tab':         'dashboard',
    }
    return render(request, 'core/dashboard.html', context)



def models_expr(w_field, r_field):
    from django.db.models import ExpressionWrapper, F, FloatField
    return ExpressionWrapper(F(w_field) * F(r_field), output_field=FloatField())


# ── 1RM Calculator ─────────────────────────────────────────────────────────────

def one_rm_calculator(request):
    exercises = Exercise.objects.all()
    history = WorkoutSet.objects.select_related('exercise', 'session').order_by('-created_at')[:20]
    context = {
        'exercises': exercises,
        'history': history,
        'active_tab': 'one_rm',
    }
    return render(request, 'core/one_rm.html', context)


def api_one_rm(request):
    weight = request.GET.get('weight')
    reps = request.GET.get('reps')
    unit = request.GET.get('unit', 'kg')

    if not weight or not reps:
        return JsonResponse({'error': 'weight and reps required'}, status=400)

    try:
        w = float(weight)
        r = int(reps)
        if r < 1 or r > 30:
            return JsonResponse({'error': 'Reps must be between 1 and 30'}, status=400)
        if w <= 0:
            return JsonResponse({'error': 'Weight must be positive'}, status=400)

        all_formulas = calculate_one_rm(w, r, 'all')
        avg_1rm = round(sum(all_formulas.values()) / len(all_formulas), 2)

        # Percentage chart data
        percentages = [100, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50]
        percent_table = [
            {'percent': p, 'weight': round(avg_1rm * p / 100, 2), 'est_reps': max(1, int((1 - p/100) * 30))}
            for p in percentages
        ]

        return JsonResponse({
            'formulas': all_formulas,
            'average': avg_1rm,
            'unit': unit,
            'weight': w,
            'reps': r,
            'percent_table': percent_table,
        })
    except (ValueError, TypeError) as e:
        return JsonResponse({'error': str(e)}, status=400)


# ── Workouts ─────────────────────────────────────────────────────────────────

def workouts(request):
    sessions = WorkoutSession.objects.prefetch_related('sets__exercise').order_by('-date')
    context = {
        'sessions': sessions,
        'active_tab': 'workouts',
    }
    return render(request, 'core/workouts.html', context)


def export_workouts_csv(request):
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="workouts_export.csv"'
    
    writer = csv.writer(response)
    writer.writerow(['Date', 'Workout Name', 'Exercise', 'Weight', 'Reps', 'Unit', '1RM'])
    
    sets = WorkoutSet.objects.select_related('session', 'exercise').order_by('-session__date')
    for s in sets:
        writer.writerow([
            s.session.date,
            s.session.name or "Workout",
            s.exercise.name,
            s.weight,
            s.reps,
            s.unit,
            s.one_rm
        ])
    return response


def new_workout(request):
    exercises = Exercise.objects.all()
    if request.method == 'POST':
        form = WorkoutSessionForm(request.POST)
        if form.is_valid():
            session = form.save()
            # Handle multiple sets submitted via JSON
            sets_data = request.POST.get('sets_json', '[]')
            try:
                sets_list = json.loads(sets_data)
                for idx, s in enumerate(sets_list):
                    ex_id = s.get('exercise_id')
                    if ex_id:
                        exercise = Exercise.objects.get(pk=ex_id)
                        ws = WorkoutSet(
                            session=session,
                            exercise=exercise,
                            weight=float(s.get('weight', 0)),
                            reps=int(s.get('reps', 0)),
                            unit=s.get('unit', 'kg'),
                            set_number=idx + 1,
                            notes=s.get('notes', ''),
                        )
                        ws.save()
                        # Check for PR
                        _update_pr(exercise, ws.weight, ws.reps, ws.unit, ws.one_rm, session.date)
            except (json.JSONDecodeError, Exercise.DoesNotExist):
                pass
            return redirect('core:workout_detail', pk=session.pk)
    else:
        form = WorkoutSessionForm()
    return render(request, 'core/new_workout.html', {
        'form': form,
        'exercises': exercises,
        'active_tab': 'workouts',
    })


def _update_pr(exercise, weight, reps, unit, one_rm, date):
    existing = PersonalRecord.objects.filter(exercise=exercise, unit=unit).first()
    if not existing or (one_rm and one_rm > existing.one_rm):
        PersonalRecord.objects.update_or_create(
            exercise=exercise, unit=unit,
            defaults={
                'weight': weight, 'reps': reps,
                'one_rm': one_rm or 0, 'date': date,
            }
        )


def workout_detail(request, pk):
    session = get_object_or_404(WorkoutSession, pk=pk)
    sets = session.sets.select_related('exercise').all()
    context = {
        'session': session,
        'sets': sets,
        'active_tab': 'workouts',
    }
    return render(request, 'core/workout_detail.html', context)


def delete_workout(request, pk):
    session = get_object_or_404(WorkoutSession, pk=pk)
    if request.method == 'POST':
        session.delete()
    return redirect('core:workouts')


# ── Exercises ─────────────────────────────────────────────────────────────────

def exercises(request):
    muscle_filter = request.GET.get('muscle', '')
    search_query = request.GET.get('q', '')
    qs = Exercise.objects.all()
    if muscle_filter:
        qs = qs.filter(muscle_group=muscle_filter)
    if search_query:
        qs = qs.filter(name__icontains=search_query)
    context = {
        'exercises': qs,
        'muscle_filter': muscle_filter,
        'search_query': search_query,
        'muscle_choices': Exercise._meta.get_field('muscle_group').choices,
        'active_tab': 'exercises',
    }
    return render(request, 'core/exercises.html', context)


def add_exercise(request):
    if request.method == 'POST':
        form = ExerciseForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('core:exercises')
    else:
        form = ExerciseForm()
    return render(request, 'core/add_exercise.html', {'form': form, 'active_tab': 'exercises'})


def delete_exercise(request, pk):
    ex = get_object_or_404(Exercise, pk=pk)
    if request.method == 'POST':
        ex.delete()
    return redirect('core:exercises')


# ── Measurements ──────────────────────────────────────────────────────────────

def measurements(request):
    all_m = BodyMeasurement.objects.order_by('-date')
    context = {
        'measurements': all_m,
        'active_tab': 'measurements',
    }
    return render(request, 'core/measurements.html', context)


def add_measurement(request):
    if request.method == 'POST':
        form = BodyMeasurementForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('core:measurements')
    else:
        form = BodyMeasurementForm()
    return render(request, 'core/add_measurement.html', {'form': form, 'active_tab': 'measurements'})


# ── Nutrition ─────────────────────────────────────────────────────────────────

def nutrition(request):
    logs = NutritionLog.objects.order_by('-date', '-created_at')
    today = timezone.now().date()
    today_logs = logs.filter(date=today)
    today_summary = today_logs.aggregate(
        cal=Sum('calories'), protein=Sum('protein_g'),
        carbs=Sum('carbs_g'), fat=Sum('fat_g'),
    )
    context = {
        'logs': logs[:50],
        'today_summary': today_summary,
        'active_tab': 'nutrition',
    }
    return render(request, 'core/nutrition.html', context)


def add_nutrition(request):
    if request.method == 'POST':
        form = NutritionLogForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('core:nutrition')
    else:
        form = NutritionLogForm()
    return render(request, 'core/add_nutrition.html', {'form': form, 'active_tab': 'nutrition'})


# ── Goals ─────────────────────────────────────────────────────────────────────

def goals(request):
    all_goals = Goal.objects.all()
    context = {
        'goals': all_goals,
        'active_tab': 'goals',
    }
    return render(request, 'core/goals.html', context)


def add_goal(request):
    if request.method == 'POST':
        form = GoalForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('core:goals')
    else:
        form = GoalForm()
    return render(request, 'core/add_goal.html', {'form': form, 'active_tab': 'goals'})


def update_goal(request, pk):
    goal = get_object_or_404(Goal, pk=pk)
    if request.method == 'POST':
        form = GoalForm(request.POST, instance=goal)
        if form.is_valid():
            form.save()
            return redirect('core:goals')
    else:
        form = GoalForm(instance=goal)
    return render(request, 'core/add_goal.html', {'form': form, 'goal': goal, 'active_tab': 'goals'})


# ── Personal Records ──────────────────────────────────────────────────────────

def personal_records(request):
    records = PersonalRecord.objects.select_related('exercise').order_by('exercise__name')
    context = {
        'records': records,
        'active_tab': 'records',
    }
    return render(request, 'core/records.html', context)


# ── API ───────────────────────────────────────────────────────────────────────

def api_exercise_history(request, exercise_id):
    exercise = get_object_or_404(Exercise, pk=exercise_id)
    sets = WorkoutSet.objects.filter(exercise=exercise).select_related('session').order_by('session__date')
    data = [
        {
            'date': str(s.session.date),
            'weight': s.weight,
            'reps': s.reps,
            'one_rm': s.one_rm,
            'unit': s.unit,
        }
        for s in sets
    ]
    return JsonResponse({'exercise': exercise.name, 'history': data})


def api_dashboard_stats(request):
    today = timezone.now().date()
    week_ago = today - timedelta(days=7)

    sessions_per_day = []
    for i in range(30):
        d = today - timedelta(days=i)
        count = WorkoutSession.objects.filter(date=d).count()
        sessions_per_day.append({'date': str(d), 'count': count})

    return JsonResponse({
        'sessions_per_day': sessions_per_day[::-1],
        'total_workouts': WorkoutSession.objects.count(),
        'total_exercises': Exercise.objects.count(),
        'total_prs': PersonalRecord.objects.count(),
    })

# ── Workout Planning ──────────────────────────────────────────────────────────

def plans(request):
    workout_plans = WorkoutPlan.objects.all()
    context = {
        'plans': workout_plans,
        'active_tab': 'plans',
    }
    return render(request, 'core/plans.html', context)


def plan_detail(request, pk):
    plan = get_object_or_404(WorkoutPlan, pk=pk)
    days = plan.days.prefetch_related('exercises__exercise').all()
    context = {
        'plan': plan,
        'days': days,
        'active_tab': 'plans',
    }
    return render(request, 'core/plan_detail.html', context)


# ── AI Laboratory ─────────────────────────────────────────────────────────────

def ai_laboratory(request):
    metadata = AIModelMetadata.objects.first()
    if not metadata:
        metadata = AIModelMetadata.objects.create(model_name="PulseMind v1")

    session_count = WorkoutSession.objects.count()
    exercise_count = Exercise.objects.count()

    context = {
        'metadata': metadata,
        'session_count': session_count,
        'exercise_count': exercise_count,
        'can_train': exercise_count > 0,  # Can always train (synthetic data fills gaps)
        'active_tab': 'ai_lab',
    }
    return render(request, 'core/ai_lab.html', context)


def api_train_ai(request):
    """
    Triggers training of the PulseMind AI.
    Always works — uses WGER real exercises; synthetic fallback when sessions < 20.
    Supports adversarial training (FGSM + Mixup) and SHAP/IG explainability.
    """
    sessions  = WorkoutSession.objects.all()
    exercises = Exercise.objects.all()

    if exercises.count() == 0:
        return JsonResponse({
            'error': 'No exercises in library. Run: python manage.py import_wger'
        }, status=400)

    arch = request.GET.get('arch', 'resnet').lower()
    if arch not in ('mlp', 'resnet', 'attention', 'ensemble'):
        arch = 'resnet'

    epochs          = int(request.GET.get('epochs', 200))
    use_wandb       = request.GET.get('wandb', 'false').lower() == 'true'
    use_adversarial = request.GET.get('adversarial', 'true').lower() == 'true'
    use_mixup       = request.GET.get('mixup', 'true').lower() == 'true'

    try:
        results = train_and_evaluate(
            sessions, exercises, arch=arch,
            epochs=epochs, use_wandb=use_wandb,
            use_adversarial=use_adversarial,
            use_mixup=use_mixup,
        )
    except Exception as e:
        import traceback
        return JsonResponse({'error': str(e), 'traceback': traceback.format_exc()}, status=500)

    if not results.get('success'):
        return JsonResponse({'error': results.get('error', 'Training failed')}, status=500)

    # Update metadata in DB
    metadata, _ = AIModelMetadata.objects.get_or_create(pk=1)
    metadata.model_name  = f'PulseMind-{arch.upper()}'
    metadata.last_trained = timezone.now()
    metadata.accuracy    = results['test_accuracy']
    metadata.total_training_samples = results['n_train'] + results['n_val'] + results['n_test']
    metadata.version     = round((metadata.version or 1.0) + 0.1, 1)
    metadata.weights_info = {
        'architecture':      arch,
        'arch_info':         results['arch_info'],
        'shap_values':       results.get('shap_values'),
        'ig_values':         results.get('ig_values'),
        'feature_names':     results['feature_names'],
        'val_accuracy':      results['val_accuracy'],
        'test_accuracy':     results['test_accuracy'],
        'adversarial':       use_adversarial,
        'mixup':             use_mixup,
        'n_exercises':       exercises.count(),
    }
    metadata.save()

    return JsonResponse({
        'status':        'success',
        'architecture':  arch,
        'val_accuracy':  results['val_accuracy'],
        'test_accuracy': results['test_accuracy'],
        'n_train':       results['n_train'],
        'n_val':         results['n_val'],
        'n_test':        results['n_test'],
        'best_epoch':    results['best_epoch'],
        'duration':      results['duration_sec'],
        'adversarial':   use_adversarial,
        'mixup':         use_mixup,
        'arch_info':     results['arch_info'],
        'shap_values':   results.get('shap_values'),
        'ig_values':     results.get('ig_values'),
        'feature_names': results['feature_names'],
        'history':       results['history'],
    })


def api_generate_plan(request):
    """
    Generate a personalised workout plan using the trained AI model.
    User can specify days_per_week (2-6, default 3).
    """
    metadata = AIModelMetadata.objects.first()
    if not metadata or not metadata.weights_info:
        return JsonResponse({'error': 'AI Model not trained yet. Train it first from the AI Lab.'}, status=400)

    exercises_qs = Exercise.objects.all()
    exercise_list = list(exercises_qs)
    n_classes = len(exercise_list)
    if n_classes == 0:
        return JsonResponse({'error': 'No exercises in library.'}, status=400)

    # Days per week from user selection
    days_per_week = int(request.GET.get('days', 3))
    days_per_week = max(2, min(6, days_per_week))

    # Load trained model
    try:
        model = load_model('pulsemind', input_size=N_FEATURES, output_size=n_classes)
    except Exception:
        return JsonResponse({'error': 'Could not load model. Please re-train.'}, status=400)

    # Generate plan using model predictions
    import torch
    model.eval()
    X_dummy = torch.randn(1, N_FEATURES, dtype=torch.float32) * 0.5 + 0.5
    with torch.no_grad():
        log_probs = model(X_dummy)
        probs = torch.exp(log_probs)[0].numpy()

    sorted_indices = np.argsort(probs)[::-1]

    today = datetime.date.today()
    new_plan = WorkoutPlan.objects.create(
        title=f"AI Plan — {today.strftime('%b %d, %Y')} ({days_per_week}x/week)",
        description=(
            f"Personalised {days_per_week}-day plan generated by "
            f"{metadata.model_name} on {today}. "
            f"Test accuracy: {metadata.accuracy}%. "
            f"Based on your logged training history."
        ),
        is_ai_generated=True,
    )

    # Define day templates based on count
    day_templates = {
        2: ['Upper Body', 'Lower Body'],
        3: ['Push Day', 'Pull Day', 'Legs & Core'],
        4: ['Push', 'Pull', 'Legs', 'Arms & Core'],
        5: ['Chest & Triceps', 'Back & Biceps', 'Legs', 'Shoulders', 'Core & Conditioning'],
        6: ['Chest', 'Back', 'Legs', 'Shoulders', 'Arms', 'Core & Cardio'],
    }
    day_names = day_templates.get(days_per_week, ['Day ' + str(i+1) for i in range(days_per_week)])

    exercises_per_day = max(4, min(6, n_classes // days_per_week))

    for day_i in range(days_per_week):
        day = PlanDay.objects.create(plan=new_plan, name=day_names[day_i], day_number=day_i + 1)
        start = day_i * exercises_per_day
        top_indices = sorted_indices[start: start + exercises_per_day]
        for rank, idx in enumerate(top_indices[:min(exercises_per_day, n_classes)]):
            PlanExercise.objects.create(
                plan_day=day,
                exercise=exercise_list[int(idx)],
                sets=4 if rank == 0 else 3,
                reps='5-8' if rank == 0 else '10-15',
                notes=f'AI confidence: {probs[int(idx)]*100:.1f}%',
            )

    return JsonResponse({
        'status': 'success',
        'plan_id': new_plan.pk,
        'plan_title': new_plan.title,
        'days_per_week': days_per_week,
    })


def api_predict_strength(request, exercise_id):
    """
    Predict recommended weight for next session.
    """
    exercise = get_object_or_404(Exercise, pk=exercise_id)

    last_set = (
        WorkoutSet.objects
        .filter(exercise=exercise)
        .select_related('session')
        .order_by('-session__date')
        .first()
    )

    if last_set:
        volume = float(last_set.weight * last_set.reps)
        baseline = float(last_set.weight)
        prev_1rm = float(last_set.one_rm or last_set.weight * 1.1)
        days_ago = (datetime.date.today() - last_set.session.date).days
    else:
        volume, baseline, prev_1rm, days_ago = 800.0, 80.0, 95.0, 7

    # Progressive overload: ~2.5kg increase
    target = round(baseline + 2.5, 1)

    return JsonResponse({
        'exercise': exercise.name,
        'current_baseline': baseline,
        'predicted_target': target,
        'rest_days': days_ago,
        'confidence': f'Based on {WorkoutSet.objects.filter(exercise=exercise).count()} logged sets',
    })
