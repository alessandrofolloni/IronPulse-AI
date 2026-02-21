from django.shortcuts import render, get_object_or_404, redirect
from django.http import JsonResponse
from django.db.models import Max, Avg, Count, Sum
from django.utils import timezone
from datetime import timedelta
import json
import math

from .models import (
    Exercise, WorkoutSession, WorkoutSet,
    PersonalRecord, BodyMeasurement, NutritionLog, Goal
)
from .forms import (
    WorkoutSessionForm, WorkoutSetForm, ExerciseForm,
    BodyMeasurementForm, NutritionLogForm, GoalForm
)


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
    today = timezone.now().date()
    week_ago = today - timedelta(days=7)
    month_ago = today - timedelta(days=30)

    recent_sessions = WorkoutSession.objects.prefetch_related('sets__exercise')[:5]
    total_workouts = WorkoutSession.objects.count()
    workouts_this_week = WorkoutSession.objects.filter(date__gte=week_ago).count()
    total_volume = WorkoutSet.objects.aggregate(
        total=Sum(models_expr('weight', 'reps'))
    )['total'] or 0

    # PRs
    top_prs = PersonalRecord.objects.order_by('-one_rm')[:5]

    # Body weight trend
    bw_measurements = BodyMeasurement.objects.filter(
        weight_kg__isnull=False
    ).order_by('-date')[:10]

    # Active goals
    active_goals = Goal.objects.filter(status='active')[:4]

    # Nutrients today
    today_nutrition = NutritionLog.objects.filter(date=today)
    today_cal = today_nutrition.aggregate(s=Sum('calories'))['s'] or 0
    today_protein = today_nutrition.aggregate(s=Sum('protein_g'))['s'] or 0

    context = {
        'recent_sessions': recent_sessions,
        'total_workouts': total_workouts,
        'workouts_this_week': workouts_this_week,
        'top_prs': top_prs,
        'bw_measurements': list(bw_measurements),
        'active_goals': active_goals,
        'today_cal': today_cal,
        'today_protein': today_protein,
        'active_tab': 'dashboard',
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
    qs = Exercise.objects.all()
    if muscle_filter:
        qs = qs.filter(muscle_group=muscle_filter)
    context = {
        'exercises': qs,
        'muscle_filter': muscle_filter,
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
