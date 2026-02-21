import pytest
from django.utils import timezone
from core.models import Exercise, WorkoutSession, WorkoutSet, PersonalRecord, BodyMeasurement, NutritionLog, Goal

@pytest.mark.django_db
def test_exercise_creation():
    exercise = Exercise.objects.create(
        name="Bench Press",
        muscle_group="chest",
        is_compound=True
    )
    assert str(exercise) == "Bench Press"
    assert exercise.is_compound is True

@pytest.mark.django_db
def test_workout_session_creation():
    session = WorkoutSession.objects.create(name="Morning Chest")
    assert "Morning Chest" in str(session)

@pytest.mark.django_db
def test_workout_set_calculation():
    session = WorkoutSession.objects.create(name="Test Session")
    exercise = Exercise.objects.create(name="Squat")
    
    # Epley formula: 1RM = weight * (1 + reps/30)
    # 100 * (1 + 10/30) = 100 * 1.333... = 133.33
    workout_set = WorkoutSet.objects.create(
        session=session,
        exercise=exercise,
        weight=100.0,
        reps=10,
        set_number=1
    )
    assert workout_set.one_rm == 133.33

@pytest.mark.django_db
def test_body_measurement_bmi():
    measurement = BodyMeasurement.objects.create(
        weight_kg=70.0,
        height_cm=175.0
    )
    # BMI = 70 / (1.75^2) = 70 / 3.0625 = 22.857...
    assert measurement.bmi() == 22.9

@pytest.mark.django_db
def test_goal_progress():
    goal = Goal.objects.create(
        title="Lose Weight",
        target_value=70.0,
        current_value=75.0
    )
    # This might depend on how progress is calculated.
    # Current code: round((self.current_value / self.target_value) * 100, 1)
    # (75 / 70) * 100 = 107.1
    assert goal.progress_percent() == 107.1
