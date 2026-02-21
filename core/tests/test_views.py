import pytest
from django.urls import reverse
from core.models import Exercise, WorkoutSession, WorkoutSet

@pytest.mark.django_db
def test_dashboard_view(client):
    url = reverse('core:dashboard')
    response = client.get(url)
    assert response.status_code == 200
    assert "Dashboard" in response.content.decode()

@pytest.mark.django_db
def test_exercises_view(client):
    Exercise.objects.create(name="Deadlift", muscle_group="back")
    url = reverse('core:exercises')
    response = client.get(url)
    assert response.status_code == 200
    assert "Deadlift" in response.content.decode()

@pytest.mark.django_db
def test_new_workout_post(client):
    exercise = Exercise.objects.create(name="Bench Press", muscle_group="chest")
    url = reverse('core:new_workout')
    data = {
        'date': '2023-10-01',
        'duration_minutes': 60,
        'sets_json': '[{"exercise_id": %d, "weight": 80, "reps": 8, "unit": "kg"}]' % exercise.pk
    }
    response = client.post(url, data)
    # Redirects to workout_detail on success
    assert response.status_code == 302
    assert WorkoutSession.objects.count() == 1
    assert WorkoutSet.objects.count() == 1
    
@pytest.mark.django_db
def test_one_rm_calculator_api(client):
    url = reverse('core:api_one_rm')
    response = client.get(url, {'weight': 100, 'reps': 10})
    assert response.status_code == 200
    data = response.json()
    assert 'average' in data
    assert data['weight'] == 100.0
    assert data['reps'] == 10
