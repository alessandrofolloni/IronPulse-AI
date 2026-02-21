from django.urls import path
from . import views

app_name = 'core'

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('one-rm/', views.one_rm_calculator, name='one_rm'),
    path('workouts/', views.workouts, name='workouts'),
    path('workouts/export-csv/', views.export_workouts_csv, name='export_workouts_csv'),
    path('workouts/new/', views.new_workout, name='new_workout'),
    path('workouts/<int:pk>/', views.workout_detail, name='workout_detail'),
    path('workouts/<int:pk>/delete/', views.delete_workout, name='delete_workout'),
    path('exercises/', views.exercises, name='exercises'),
    path('exercises/add/', views.add_exercise, name='add_exercise'),
    path('exercises/<int:pk>/delete/', views.delete_exercise, name='delete_exercise'),
    path('measurements/', views.measurements, name='measurements'),
    path('measurements/add/', views.add_measurement, name='add_measurement'),
    path('nutrition/', views.nutrition, name='nutrition'),
    path('nutrition/add/', views.add_nutrition, name='add_nutrition'),
    path('goals/', views.goals, name='goals'),
    path('goals/add/', views.add_goal, name='add_goal'),
    path('goals/<int:pk>/update/', views.update_goal, name='update_goal'),
    path('records/', views.personal_records, name='records'),
    path('plans/', views.plans, name='plans'),
    path('plans/<int:pk>/', views.plan_detail, name='plan_detail'),
    path('ai-lab/', views.ai_laboratory, name='ai_lab'),
    # API endpoints
    path('api/one-rm/', views.api_one_rm, name='api_one_rm'),
    path('api/exercise-history/<int:exercise_id>/', views.api_exercise_history, name='api_exercise_history'),
    path('api/dashboard-stats/', views.api_dashboard_stats, name='api_dashboard_stats'),
    path('api/ai/train/', views.api_train_ai, name='api_train_ai'),
    path('api/ai/generate-plan/', views.api_generate_plan, name='api_generate_plan'),
]
