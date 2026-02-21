from django.contrib import admin
from .models import Exercise, WorkoutSession, WorkoutSet, PersonalRecord, BodyMeasurement, NutritionLog, Goal


@admin.register(Exercise)
class ExerciseAdmin(admin.ModelAdmin):
    list_display = ['name', 'muscle_group', 'difficulty', 'is_compound']
    list_filter = ['muscle_group', 'difficulty', 'is_compound']
    search_fields = ['name']


class WorkoutSetInline(admin.TabularInline):
    model = WorkoutSet
    extra = 0


@admin.register(WorkoutSession)
class WorkoutSessionAdmin(admin.ModelAdmin):
    list_display = ['name', 'date', 'duration_minutes']
    inlines = [WorkoutSetInline]


@admin.register(PersonalRecord)
class PersonalRecordAdmin(admin.ModelAdmin):
    list_display = ['exercise', 'one_rm', 'unit', 'date']
    list_filter = ['unit']


@admin.register(BodyMeasurement)
class BodyMeasurementAdmin(admin.ModelAdmin):
    list_display = ['date', 'weight_kg', 'body_fat_percent']


@admin.register(NutritionLog)
class NutritionLogAdmin(admin.ModelAdmin):
    list_display = ['date', 'meal_name', 'calories', 'protein_g']


@admin.register(Goal)
class GoalAdmin(admin.ModelAdmin):
    list_display = ['title', 'goal_type', 'status', 'deadline']
    list_filter = ['status', 'goal_type']
