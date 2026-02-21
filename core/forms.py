from django import forms
from .models import WorkoutSession, WorkoutSet, Exercise, BodyMeasurement, NutritionLog, Goal


class ExerciseForm(forms.ModelForm):
    class Meta:
        model = Exercise
        fields = ['name', 'muscle_group', 'description', 'difficulty', 'is_compound']
        widgets = {
            'description': forms.Textarea(attrs={'rows': 3}),
        }


class WorkoutSessionForm(forms.ModelForm):
    class Meta:
        model = WorkoutSession
        fields = ['name', 'date', 'notes', 'duration_minutes']
        widgets = {
            'date': forms.DateInput(attrs={'type': 'date'}),
            'notes': forms.Textarea(attrs={'rows': 3}),
        }


class WorkoutSetForm(forms.ModelForm):
    class Meta:
        model = WorkoutSet
        fields = ['exercise', 'weight', 'reps', 'unit', 'notes']


class BodyMeasurementForm(forms.ModelForm):
    class Meta:
        model = BodyMeasurement
        fields = [
            'date', 'weight_kg', 'height_cm', 'body_fat_percent',
            'chest_cm', 'waist_cm', 'hips_cm', 'bicep_cm', 'thigh_cm', 'notes'
        ]
        widgets = {
            'date': forms.DateInput(attrs={'type': 'date'}),
            'notes': forms.Textarea(attrs={'rows': 3}),
        }


class NutritionLogForm(forms.ModelForm):
    class Meta:
        model = NutritionLog
        fields = ['date', 'meal_name', 'calories', 'protein_g', 'carbs_g', 'fat_g', 'notes']
        widgets = {
            'date': forms.DateInput(attrs={'type': 'date'}),
            'notes': forms.Textarea(attrs={'rows': 2}),
        }


class GoalForm(forms.ModelForm):
    class Meta:
        model = Goal
        fields = ['title', 'goal_type', 'description', 'target_value', 'current_value', 'unit', 'deadline', 'status']
        widgets = {
            'deadline': forms.DateInput(attrs={'type': 'date'}),
            'description': forms.Textarea(attrs={'rows': 3}),
        }
