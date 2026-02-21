from django.db import models
from django.utils import timezone


MUSCLE_GROUPS = [
    ('chest', 'Chest'),
    ('back', 'Back'),
    ('shoulders', 'Shoulders'),
    ('biceps', 'Biceps'),
    ('triceps', 'Triceps'),
    ('legs', 'Legs'),
    ('glutes', 'Glutes'),
    ('core', 'Core'),
    ('full_body', 'Full Body'),
    ('cardio', 'Cardio'),
]

DIFFICULTY_LEVELS = [
    ('beginner', 'Beginner'),
    ('intermediate', 'Intermediate'),
    ('advanced', 'Advanced'),
]

WEIGHT_UNIT = [
    ('kg', 'kg'),
    ('lbs', 'lbs'),
]


class Exercise(models.Model):
    name = models.CharField(max_length=200, db_index=True)
    muscle_group = models.CharField(max_length=50, choices=MUSCLE_GROUPS, default='chest')
    description = models.TextField(blank=True)
    difficulty = models.CharField(max_length=20, choices=DIFFICULTY_LEVELS, default='intermediate')
    is_compound = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name

    class Meta:
        ordering = ['name']


class WorkoutSession(models.Model):
    name = models.CharField(max_length=200, blank=True)
    date = models.DateField(default=timezone.now, db_index=True)
    notes = models.TextField(blank=True)
    duration_minutes = models.PositiveIntegerField(null=True, blank=True)
    # Link to a plan day if this workout was based on a planned session
    planned_day = models.ForeignKey('PlanDay', on_delete=models.SET_NULL, null=True, blank=True, related_name='executed_sessions')
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.name or 'Session'} - {self.date}"

    class Meta:
        ordering = ['-date']


class WorkoutPlan(models.Model):
    title = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    is_active = models.BooleanField(default=True)
    is_ai_generated = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title

    class Meta:
        ordering = ['-created_at']


class PlanDay(models.Model):
    plan = models.ForeignKey(WorkoutPlan, on_delete=models.CASCADE, related_name='days')
    name = models.CharField(max_length=100)
    day_number = models.PositiveIntegerField()

    class Meta:
        ordering = ['day_number']

    def __str__(self):
        return f"{self.plan.title} - {self.name}"


class PlanExercise(models.Model):
    plan_day = models.ForeignKey(PlanDay, on_delete=models.CASCADE, related_name='exercises')
    exercise = models.ForeignKey(Exercise, on_delete=models.CASCADE)
    sets = models.PositiveIntegerField(default=3)
    reps = models.CharField(max_length=50, default="8-12")
    target_weight = models.FloatField(null=True, blank=True)
    notes = models.CharField(max_length=300, blank=True)

    def __str__(self):
        return f"{self.exercise.name} in {self.plan_day.name}"


class AIModelMetadata(models.Model):
    model_name = models.CharField(max_length=100, default="PulseMind v1")
    version = models.FloatField(default=1.0)
    last_trained = models.DateTimeField(null=True, blank=True)
    accuracy = models.FloatField(null=True, blank=True)
    total_training_samples = models.PositiveIntegerField(default=0)
    weights_info = models.JSONField(null=True, blank=True) # To store simple model parameters

    def __str__(self):
        return f"{self.model_name} v{self.version}"


class WorkoutSet(models.Model):
    session = models.ForeignKey(WorkoutSession, on_delete=models.CASCADE, related_name='sets')
    exercise = models.ForeignKey(Exercise, on_delete=models.CASCADE)
    weight = models.FloatField(help_text='Weight lifted')
    reps = models.PositiveIntegerField()
    unit = models.CharField(max_length=5, choices=WEIGHT_UNIT, default='kg')
    one_rm = models.FloatField(null=True, blank=True, help_text='Calculated 1RM')
    notes = models.CharField(max_length=300, blank=True)
    set_number = models.PositiveIntegerField(default=1)
    created_at = models.DateTimeField(auto_now_add=True)

    def save(self, *args, **kwargs):
        # Epley formula: 1RM = weight * (1 + reps/30)
        if self.weight and self.reps:
            self.one_rm = round(self.weight * (1 + self.reps / 30), 2)
        super().save(*args, **kwargs)

    def __str__(self):
        return f"{self.exercise.name}: {self.weight}{self.unit} x {self.reps}"

    class Meta:
        ordering = ['set_number']


class PersonalRecord(models.Model):
    exercise = models.ForeignKey(Exercise, on_delete=models.CASCADE)
    weight = models.FloatField()
    reps = models.PositiveIntegerField()
    one_rm = models.FloatField()
    unit = models.CharField(max_length=5, choices=WEIGHT_UNIT, default='kg')
    date = models.DateField(default=timezone.now)
    notes = models.CharField(max_length=300, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"PR - {self.exercise.name}: {self.one_rm}{self.unit} 1RM"

    class Meta:
        ordering = ['-date']


class BodyMeasurement(models.Model):
    date = models.DateField(default=timezone.now)
    weight_kg = models.FloatField(null=True, blank=True, help_text='Body weight in kg')
    height_cm = models.FloatField(null=True, blank=True)
    body_fat_percent = models.FloatField(null=True, blank=True)
    chest_cm = models.FloatField(null=True, blank=True)
    waist_cm = models.FloatField(null=True, blank=True)
    hips_cm = models.FloatField(null=True, blank=True)
    bicep_cm = models.FloatField(null=True, blank=True)
    thigh_cm = models.FloatField(null=True, blank=True)
    notes = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    @property
    def bmi(self):
        if self.height_cm and self.weight_kg:
            h_m = self.height_cm / 100
            return round(self.weight_kg / (h_m ** 2), 1)
        return None

    @property
    def bmi_category(self):
        val = self.bmi # Use property
        if not val: return "none"
        if val < 18.5: return "under"
        if val < 25: return "normal"
        if val < 30: return "over"
        return "obese"

    @property
    def bmi_color(self):
        colors = {
            "under": "#60a5fa",
            "normal": "#4ade80",
            "over": "#facc15",
            "obese": "#f87171",
            "none": "inherit"
        }
        return colors.get(self.bmi_category, "inherit")

    def __str__(self):
        return f"Measurement on {self.date}"

    class Meta:
        ordering = ['-date']


class NutritionLog(models.Model):
    date = models.DateField(default=timezone.now)
    meal_name = models.CharField(max_length=200)
    calories = models.PositiveIntegerField(null=True, blank=True)
    protein_g = models.FloatField(null=True, blank=True)
    carbs_g = models.FloatField(null=True, blank=True)
    fat_g = models.FloatField(null=True, blank=True)
    notes = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.meal_name} - {self.date}"

    class Meta:
        ordering = ['-date']


class Goal(models.Model):
    GOAL_TYPES = [
        ('strength', 'Strength'),
        ('weight_loss', 'Weight Loss'),
        ('muscle_gain', 'Muscle Gain'),
        ('endurance', 'Endurance'),
        ('flexibility', 'Flexibility'),
        ('custom', 'Custom'),
    ]
    STATUS = [
        ('active', 'Active'),
        ('completed', 'Completed'),
        ('paused', 'Paused'),
    ]
    title = models.CharField(max_length=200)
    goal_type = models.CharField(max_length=50, choices=GOAL_TYPES, default='custom')
    description = models.TextField(blank=True)
    target_value = models.FloatField(null=True, blank=True)
    current_value = models.FloatField(null=True, blank=True)
    unit = models.CharField(max_length=50, blank=True)
    deadline = models.DateField(null=True, blank=True)
    status = models.CharField(max_length=20, choices=STATUS, default='active')
    created_at = models.DateTimeField(auto_now_add=True)

    def progress_percent(self):
        if self.target_value and self.current_value:
            return round((self.current_value / self.target_value) * 100, 1)
        return 0

    def __str__(self):
        return self.title

    class Meta:
        ordering = ['-created_at']
