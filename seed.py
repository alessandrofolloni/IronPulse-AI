"""
Seed the database with popular gym exercises.
Run: python manage.py runscript seed_exercises
Or:  python seed.py
"""
import os, django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'gymapp.settings')
django.setup()

from core.models import Exercise

EXERCISES = [
    # Chest
    ("Barbell Bench Press", "chest", "beginner", True, "The king of chest exercises. Lie on a bench, lower the bar to mid-chest, press up."),
    ("Incline Barbell Bench Press", "chest", "intermediate", True, "Upper chest focus. Set bench to 30-45°."),
    ("Dumbbell Bench Press", "chest", "beginner", True, "Greater range of motion than barbell. Use dumbbells on a flat bench."),
    ("Incline Dumbbell Press", "chest", "intermediate", True, "Targets upper pecs. Better stretch at bottom."),
    ("Dumbbell Flyes", "chest", "intermediate", False, "Isolation for chest stretch. Keep slight bend in elbows."),
    ("Cable Crossover", "chest", "intermediate", False, "Constant tension chest isolation. Great for finishing sets."),
    ("Push-Up", "chest", "beginner", True, "Bodyweight classic. Keep body straight throughout."),
    ("Dips (Chest)", "chest", "intermediate", True, "Lean forward slightly to target chest over triceps."),

    # Back
    ("Deadlift", "back", "intermediate", True, "Full body compound. Pull bar from floor to hip level."),
    ("Pull-Up", "back", "intermediate", True, "Bodyweight vertical pull. Target lats and upper back."),
    ("Lat Pulldown", "back", "beginner", True, "Machine back width builder. Pull bar to upper chest."),
    ("Seated Cable Row", "back", "beginner", True, "Horizontal pull for mid-back thickness."),
    ("Barbell Row", "back", "intermediate", True, "Heavy horizontal pull. Hinge at hips, row to lower chest."),
    ("T-Bar Row", "back", "intermediate", True, "Great for mid-back thickness. Neutral grip."),
    ("Dumbbell Row", "back", "beginner", False, "Unilateral back exercise. Brace on a bench."),
    ("Face Pull", "back", "beginner", False, "Rear delt and upper back. Use rope at face height."),

    # Shoulders
    ("Barbell Overhead Press", "shoulders", "intermediate", True, "Press bar from front rack to overhead lockout."),
    ("Dumbbell Shoulder Press", "shoulders", "beginner", True, "Seated or standing. Good for shoulder health."),
    ("Lateral Raise", "shoulders", "beginner", False, "Side delt isolation. Arms slightly to side, palms down."),
    ("Front Raise", "shoulders", "beginner", False, "Front delt focus. Raise arms to shoulder height."),
    ("Arnold Press", "shoulders", "intermediate", False, "Rotating shoulder press developed by Arnold Schwarzenegger."),
    ("Rear Delt Fly", "shoulders", "beginner", False, "Bent over or on incline bench. Works rear delts."),

    # Biceps
    ("Barbell Curl", "biceps", "beginner", False, "Classic bicep builder. Keep elbows at sides."),
    ("Dumbbell Curl", "biceps", "beginner", False, "Alternate or simultaneous. Full supination at top."),
    ("Hammer Curl", "biceps", "beginner", False, "Neutral grip. Targets brachialis and forearms."),
    ("Incline Dumbbell Curl", "biceps", "intermediate", False, "Stretches long head of bicep for peak."),
    ("Cable Curl", "biceps", "beginner", False, "Constant tension throughout ROM. Great for isolation."),
    ("Preacher Curl", "biceps", "intermediate", False, "Eliminates cheating. Deep stretch at bottom."),

    # Triceps
    ("Tricep Dip", "triceps", "beginner", True, "Bodyweight or weighted. Keep close to body."),
    ("Close-Grip Bench Press", "triceps", "intermediate", True, "Compound tricep movement. Hands shoulder-width."),
    ("Tricep Pushdown", "triceps", "beginner", False, "Cable pushdown. Keep elbows tucked to sides."),
    ("Skull Crusher", "triceps", "intermediate", False, "EZ bar or dumbbells. Lower to forehead or behind head."),
    ("Overhead Tricep Extension", "triceps", "beginner", False, "Targets long head. Extend overhead with dumbbell or cable."),
    ("Diamond Push-Up", "triceps", "beginner", True, "Hands diamond-shaped under chest."),

    # Legs
    ("Barbell Squat", "legs", "intermediate", True, "The king of leg exercises. Bar on upper back, squat to parallel."),
    ("Front Squat", "legs", "advanced", True, "Bar in front rack. More quad-focused than back squat."),
    ("Romanian Deadlift", "legs", "intermediate", True, "Hip hinge movement. Targets hamstrings and glutes."),
    ("Leg Press", "legs", "beginner", True, "Machine squat alternative. Adjust foot placement."),
    ("Leg Extension", "legs", "beginner", False, "Quad isolation machine. Full extension at top."),
    ("Leg Curl", "legs", "beginner", False, "Hamstring isolation. Prone or seated variation."),
    ("Bulgarian Split Squat", "legs", "intermediate", True, "Rear foot elevated lunges. Targets quads and glutes."),
    ("Walking Lunges", "legs", "beginner", True, "Dynamic lunge variation for overall leg development."),
    ("Hack Squat", "legs", "intermediate", True, "Machine quad-dominant squat. Keep chest up."),
    ("Calf Raise", "legs", "beginner", False, "Standing or seated. Full ROM for gastrocnemius."),

    # Glutes
    ("Hip Thrust", "glutes", "beginner", True, "Barbell on hips, drive through heels. Best glute exercise."),
    ("Glute Bridge", "glutes", "beginner", True, "Bodyweight version of hip thrust."),
    ("Cable Kickback", "glutes", "beginner", False, "Extend leg back against cable resistance."),

    # Core
    ("Plank", "core", "beginner", False, "Hold straight body position. Build anti-extension strength."),
    ("Cable Crunch", "core", "beginner", False, "Kneel facing cable, crunch downward."),
    ("Ab Wheel Rollout", "core", "intermediate", False, "Advanced core stability exercise."),
    ("Hanging Leg Raise", "core", "intermediate", False, "Hang from bar, raise legs to horizontal or vertical."),
    ("Russian Twist", "core", "beginner", False, "Seated, twist torso side to side with weight."),
    ("Dead Bug", "core", "beginner", False, "Opposite arm/leg extensions on back. Safe for spine."),

    # Cardio / Full Body
    ("Barbell Power Clean", "full_body", "advanced", True, "Olympic lift. Explosively pull bar from floor to front rack."),
    ("Kettlebell Swing", "full_body", "intermediate", True, "Hip hinge power exercise. Drive hips through."),
    ("Burpee", "cardio", "beginner", True, "Full body cardio. Squat, push-up, jump sequence."),
    ("Box Jump", "cardio", "intermediate", True, "Explosive plyometric. Jump onto box, land softly."),
    ("Battle Ropes", "cardio", "beginner", True, "Upper body cardio. Alternate or double wave patterns."),
    ("Treadmill Running", "cardio", "beginner", False, "Steady state or interval running on treadmill."),
    ("Rowing Machine", "cardio", "beginner", True, "Full body cardio. Legs first, then back, then arms."),
]

created = 0
for name, muscle, difficulty, compound, desc in EXERCISES:
    obj, is_new = Exercise.objects.get_or_create(
        name=name,
        defaults={
            'muscle_group': muscle,
            'difficulty': difficulty,
            'is_compound': compound,
            'description': desc,
        }
    )
    if is_new:
        created += 1

print(f"✅ Seeded {created} exercises ({Exercise.objects.count()} total in DB)")
