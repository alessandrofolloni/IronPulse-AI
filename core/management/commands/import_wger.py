"""
IronPulse — WGER Exercise Database Importer
============================================
Imports real exercise data from the WGER open-source fitness API.
WGER contains 3,000+ exercises with muscle groups, equipment, and categories.

Run: python manage.py import_wger

This is a one-time setup command. It is idempotent — re-running it will
not create duplicates but will update descriptions on existing exercises.
"""

import time
from django.core.management.base import BaseCommand
from django.db import transaction

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

from core.models import Exercise

# WGER API base
WGER_BASE = "https://wger.de/api/v2"

# Map WGER category IDs to our muscle groups
CATEGORY_MAP = {
    8:  'arms',       # Arms
    10: 'back',       # Back
    11: 'legs',       # Legs
    12: 'chest',      # Chest
    13: 'shoulders',  # Shoulders
    14: 'core',       # Abdomen/Core
    15: 'legs',       # Calves
    9:  'core',       # Core
}

# WGER difficulty doesn't map 1:1, derive from equipment
# Barbell/Dumbbell → intermediate, Machine → beginner, Bodyweight → beginner/advanced
COMPOUND_BY_CATEGORY = {
    'arms': False,
    'back': True,
    'legs': True,
    'chest': True,
    'shoulders': True,
    'core': False,
    'full_body': True,
    'cardio': False,
}


class Command(BaseCommand):
    help = "Import real exercise data from the WGER open-source fitness database"

    def add_arguments(self, parser):
        parser.add_argument(
            '--language', default='2',
            help='WGER language ID (2=English, default)'
        )
        parser.add_argument(
            '--max', type=int, default=500,
            help='Maximum exercises to import (default: 500)'
        )
        parser.add_argument(
            '--overwrite', action='store_true',
            help='Overwrite existing exercises with API data'
        )

    def handle(self, *args, **options):
        if not REQUESTS_AVAILABLE:
            self.stderr.write(self.style.ERROR(
                "requests library not installed. Run: pip install requests"
            ))
            return

        lang = options['language']
        max_ex = options['max']
        overwrite = options['overwrite']

        self.stdout.write(self.style.SUCCESS(
            f"Fetching exercises from WGER (language={lang}, max={max_ex})..."
        ))

        imported = 0
        updated = 0
        skipped = 0
        errors = 0

        # Fetch exercises in pages
        url = f"{WGER_BASE}/exercise/?format=json&language={lang}&limit=100&offset=0"

        while url and imported + updated < max_ex:
            try:
                resp = requests.get(url, timeout=15)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                self.stderr.write(f"  ⚠ API error: {e}")
                errors += 1
                break

            for item in data.get('results', []):
                if imported + updated >= max_ex:
                    break

                name = (item.get('name') or '').strip()
                if not name or len(name) < 2:
                    skipped += 1
                    continue

                # Get category
                category = item.get('category', {})
                cat_id = category.get('id', 0) if category else 0
                muscle_group = CATEGORY_MAP.get(cat_id, 'full_body')

                # Description from exercise_base or empty
                description = (item.get('description') or '').strip()
                # Strip HTML tags if any
                import re
                description = re.sub(r'<[^>]+>', '', description).strip()
                if len(description) > 500:
                    description = description[:500]

                # Infer compound/isolation and difficulty
                is_compound = COMPOUND_BY_CATEGORY.get(muscle_group, False)
                difficulty = 'intermediate'

                # Check equipment to refine
                equipment = item.get('equipment', [])
                equip_ids = [e.get('id') for e in equipment] if equipment else []
                if 8 in equip_ids:   # Barbell → compound
                    is_compound = True
                    difficulty = 'intermediate'
                elif 3 in equip_ids:  # Dumbbell
                    difficulty = 'beginner'
                elif 7 in equip_ids:  # Bodyweight
                    is_compound = muscle_group in ('back', 'chest', 'legs')
                    difficulty = 'beginner'
                elif 1 in equip_ids:  # Machine
                    is_compound = False
                    difficulty = 'beginner'

                # Upsert
                try:
                    with transaction.atomic():
                        obj, created = Exercise.objects.get_or_create(
                            name=name,
                            defaults={
                                'muscle_group': muscle_group,
                                'description': description,
                                'difficulty': difficulty,
                                'is_compound': is_compound,
                            }
                        )
                        if not created and overwrite:
                            obj.muscle_group = muscle_group
                            obj.description = description
                            obj.difficulty = difficulty
                            obj.is_compound = is_compound
                            obj.save()
                            updated += 1
                        elif created:
                            imported += 1
                        else:
                            skipped += 1
                except Exception as e:
                    errors += 1

            url = data.get('next')
            if url:
                time.sleep(0.3)  # polite rate-limit

        self.stdout.write(self.style.SUCCESS(
            f"\n✅ Import complete: {imported} new, {updated} updated, "
            f"{skipped} skipped, {errors} errors"
        ))
        self.stdout.write(
            f"   Total exercises in DB: {Exercise.objects.count()}"
        )
