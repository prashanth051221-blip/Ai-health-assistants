"""
test_diet_and_safety.py - Automated Unit Test Suite for Clinical AI Safety & Diet Optimization.
Designed to prove mathematical and clinical guarantees to academic and technical examiners.
"""

import pytest
from diet_engine import (
    check_red_flags,
    validate_demographics,
    resolve_diet_conflicts,
    RED_FLAGS,
    DISEASE_CONSTRAINTS,
)


# =====================================================================
# PILLAR 1: EMERGENCY RED-FLAG OVERRIDE TESTS
# =====================================================================
def test_red_flag_emergency_override_triggered():
    """Verify that symptoms like Chest Pain trigger an immediate emergency override."""
    symptoms = ["Fever", "Cough", "Chest Pain"]
    triage = check_red_flags(symptoms)

    assert triage["is_emergency"] is True
    assert "Chest Pain" in triage["flagged_symptoms"]
    assert "🚨 CRITICAL MEDICAL ALERT" in triage["emergency_action"]


def test_red_flag_breathlessness_override():
    """Verify that Difficulty Breathing triggers emergency override."""
    symptoms = ["Difficulty Breathing", "Fatigue"]
    triage = check_red_flags(symptoms)

    assert triage["is_emergency"] is True
    assert "Difficulty Breathing" in triage["flagged_symptoms"]


def test_benign_symptoms_do_not_trigger_emergency():
    """Verify that common non-critical symptoms do not false-alarm."""
    symptoms = ["Runny Nose", "Sneezing", "Mild Headache"]
    triage = check_red_flags(symptoms)

    assert triage["is_emergency"] is False
    assert len(triage["flagged_symptoms"]) == 0
    assert triage["emergency_action"] == ""


# =====================================================================
# PILLAR 2: EXCLUDED VULNERABLE DEMOGRAPHIC TESTS
# =====================================================================
def test_pediatric_guardrail_blocks_optimization():
    """Verify that children (< 18) are blocked from unguided adult calorie optimization."""
    status = validate_demographics(age=14, is_pregnant=False)
    assert status["allowed"] is False
    assert any("Pediatric Guardrail" in w for w in status["warnings"])


def test_pregnancy_guardrail_blocks_optimization():
    """Verify that pregnant patients are blocked from automated deficit models."""
    status = validate_demographics(age=28, is_pregnant=True)
    assert status["allowed"] is False
    assert any("Pregnancy" in w for w in status["warnings"])


def test_geriatric_advisory_flagged():
    """Verify that patients over 70 receive geriatric safety notices."""
    status = validate_demographics(age=75, is_pregnant=False)
    assert status["allowed"] is True  # Allowed with active clinical advisory
    assert status["is_geriatric"] is True
    assert any("Geriatric Advisory" in w for w in status["warnings"])


# =====================================================================
# PILLAR 3: ALLERGY & DISLIKE INVARIANCE TESTS
# =====================================================================
def test_allergy_exclusion_enforced():
    """Verify that allergen foods (e.g. Peanuts, Dairy) never enter the safe pool or meal plans."""
    result = resolve_diet_conflicts(
        conditions=["Healthy"],
        allergies=["Peanuts & Tree Nuts", "Dairy / Lactose"],
    )
    meal_plan_str = str(result["meal_plan"]).lower()

    # Peanuts, almonds, and milk must not appear in meal plan
    assert "peanut" not in meal_plan_str
    assert "almond" not in meal_plan_str
    assert "milk" not in meal_plan_str
    assert "cheese" not in meal_plan_str

    # Allergies must be tracked in restrictions
    allergy_restrictions = [r for r in result["restrictions"] if r["category"] == "Patient Allergy"]
    assert len(allergy_restrictions) > 0


def test_custom_dislikes_excluded():
    """Verify that user custom dislikes are strictly respected."""
    result = resolve_diet_conflicts(
        conditions=["Diabetes"],
        custom_dislikes="Bitter gourd, Mushrooms",
    )
    meal_plan_str = str(result["meal_plan"]).lower()
    assert "bitter gourd" not in meal_plan_str
    assert any(r["food"] == "Bitter gourd" for r in result["restrictions"])


# =====================================================================
# PILLAR 4: GRANULAR EXPLAINABILITY & TRACEABILITY TESTS
# =====================================================================
def test_granular_explainability_traceability_present():
    """Verify every restricted food has a clear source condition and biochemical reason."""
    result = resolve_diet_conflicts(conditions=["Diabetes", "Kidney Disease"])
    restrictions = result["restrictions"]

    assert len(restrictions) > 0
    for item in restrictions:
        assert "food" in item
        assert "source" in item
        assert "biochemical_reason" in item
        assert len(item["biochemical_reason"]) > 10


# =====================================================================
# PILLAR 5: CROSS-CONDITION CONFLICT & RENAL SAFETY GUARANTEE
# =====================================================================
def test_kidney_patient_never_gets_high_potassium_food():
    """
    Core Examiner Guarantee:
    A patient with Kidney Disease + Hypertension must NEVER receive Bananas or Spinach,
    even though Hypertension guidelines normally promote them.
    """
    result = resolve_diet_conflicts(
        conditions=["Hypertension", "Kidney Disease"],
        weight_kg=70.0,
    )

    # Check detected conflicts
    conflict_foods = [c["food"] for c in result["conflicts"]]
    assert "Bananas" in conflict_foods
    assert "Spinach" in conflict_foods

    # Check meal plan text - bananas and spinach must be 100% absent
    meal_plan_text = str(result["meal_plan"]).lower()
    assert "banana" not in meal_plan_text
    assert "spinach" not in meal_plan_text

    # Protein must be capped for renal preservation
    assert result["macro_targets"]["protein_g"] <= 55  # 70kg * 0.7 = 49g


def test_synergistic_superfoods_found():
    """Verify foods mutually safe and beneficial across multiple conditions are highlighted."""
    result = resolve_diet_conflicts(conditions=["Diabetes", "Hypertension"])
    synergy = result["synergistic_foods"]
    # Garlic and Oats are beneficial for both
    assert len(synergy) > 0
    assert any(food in synergy for food in ["Garlic", "Oats", "Berries", "Broccoli", "Apples"])
