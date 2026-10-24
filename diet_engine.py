"""
diet_engine.py - Clinical Safety & Multi-Condition Constraint Optimization Engine
Provides deterministic, rule-based clinical safety guardrails, emergency triage,
cross-condition dietary conflict resolution, and macro optimization.
"""

from typing import List, Dict, Any, Tuple, Optional


# ==========================================
# 1. EMERGENCY RED-FLAG TRIAGE SYSTEM
# ==========================================
RED_FLAGS: Dict[str, str] = {
    "Chest Pain": "Potential acute coronary syndrome, angina, or pulmonary emergency.",
    "Difficulty Breathing": "Acute respiratory distress, pulmonary edema, or severe asthma exacerbation.",
    "Confusion": "Altered mental status indicating potential stroke, severe hypoxia, or systemic infection.",
    "Palpitations": "Cardiac arrhythmia or severe tachycardia requiring urgent clinical assessment.",
    "Blood in Urine": "Macroscopic hematuria signaling renal injury, severe infection, or acute nephropathy.",
}


def check_red_flags(selected_symptoms: List[str]) -> Dict[str, Any]:
    """
    Checks if any patient symptom matches critical red-flag emergency symptoms.
    If detected, automated ML diagnosis should be suppressed in favor of an emergency referral.
    """
    flagged = {}
    for symptom in selected_symptoms:
        if symptom in RED_FLAGS:
            flagged[symptom] = RED_FLAGS[symptom]

    return {
        "is_emergency": len(flagged) > 0,
        "flagged_symptoms": flagged,
        "emergency_action": (
            "🚨 CRITICAL MEDICAL ALERT: Immediate emergency medical evaluation required. "
            "Automated prediction is overridden for patient safety. Please contact emergency services (911 / 112) or visit the nearest emergency room immediately."
            if flagged
            else ""
        ),
    }


# ==========================================
# 2. EXCLUDED VULNERABLE DEMOGRAPHIC GUARDRAILS
# ==========================================
def validate_demographics(age: int, is_pregnant: bool = False) -> Dict[str, Any]:
    """
    Validates patient age and pregnancy status against adult nutritional models.
    Blocks or warns for pediatric, pregnant/lactating, or geriatric populations.
    """
    warnings = []
    blocked = False

    if is_pregnant:
        blocked = True
        warnings.append(
            "🤰 Pregnancy & Lactation Guardrail: Standard adult caloric/diet models are not clinically safe "
            "for maternal and fetal development. Please consult an obstetrician or registered dietitian."
        )

    if age < 18:
        blocked = True
        warnings.append(
            f"👶 Pediatric Guardrail: Patient age ({age}) is under 18. Pediatric growth curves require specialized "
            "pediatric nutritional oversight. Adult BMR/TDEE formulas do not apply."
        )
    elif age > 70:
        warnings.append(
            f"👴 Geriatric Advisory: Patient age ({age}) is over 70. Sarcopenia risk and age-related decline "
            "in glomerular filtration rate require personalized clinical supervision before strict protein restrictions."
        )

    return {
        "allowed": not blocked,
        "warnings": warnings,
        "is_geriatric": age > 70,
    }


# ==========================================
# 3. ALLERGEN TAXONOMY & INGREDIENT MAPPING
# ==========================================
ALLERGEN_MAP: Dict[str, List[str]] = {
    "Dairy / Lactose": ["Milk", "Cheese", "Yogurt", "Butter", "Curd", "Paneer", "Whey", "Cream"],
    "Peanuts & Tree Nuts": ["Peanuts", "Peanut butter", "Almonds", "Walnuts", "Brazil nuts", "Cashews", "Pistachios"],
    "Gluten / Wheat": ["Wheat bread", "White bread", "Whole wheat roti", "Barley", "Rye", "Pasta", "Semolina", "Toast"],
    "Shellfish & Seafood": ["Shrimp", "Prawns", "Crab", "Lobster", "Clams", "Oysters", "Sardines", "Salmon", "Tuna", "Fish"],
    "Eggs": ["Eggs", "Egg whites", "Egg yolks", "Mayonnaise"],
    "Soy": ["Soy", "Tofu", "Soy sauce", "Soy milk", "Edamame"],
}


# ==========================================
# 4. CLINICAL DISEASE CONSTRAINTS & BIOCHEMICAL RULES
# ==========================================
DISEASE_CONSTRAINTS: Dict[str, Dict[str, Any]] = {
    "Diabetes": {
        "promoted": ["Oats", "Bitter gourd", "Quinoa", "Beans", "Apples", "Berries", "Broccoli", "Cabbage", "Chia seeds", "Garlic", "Egg whites"],
        "forbidden": [
            ("Refined sugar", "High glycemic index causes rapid blood glucose spikes"),
            ("White bread", "Refined carbohydrates trigger sharp glycemic surges"),
            ("Sugary juices", "Fructose and liquid sugars cause insulin spikes"),
            ("Sweets", "High sucrose content exacerbates insulin resistance"),
            ("Potatoes", "High starch load causes steep glycemic response"),
        ],
        "protein_target_multiplier": 1.0,
        "carbs_cap_percent": 45,
    },
    "Kidney Disease": {
        "promoted": ["Apples", "Blueberries", "Cauliflower", "Cabbage", "Cucumbers", "Garlic", "Olive oil", "White rice", "Egg whites", "Bell peppers"],
        "forbidden": [
            ("Bananas", "High potassium; failing kidneys cannot excrete potassium, risking fatal cardiac arrhythmias"),
            ("Spinach", "High potassium and oxalate content accelerates renal failure and stone formation"),
            ("Nuts", "High phosphorus load causes renal osteodystrophy and hyperphosphatemia"),
            ("Potatoes", "Excess potassium exacerbates hyperkalemia in renal failure"),
            ("Lentils", "High phosphorus and potassium content; difficult for impaired kidneys to filter"),
            ("Dark sodas", "Loaded with synthetic inorganic phosphates that damage renal tubules"),
            ("High sodium salt", "Excess sodium causes fluid retention and accelerated nephron loss"),
        ],
        "protein_target_multiplier": 0.7,  # Strictly capped at 0.6 - 0.8 g/kg for CKD
        "potassium_restricted": True,
        "phosphorus_restricted": True,
    },
    "Hypertension": {
        "promoted": ["Bananas", "Spinach", "Beetroot", "Oats", "Garlic", "Berries", "Broccoli", "Apples", "Olive oil", "Cucumbers"],
        "forbidden": [
            ("Table salt", "Excess sodium triggers water retention and elevates arterial pressure"),
            ("Pickles", "Extremely high sodium preservatives elevate vascular resistance"),
            ("Canned soups", "High sodium content disrupts renin-angiotensin-aldosterone balance"),
            ("Processed meat", "High sodium, nitrates, and saturated fat impair endothelial function"),
            ("Alcohol", "Causes acute sympathetic nervous system stimulation and arterial hypertension"),
        ],
        "protein_target_multiplier": 1.0,
        "sodium_restricted": True,
    },
    "Coronary Artery Disease": {
        "promoted": ["Oats", "Salmon", "Avocados", "Leafy greens", "Walnuts", "Olive oil", "Berries", "Garlic", "Chia seeds", "Apples"],
        "forbidden": [
            ("Deep-fried food", "Trans fats elevate LDL cholesterol and accelerate arterial plaque"),
            ("High-fat red meat", "High saturated fats promote atheroma formation in coronary vessels"),
            ("Trans fats", "Strongly atherogenic, worsening myocardial ischemia"),
            ("Excess salt", "Causes hypervolemia and increases cardiac workload"),
            ("Butter", "High saturated animal fat elevates LDL-C"),
        ],
        "protein_target_multiplier": 1.1,
        "sat_fat_cap_percent": 7,
    },
    "Osteoarthritis": {
        "promoted": ["Salmon", "Walnuts", "Olive oil", "Berries", "Broccoli", "Ginger", "Turmeric", "Cauliflower", "Garlic"],
        "forbidden": [
            ("Deep-fried food", "Accelerates systemic inflammatory cytokines (IL-6, TNF-alpha)"),
            ("Saturated fats", "Pro-inflammatory lipids aggravate synovial joint inflammation"),
            ("Added sugars", "Triggers advanced glycation end-products (AGEs) damaging cartilage"),
            ("Alcohol", "Promotes systemic inflammation and impairs chondrocyte recovery"),
        ],
        "protein_target_multiplier": 1.1,
    },
    "Gout": {
        "promoted": ["Cherries", "Berries", "Low-fat curd", "Apples", "Cucumbers", "Olive oil", "Eggs", "Oats", "Carrots"],
        "forbidden": [
            ("Sardines", "Extremely high purine content converts into uric acid crystals"),
            ("Mackerel", "High purines trigger acute hyperuricemic gouty arthritis"),
            ("Red meat", "Purine breakdown elevates serum urate levels"),
            ("Alcohol", "Inhibits renal excretion of uric acid, triggering acute flares"),
            ("High-fructose corn syrup", "Fructose metabolism depletes ATP and spikes uric acid synthesis"),
        ],
        "protein_target_multiplier": 0.9,
    },
}


# ==========================================
# 5. MEDICATION-FOOD INTERACTION MATRIX
# ==========================================
MEDICATION_INTERACTIONS: Dict[str, Dict[str, Any]] = {
    "Blood Thinners (Warfarin)": {
        "restricted": [
            ("Spinach", "High Vitamin K counteracts Warfarin anticoagulation, triggering blood clots"),
            ("Kale", "Excess Vitamin K disrupts therapeutic International Normalized Ratio (INR)"),
            ("Cranberry juice (large amounts)", "Inhibits Warfarin metabolism, elevating bleeding hazard"),
        ],
        "clinical_note": "Maintain strictly consistent daily Vitamin K intake rather than erratic surges.",
    },
    "ACE Inhibitors / ARBs (Blood Pressure)": {
        "restricted": [
            ("Salt substitutes (Potassium Chloride)", "ACE inhibitors cause potassium retention; potassium salt triggers lethal hyperkalemia"),
            ("Excess bananas / potassium supplements", "Elevates serum potassium beyond safe renal threshold"),
        ],
        "clinical_note": "Avoid potassium chloride salt alternatives. Season with lemon, herbs, and garlic instead.",
    },
    "Statins (Cholesterol)": {
        "restricted": [
            ("Grapefruit", "Inhibits intestinal CYP3A4 enzymes, causing toxic statin accumulation in muscle tissue (rhabdomyolysis)"),
            ("Grapefruit juice", "Blocks statin breakdown, significantly increasing drug toxicity risk"),
        ],
        "clinical_note": "Grapefruit and its juice must be completely avoided when taking Atorvastatin or Simvastatin.",
    },
}


# ==========================================
# 6. CORE CONSTRAINT SOLVER & DIET RESOLVER
# ==========================================
def resolve_diet_conflicts(
    conditions: List[str],
    medications: Optional[List[str]] = None,
    allergies: Optional[List[str]] = None,
    custom_dislikes: Optional[str] = None,
    weight_kg: float = 70.0,
    height_m: float = 1.70,
    age: int = 35,
    gender: str = "Male",
    activity_level: str = "Moderate",
    goal: str = "Healthy",
    is_pregnant: bool = False,
) -> Dict[str, Any]:
    """
    Master constraint solver executing:
    1. Demographic guardrails check
    2. Cross-condition conflict detection (Hard constraints override promotion)
    3. Granular explainability (trace every restriction back to source + rationale)
    4. Synergistic superfood identification (set intersection)
    5. Caloric & Macronutrient mathematical allocation (Mifflin-St Jeor + condition bounding)
    6. Conflict-free 4-meal daily plan generation strictly from the safe pool
    """
    medications = medications or []
    allergies = allergies or []
    conditions = [c for c in conditions if c in DISEASE_CONSTRAINTS]

    # --- 1. Demographic Validation ---
    demo_status = validate_demographics(age=age, is_pregnant=is_pregnant)

    # --- 2. Build Conflict & Restriction Tracker ---
    conflicts: List[Dict[str, str]] = []
    restrictions: List[Dict[str, str]] = []
    forbidden_food_set = set()

    # Track promoted foods per condition
    promoted_by_condition: Dict[str, List[str]] = {}
    for cond in conditions:
        promoted_by_condition[cond] = DISEASE_CONSTRAINTS[cond]["promoted"]

    # Detect Cross-Condition Conflicts (Condition A promotes, Condition B forbids)
    for c_a in conditions:
        for c_b in conditions:
            if c_a == c_b:
                continue

            promoted_a = set(DISEASE_CONSTRAINTS[c_a]["promoted"])
            forbidden_b_items = {item: reason for item, reason in DISEASE_CONSTRAINTS[c_b]["forbidden"]}

            collision_items = promoted_a.intersection(forbidden_b_items.keys())
            for item in collision_items:
                reason = forbidden_b_items[item]
                conflict_entry = {
                    "food": item,
                    "promoted_by": c_a,
                    "restricted_by": c_b,
                    "reason": reason,
                    "resolution": f"Negative Safety Invariance: {c_b} restriction strictly overrides {c_a} promotion for patient safety.",
                }
                if not any(c["food"] == item and c["restricted_by"] == c_b for c in conflicts):
                    conflicts.append(conflict_entry)

    # Add all disease forbidden items to restrictions list
    for cond in conditions:
        for item, reason in DISEASE_CONSTRAINTS[cond]["forbidden"]:
            forbidden_food_set.add(item.lower())
            if not any(r["food"] == item for r in restrictions):
                restrictions.append({
                    "food": item,
                    "source": cond,
                    "biochemical_reason": reason,
                    "category": "Disease Contraindication",
                })

    # Add Medication Interactions to restrictions
    for med in medications:
        if med in MEDICATION_INTERACTIONS:
            for item, reason in MEDICATION_INTERACTIONS[med]["restricted"]:
                forbidden_food_set.add(item.lower())
                if not any(r["food"] == item for r in restrictions):
                    restrictions.append({
                        "food": item,
                        "source": med,
                        "biochemical_reason": reason,
                        "category": "Drug-Food Interaction",
                    })

    # Add Allergies to restrictions
    for allergy in allergies:
        if allergy in ALLERGEN_MAP:
            for item in ALLERGEN_MAP[allergy]:
                forbidden_food_set.add(item.lower())
                if not any(r["food"] == item for r in restrictions):
                    restrictions.append({
                        "food": item,
                        "source": f"Allergy ({allergy})",
                        "biochemical_reason": "Immunological hypersensitivity hazard",
                        "category": "Patient Allergy",
                    })

    # Add Custom Dislikes to restrictions
    if custom_dislikes:
        dislike_items = [d.strip() for d in custom_dislikes.split(",") if d.strip()]
        for d in dislike_items:
            forbidden_food_set.add(d.lower())
            restrictions.append({
                "food": d,
                "source": "Patient Dislike",
                "biochemical_reason": "Excluded based on personal preference",
                "category": "Personal Preference",
            })

    # --- 3. Compute Synergistic Superfoods ---
    # Foods promoted by at least one condition that are NOT forbidden by ANY condition/med/allergy
    safe_promoted_pool = set()
    condition_counts: Dict[str, int] = {}

    for cond, items in promoted_by_condition.items():
        for item in items:
            if not _is_food_forbidden(item, forbidden_food_set):
                safe_promoted_pool.add(item)
                condition_counts[item] = condition_counts.get(item, 0) + 1

    # Synergistic foods: promoted by multiple conditions, or highly safe staples
    synergistic_foods = [item for item, count in condition_counts.items() if count > 1]
    if not synergistic_foods:
        # Fallback to general safe promoted foods if only 1 condition selected
        synergistic_foods = list(safe_promoted_pool)[:6]

    # --- 4. Caloric & Macronutrient Mathematical Allocation ---
    # Mifflin-St Jeor Equation
    s_factor = 5 if gender == "Male" else -161
    bmr = (10 * weight_kg) + (6.25 * height_m * 100) - (5 * age) + s_factor

    activity_multipliers = {
        "Sedentary": 1.2,
        "Lightly Active": 1.375,
        "Moderate": 1.55,
        "Very Active": 1.725,
    }
    multiplier = activity_multipliers.get(activity_level, 1.375)
    tdee = bmr * multiplier

    # Adjust for goal
    if goal == "Weight Loss":
        target_calories = max(1200, round(tdee - 450))
    elif goal == "Muscle Gain":
        target_calories = round(tdee + 350)
    else:
        target_calories = round(tdee)

    # Dynamic Protein Constraint
    if "Kidney Disease" in conditions:
        protein_g = round(weight_kg * 0.7)  # Clinical renal ceiling
        protein_note = "Strictly capped at ~0.7 g/kg body weight to preserve residual renal function."
    elif goal == "Muscle Gain":
        protein_g = round(weight_kg * 1.6)
        protein_note = "Elevated to 1.6 g/kg to support hypertrophy."
    else:
        protein_g = round(weight_kg * 1.0)
        protein_note = "Balanced at 1.0 g/kg for tissue maintenance."

    # Carbohydrate & Fat Partitioning
    fat_cal = target_calories * 0.28
    fats_g = round(fat_cal / 9)

    protein_cal = protein_g * 4
    carbs_cal = target_calories - protein_cal - fat_cal
    carbs_g = max(50, round(carbs_cal / 4))

    macro_targets = {
        "calories": target_calories,
        "protein_g": protein_g,
        "carbs_g": carbs_g,
        "fats_g": fats_g,
        "bmr": round(bmr),
        "tdee": round(tdee),
        "protein_note": protein_note,
    }

    # --- 5. Generate Conflict-Free Meal Plan ---
    meal_plan = _build_conflict_free_meal_plan(
        conditions=conditions,
        forbidden_set=forbidden_food_set,
        safe_pool=safe_promoted_pool,
        target_calories=target_calories,
    )

    return {
        "demographics_status": demo_status,
        "conflicts": conflicts,
        "restrictions": restrictions,
        "synergistic_foods": sorted(synergistic_foods),
        "macro_targets": macro_targets,
        "meal_plan": meal_plan,
        "safe_foods_count": len(safe_promoted_pool),
    }


def _is_food_forbidden(food: str, forbidden_set: set) -> bool:
    """Helper to check if food contains any forbidden term."""
    food_lower = food.lower()
    for f in forbidden_set:
        if f in food_lower or food_lower in f:
            return True
    return False


def _build_conflict_free_meal_plan(
    conditions: List[str],
    forbidden_set: set,
    safe_pool: set,
    target_calories: int,
) -> Dict[str, str]:
    """
    Constructs a 4-meal daily menu ensuring zero forbidden foods appear.
    """
    is_kidney = "Kidney Disease" in conditions
    is_diabetes = "Diabetes" in conditions
    is_dairy_free = any("milk" in f or "dairy" in f for f in forbidden_set)
    is_egg_free = any("egg" in f for f in forbidden_set)

    # 1. Breakfast Selection
    if is_kidney and not is_egg_free:
        breakfast = "Egg whites scrambled with olive oil, bell peppers, and white toast (Low potassium & phosphorus)."
    elif is_diabetes:
        breakfast = "Steel-cut oats cooked with water/almond milk, topped with chia seeds and blueberries (Low GI)."
    elif not is_dairy_free:
        breakfast = "Plain Greek yogurt or curd topped with sliced apples, cinnamon, and flaxseeds."
    else:
        breakfast = "Warm oat porridge with sliced apples, chia seeds, and cinnamon."

    # 2. Lunch Selection
    if is_kidney:
        lunch = "Steamed white rice with olive oil, sauteed cauliflower, zucchini, and lean boiled chicken or egg whites."
    elif is_diabetes:
        lunch = "Steamed quinoa or brown rice with garlic-sautéed broccoli, beans, and fresh cucumber salad."
    else:
        lunch = "Mixed vegetable bowl with lentils or grilled tofu, steamed carrots, and whole wheat flatbread."

    # 3. Snack Selection
    if is_kidney:
        snack = "Crisp apple slices or fresh blueberries with cold water."
    else:
        snack = "Roasted chickpeas (chana) or apple slices paired with walnuts (if not allergic)."

    # 4. Dinner Selection
    if is_kidney:
        dinner = "Steamed cabbage and carrot stir-fry seasoned with garlic and lemon juice, served with 1 white roti or rice."
    elif is_diabetes:
        dinner = "Steamed broccoli, cauliflower, and tofu or baked fish seasoned with garlic, black pepper, and olive oil."
    else:
        dinner = "Warm vegetable soup with mixed greens, beans, and light seasoning."

    # Final sanity check: verify no forbidden keywords slipped into the menu
    clean_plan = {}
    for meal_name, text in [("Breakfast", breakfast), ("Lunch", lunch), ("Snack", snack), ("Dinner", dinner)]:
        clean_text = text
        for forbidden in forbidden_set:
            if forbidden in clean_text.lower():
                # Replace with safe generic staple
                clean_text = clean_text.replace(forbidden, "safe steamed vegetables")
        clean_plan[meal_name] = clean_text

    return clean_plan
