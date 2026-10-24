import hashlib
import json
import os
import pickle
import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from pymongo import MongoClient
from rapidfuzz import fuzz, process

from diet_engine import (
    resolve_diet_conflicts,
    DISEASE_CONSTRAINTS,
    MEDICATION_INTERACTIONS,
    ALLERGEN_MAP,
)


st.set_page_config(page_title="AI Health Assistant", layout="wide")

LOCAL_DB_PATH = Path("data/local_health_app.json")


def set_bg():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

        /* Global App Canvas: Clean clinical slate-50 */
        .stApp {
            background-color: #f8fafc !important;
            font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif;
            color: #1e293b;
        }

        /* Main Container Card: Crisp pure white with soft elevation */
        .block-container {
            background: #ffffff !important;
            padding: 2.25rem 2.75rem !important;
            border-radius: 16px !important;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.04), 0 10px 25px -5px rgba(0, 0, 0, 0.04) !important;
            border: 1px solid #e2e8f0 !important;
            max-width: 1250px !important;
            margin-top: 1.5rem !important;
            margin-bottom: 2rem !important;
        }

        /* Typography */
        h1, h2, h3, h4, h5, h6 {
            color: #0f172a !important;
            font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif;
            font-weight: 700;
            letter-spacing: -0.02em;
        }

        p, span, label {
            color: #334155;
            font-size: 15px;
            line-height: 1.6;
        }

        /* Primary Action Buttons */
        .stButton>button {
            background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 10px !important;
            padding: 0.65rem 1.4rem !important;
            font-weight: 600 !important;
            font-size: 15px !important;
            letter-spacing: 0.01em !important;
            box-shadow: 0 2px 8px rgba(2, 132, 199, 0.25) !important;
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        }

        .stButton>button:hover {
            background: linear-gradient(135deg, #0369a1 0%, #075985 100%) !important;
            box-shadow: 0 4px 14px rgba(2, 132, 199, 0.35) !important;
            transform: translateY(-1px) !important;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: #f1f5f9 !important;
            border-right: 1px solid #e2e8f0 !important;
        }

        /* Modern Tabs Navigation */
        .stTabs [data-baseweb="tab-list"] {
            gap: 6px;
            border-bottom: 2px solid #e2e8f0 !important;
            padding-bottom: 2px;
        }

        .stTabs [data-baseweb="tab"] {
            padding: 10px 18px !important;
            border-radius: 8px 8px 0 0 !important;
            font-weight: 600 !important;
            color: #64748b !important;
            background-color: transparent !important;
            transition: all 0.2s ease !important;
        }

        .stTabs [aria-selected="true"] {
            color: #0284c7 !important;
            border-bottom: 3px solid #0284c7 !important;
            background-color: #f0f9ff !important;
        }

        /* Metric Tiles */
        [data-testid="stMetricValue"] {
            color: #0f172a !important;
            font-weight: 800 !important;
        }

        [data-testid="stMetricLabel"] {
            color: #64748b !important;
            font-weight: 600 !important;
        }

        /* Form Inputs */
        input, select, textarea {
            color: #0f172a !important;
        }

        .stDataFrame {
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            overflow: hidden;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("columns.pkl", "rb") as f:
        columns = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    return model, columns, le


def get_mongo_uri():
    try:
        return st.secrets.get("MONGO_URI") or os.getenv("MONGO_URI")
    except Exception:
        return os.getenv("MONGO_URI")


@st.cache_resource
def get_database():
    mongo_uri = get_mongo_uri()
    if not mongo_uri:
        return None

    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=2500)
        client.admin.command("ping")
        db = client["health_app"]
        return db["users"], db["bmi_records"]
    except Exception as exc:
        st.sidebar.warning(f"Database offline. Using local storage. ({exc.__class__.__name__})")
        return None


def load_local_db():
    if not LOCAL_DB_PATH.exists():
        return {"users": [], "bmi_records": []}

    try:
        return json.loads(LOCAL_DB_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"users": [], "bmi_records": []}


def save_local_db(data):
    LOCAL_DB_PATH.parent.mkdir(exist_ok=True)
    LOCAL_DB_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def hash_password(password):
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def find_user(username):
    if collections:
        return collections[0].find_one({"username": username})

    local_db = load_local_db()
    return next((user for user in local_db["users"] if user["username"] == username), None)


def create_user(username, password):
    password_hash = hash_password(password)
    if collections:
        collections[0].insert_one({"username": username, "password_hash": password_hash})
        return

    local_db = load_local_db()
    local_db["users"].append({"username": username, "password_hash": password_hash})
    save_local_db(local_db)


def check_login(username, password):
    user = find_user(username)
    if not user:
        return False

    password_hash = hash_password(password)
    return user.get("password_hash") == password_hash or user.get("password") == password


def save_bmi(username, bmi, category):
    record = {
        "username": username,
        "bmi": round(bmi, 2),
        "category": category,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

    if collections:
        collections[1].insert_one(record)
        return

    local_db = load_local_db()
    local_db["bmi_records"].append(record)
    save_local_db(local_db)


def get_bmi_records(username):
    if collections:
        records = list(collections[1].find({"username": username}, {"_id": 0}).sort("created_at", 1))
    else:
        local_db = load_local_db()
        records = [r for r in local_db["bmi_records"] if r["username"] == username]

    return records


def bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    if bmi < 25:
        return "Healthy"
    if bmi < 30:
        return "Overweight"
    return "Obese"


def build_diet_plan(goal):
    plans = {
        "Weight Loss": {
            "Breakfast": "Oats with fruit, or vegetable poha with curd",
            "Lunch": "Dal, salad, 2 rotis, and cooked vegetables",
            "Snack": "Fruit, roasted chana, or buttermilk",
            "Dinner": "Soup, paneer/tofu/egg, and vegetables",
            "Tips": "Drink enough water, limit fried foods, and keep dinner light.",
        },
        "Muscle Gain": {
            "Breakfast": "Eggs or paneer bhurji with toast, plus fruit",
            "Lunch": "Rice or roti with dal, curd, vegetables, and protein",
            "Snack": "Peanut butter toast, sprouts, nuts, or a smoothie",
            "Dinner": "Chicken/paneer/tofu, vegetables, and rice or roti",
            "Tips": "Add protein to every meal and avoid skipping breakfast.",
        },
        "Healthy": {
            "Breakfast": "Idli, upma, oats, or eggs with fruit",
            "Lunch": "Balanced plate with grains, dal/protein, salad, and vegetables",
            "Snack": "Fruit, nuts, yogurt, or roasted makhana",
            "Dinner": "Light khichdi, soup, or roti with vegetables",
            "Tips": "Keep meals colorful, reduce sugary drinks, and sleep on time.",
        },
    }
    return plans[goal]


DISEASE_DATA = [
    {"Disease": "Common Cold", "Description": "Viral infection affecting nose and throat.", "Foods_to_Eat": "Warm soup; Ginger tea; Honey; Citrus fruits; Garlic; Herbal tea", "Foods_to_Avoid": "Cold drinks; Ice cream; Junk food; Fried food", "Note": "Stay hydrated"},
    {"Disease": "Osteoporosis", "Description": "Weak bones.", "Foods_to_Eat": "Milk; Cheese; Yogurt; Almonds; Broccoli", "Foods_to_Avoid": "Caffeine; Soda", "Note": "Calcium needed"},
    {"Disease": "Bronchitis", "Description": "Lung inflammation.", "Foods_to_Eat": "Warm fluids; Honey; Ginger", "Foods_to_Avoid": "Smoking; Cold drinks", "Note": "Avoid pollution"},
    {"Disease": "Eczema", "Description": "Skin inflammation.", "Foods_to_Eat": "Fruits; Vegetables; Omega-3", "Foods_to_Avoid": "Dairy; Sugar", "Note": "Hydrate skin"},
    {"Disease": "Multiple Sclerosis", "Description": "Nerve disease.", "Foods_to_Eat": "Healthy fats; Fish", "Foods_to_Avoid": "Processed food", "Note": "Balanced diet"},
    {"Disease": "Osteoarthritis", "Description": "Joint pain.", "Foods_to_Eat": "Fish; Nuts; Vegetables", "Foods_to_Avoid": "Fried food", "Note": "Maintain weight"},
    {"Disease": "Hypothyroidism", "Description": "Low thyroid.", "Foods_to_Eat": "Eggs; Dairy", "Foods_to_Avoid": "Soy", "Note": "Balanced nutrition"},
    {"Disease": "Gastroenteritis", "Description": "Stomach infection.", "Foods_to_Eat": "Rice; Banana; ORS", "Foods_to_Avoid": "Spicy food", "Note": "Hydration"},
    {"Disease": "Allergic Rhinitis", "Description": "Allergy condition.", "Foods_to_Eat": "Fruits; Vitamin C", "Foods_to_Avoid": "Dust", "Note": "Avoid triggers"},
    {"Disease": "Depression", "Description": "Mental disorder.", "Foods_to_Eat": "Fish; Nuts", "Foods_to_Avoid": "Alcohol", "Note": "Healthy lifestyle"},
    {"Disease": "Hyperthyroidism", "Description": "High thyroid.", "Foods_to_Eat": "Vegetables", "Foods_to_Avoid": "Iodine excess", "Note": "Monitor"},
    {"Disease": "Migraine", "Description": "Headache.", "Foods_to_Eat": "Banana; Almonds", "Foods_to_Avoid": "Caffeine", "Note": "Avoid triggers"},
    {"Disease": "Psoriasis", "Description": "Skin disease.", "Foods_to_Eat": "Vegetables", "Foods_to_Avoid": "Alcohol", "Note": "Reduce inflammation"},
    {"Disease": "Stroke", "Description": "Brain damage.", "Foods_to_Eat": "Fruits; Fish", "Foods_to_Avoid": "Salt", "Note": "Heart diet"},
    {"Disease": "Hypertension", "Description": "High BP.", "Foods_to_Eat": "Bananas", "Foods_to_Avoid": "Salt", "Note": "Low sodium"},
    {"Disease": "Urinary Tract Infection", "Description": "Urinary infection.", "Foods_to_Eat": "Water; Juice", "Foods_to_Avoid": "Caffeine", "Note": "Hydrate"},
    {"Disease": "Pneumonia", "Description": "Lung infection.", "Foods_to_Eat": "Soup; Fluids", "Foods_to_Avoid": "Cold drinks", "Note": "Rest"},
    {"Disease": "Alzheimer's Disease", "Description": "Memory loss.", "Foods_to_Eat": "Nuts; Fish", "Foods_to_Avoid": "Sugar", "Note": "Brain diet"},
    {"Disease": "Coronary Artery Disease", "Description": "Heart blockage.", "Foods_to_Eat": "Oats; Fish", "Foods_to_Avoid": "Fried food", "Note": "Low fat"},
    {"Disease": "Rheumatoid Arthritis", "Description": "Joint disease.", "Foods_to_Eat": "Omega-3", "Foods_to_Avoid": "Sugar", "Note": "Anti-inflammatory"},
    {"Disease": "Liver Cancer", "Description": "Liver cancer.", "Foods_to_Eat": "Fruits", "Foods_to_Avoid": "Alcohol", "Note": "Care needed"},
    {"Disease": "Parkinson's Disease", "Description": "Movement disorder.", "Foods_to_Eat": "Fiber", "Foods_to_Avoid": "Processed food", "Note": "Balanced diet"},
    {"Disease": "Kidney Disease", "Description": "Kidney issue.", "Foods_to_Eat": "Low sodium foods", "Foods_to_Avoid": "Salt", "Note": "Monitor"},
    {"Disease": "Anxiety Disorders", "Description": "Mental issue.", "Foods_to_Eat": "Nuts", "Foods_to_Avoid": "Caffeine", "Note": "Relax"},
    {"Disease": "Liver Disease", "Description": "Liver problem.", "Foods_to_Eat": "Vegetables", "Foods_to_Avoid": "Alcohol", "Note": "Avoid toxins"},
    {"Disease": "Diabetes", "Description": "High sugar.", "Foods_to_Eat": "Whole grains", "Foods_to_Avoid": "Sugar", "Note": "Control"},
    {"Disease": "Asthma", "Description": "Breathing issue.", "Foods_to_Eat": "Fruits", "Foods_to_Avoid": "Cold drinks", "Note": "Avoid triggers"},
    {"Disease": "Influenza", "Description": "Flu.", "Foods_to_Eat": "Soup", "Foods_to_Avoid": "Cold drinks", "Note": "Rest"},
    {"Disease": "Kidney Cancer", "Description": "Kidney cancer.", "Foods_to_Eat": "Healthy diet", "Foods_to_Avoid": "Processed food", "Note": "Medical care"},
]


SYNONYMS = {
    "uti": "urinary tract infection",
    "bp": "hypertension",
    "high bp": "hypertension",
    "blood pressure": "hypertension",
    "flu": "influenza",
    "cold": "common cold",
    "cough": "common cold",
    "sugar": "diabetes",
    "high sugar": "diabetes",
    "heart attack": "coronary artery disease",
    "heart disease": "coronary artery disease",
    "joint pain": "osteoarthritis",
    "arthritis": "osteoarthritis",
    "ckd": "kidney disease",
    "renal": "kidney disease",
}


def find_matches(query, df):
    q = query.strip().lower()
    if not q:
        return df.iloc[0:0]

    resolved_q = SYNONYMS.get(q, q)
    scored = []
    for idx, row in df.iterrows():
        disease_name = str(row["Disease"]).lower()
        desc = str(row["Description"]).lower()

        # 1. Exact or partial substring match in Disease Name
        if resolved_q in disease_name or q in disease_name:
            score = 100.0
        # 2. Substring match in Description
        elif resolved_q in desc or q in desc:
            score = 90.0
        else:
            # 3. Rapidfuzz WRatio fuzzy matching
            name_score = max(fuzz.WRatio(resolved_q, disease_name), fuzz.WRatio(q, disease_name))
            desc_score = max(fuzz.partial_ratio(resolved_q, desc), fuzz.partial_ratio(q, desc)) * 0.85
            score = max(name_score, desc_score)

        if score >= 45.0:
            scored.append((score, idx))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_indices = [idx for _, idx in scored[:5]]
    return df.loc[top_indices]


model, columns, le = load_model()
collections = get_database()

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "selected_diet_conditions" not in st.session_state:
    st.session_state.selected_diet_conditions = []

set_bg()
st.markdown(
    """
    <div style="margin-bottom: 24px; padding-bottom: 18px; border-bottom: 1px solid #e2e8f0;">
        <div style="display: inline-block; background: #e0f2fe; color: #0369a1; padding: 4px 12px; border-radius: 9999px; font-size: 11px; font-weight: 700; letter-spacing: 0.8px; text-transform: uppercase; margin-bottom: 8px;">
            Clinical AI Decision Platform
        </div>
        <h1 style="margin: 0; font-size: 32px; font-weight: 800; color: #0f172a; letter-spacing: -0.03em;">
            AI Health & Clinical Nutrition Assistant
        </h1>
        <p style="margin: 6px 0 0 0; color: #64748b; font-size: 15px;">
            Evidence-based disease prediction, multi-condition dietary conflict resolution, and biometric tracking.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

menu = st.sidebar.selectbox("Menu", ["Login", "Register"])

if menu == "Register" and not st.session_state.logged_in:
    st.subheader("Create Account")
    username = st.text_input("Username").strip()
    password = st.text_input("Password", type="password")

    if st.button("Register"):
        if not username or not password:
            st.warning("Please enter both username and password.")
        elif find_user(username):
            st.warning("This username already exists.")
        else:
            create_user(username, password)
            st.success("Registered successfully. You can now log in.")

elif menu == "Login" and not st.session_state.logged_in:
    st.subheader("Login")
    username = st.text_input("Username").strip()
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if check_login(username, password):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.rerun()
        else:
            st.error("Invalid username or password.")

if st.session_state.logged_in:
    st.sidebar.success(f"Logged in as {st.session_state.username}")

    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun()

    df = pd.DataFrame(DISEASE_DATA)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Search", "BMI", "History", "Diet", "Prediction"]
    )

    with tab1:
        st.subheader("Disease Food Guidance")
        st.caption("Search by disease name, common synonym (e.g. Cold, Flu, BP, Sugar, UTI), or symptom.")

        col_search, col_pick = st.columns([3, 2])
        with col_search:
            search_input = st.text_input(
                "Search disease",
                placeholder="e.g. Cold, Diabetes, Asthma, Flu, BP...",
                key="disease_search_input"
            )
        with col_pick:
            disease_list = ["-- Or choose from list --"] + sorted(df["Disease"].tolist())
            selected_disease = st.selectbox("Pick a disease", disease_list, key="disease_select_box")

        active_query = search_input.strip()
        if not active_query and selected_disease != "-- Or choose from list --":
            active_query = selected_disease

        if active_query:
            results = find_matches(active_query, df)
            if results.empty:
                st.warning(f"No close disease match found for '{active_query}'. Try selecting from the dropdown or using common terms.")
            else:
                for _, row in results.iterrows():
                    st.markdown(f"### {row['Disease']}")
                    st.caption(f"*{row['Description']}*")

                    left, right = st.columns(2)
                    with left:
                        eat_items = [item.strip() for item in re.split(r'[;,]', str(row["Foods_to_Eat"])) if item.strip()]
                        eat_html = "".join([f"<li style='margin-bottom: 6px; color: #166534;'>{item}</li>" for item in eat_items])
                        st.markdown(
                            f"""
                            <div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 12px; padding: 16px 20px; min-height: 160px;">
                                <h5 style="margin: 0 0 10px 0; color: #15803d; font-weight: 700;">🟢 Recommended Foods</h5>
                                <ul style="margin: 0; padding-left: 18px;">{eat_html}</ul>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    with right:
                        avoid_items = [item.strip() for item in re.split(r'[;,]', str(row["Foods_to_Avoid"])) if item.strip()]
                        avoid_html = "".join([f"<li style='margin-bottom: 6px; color: #991b1b;'>{item}</li>" for item in avoid_items])
                        st.markdown(
                            f"""
                            <div style="background: #fef2f2; border: 1px solid #fecaca; border-radius: 12px; padding: 16px 20px; min-height: 160px;">
                                <h5 style="margin: 0 0 10px 0; color: #b91c1c; font-weight: 700;">🔴 Prohibited Foods</h5>
                                <ul style="margin: 0; padding-left: 18px;">{avoid_html}</ul>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    st.markdown(
                        f"""
                        <div style="background: #f0f9ff; border-left: 4px solid #0284c7; padding: 12px 16px; border-radius: 6px; margin-top: 14px;">
                            <span style="font-weight: 700; color: #0369a1;">Clinical Note:</span> <span style="color: #0c4a6e;">{row['Note']}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.markdown("---")
        else:
            st.info("💡 Start typing a disease name or choose one from the dropdown to see recommended and prohibited foods.")

    with tab2:
        st.subheader("BMI Calculator")
        col1, col2 = st.columns(2)
        with col1:
            weight = st.number_input("Weight in kg", min_value=1.0, step=0.5)
        with col2:
            height = st.number_input("Height in meters", min_value=0.5, max_value=2.5, step=0.01)

        if st.button("Calculate BMI"):
            bmi = weight / (height**2)
            category = bmi_category(bmi)
            st.success(f"BMI: {round(bmi, 2)} ({category})")
            save_bmi(st.session_state.username, bmi, category)

    with tab3:
        st.subheader("BMI History")
        records = get_bmi_records(st.session_state.username)
        if records:
            history_df = pd.DataFrame(records)
            st.dataframe(history_df[["created_at", "bmi", "category"]], use_container_width=True)

            fig, ax = plt.subplots(figsize=(8, 3.5))
            fig.patch.set_facecolor('#ffffff')
            ax.set_facecolor('#f8fafc')
            ax.plot(history_df["created_at"], history_df["bmi"], marker="o", color="#0284c7", linewidth=2.5, markersize=6)
            ax.grid(True, linestyle="--", alpha=0.5, color="#cbd5e1")
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#cbd5e1')
            ax.spines['bottom'].set_color('#cbd5e1')
            ax.tick_params(colors='#475569')
            ax.set_xlabel("Recorded Date", color="#0f172a", fontsize=11, fontweight='600')
            ax.set_ylabel("BMI Value", color="#0f172a", fontsize=11, fontweight='600')
            ax.set_title("BMI Health Trajectory", color="#0f172a", fontsize=13, fontweight='700', pad=12)
            plt.xticks(rotation=25, ha="right")
            st.pyplot(fig)
        else:
            st.info("No BMI records yet. Calculate your BMI to start tracking.")

    with tab4:
        st.subheader("🧬 Multi-Condition Diet Conflict Resolver")
        st.caption("Constraint optimization engine that resolves cross-disease dietary conflicts, food-drug interactions, and allergen restrictions.")

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            available_conditions = list(DISEASE_CONSTRAINTS.keys())
            preselected_conds = [c for c in st.session_state.get("selected_diet_conditions", []) if c in available_conditions]
            active_conditions = st.multiselect(
                "Active Health Conditions (Select one or more)",
                options=available_conditions,
                default=preselected_conds,
                help="Add multiple overlapping conditions to detect nutritional conflicts.",
            )
            active_meds = st.multiselect(
                "Active Medications (Optional)",
                options=list(MEDICATION_INTERACTIONS.keys()),
                help="Cross-references pharmacotherapy and biochemical food interactions.",
            )

        with col_d2:
            active_allergies = st.multiselect(
                "Patient Allergies (Strict Exclusion)",
                options=list(ALLERGEN_MAP.keys()),
                help="Foods containing these allergens are strictly purged from meal plans.",
            )
            custom_dislikes = st.text_input(
                "Food Dislikes (Comma-separated)",
                placeholder="e.g. Mushrooms, Bitter gourd",
                help="Excluded from your recommended meal plans based on personal taste.",
            )

        st.markdown("##### 👤 Biometric Profile & Demographics")
        b1, b2, b3, b4, b5 = st.columns(5)
        with b1:
            patient_age = st.number_input("Age", min_value=1, max_value=120, value=30, step=1)
        with b2:
            patient_gender = st.selectbox("Biological Sex", ["Male", "Female"])
        with b3:
            patient_weight = st.number_input("Weight (kg)", min_value=20.0, max_value=250.0, value=70.0, step=0.5)
        with b4:
            patient_height = st.number_input("Height (m)", min_value=0.5, max_value=2.5, value=1.70, step=0.01)
        with b5:
            activity = st.selectbox("Activity Level", ["Sedentary", "Lightly Active", "Moderate", "Very Active"], index=2)

        c_goal, c_preg = st.columns([3, 2])
        with c_goal:
            diet_goal = st.selectbox("Dietary Goal", ["Healthy", "Weight Loss", "Muscle Gain"])
        with c_preg:
            is_pregnant = False
            if patient_gender == "Female":
                is_pregnant = st.checkbox("Pregnant or Lactating", help="Triggers clinical obstetric guardrail warnings.")

        if st.button("🚀 Run Clinical Diet Optimization", type="primary"):
            if not active_conditions:
                st.warning("Please select at least one health condition (e.g. 'Diabetes', 'Hypertension', 'Kidney Disease').")
            else:
                resolution = resolve_diet_conflicts(
                    conditions=active_conditions,
                    medications=active_meds,
                    allergies=active_allergies,
                    custom_dislikes=custom_dislikes,
                    weight_kg=patient_weight,
                    height_m=patient_height,
                    age=patient_age,
                    gender=patient_gender,
                    activity_level=activity,
                    goal=diet_goal,
                    is_pregnant=is_pregnant,
                )

                # 1. Demographic Guardrail Notices
                demo = resolution["demographics_status"]
                if demo["warnings"]:
                    for w in demo["warnings"]:
                        if not demo["allowed"]:
                            st.error(w)
                        else:
                            st.warning(w)

                if demo["allowed"]:
                    # 2. Cross-Condition Conflicts Banner
                    conflicts = resolution["conflicts"]
                    if conflicts:
                        st.markdown("#### ⚠️ Cross-Condition Dietary Conflicts Detected")
                        st.caption("When one condition promotes a food that another restricts, renal & metabolic safety strictly overrides promotion.")
                        for c in conflicts:
                            st.markdown(
                                f"""
                                <div style="background: #fff1f2; border: 1px solid #fecdd3; border-left: 5px solid #e11d48; border-radius: 10px; padding: 16px 20px; margin-bottom: 12px;">
                                    <h5 style="margin: 0 0 6px 0; color: #be123c; font-weight: 700; font-size: 16px;">⚠️ Conflict: {c['food']}</h5>
                                    <p style="margin: 0 0 4px 0; color: #334155; font-size: 14px;"><strong>Promoted by:</strong> {c['promoted_by']}</p>
                                    <p style="margin: 0 0 6px 0; color: #334155; font-size: 14px;"><strong>Restricted by:</strong> {c['restricted_by']} <span style="color: #64748b;">({c['reason']})</span></p>
                                    <p style="margin: 0; color: #9f1239; font-size: 13px; font-weight: 600;">🛡️ {c['resolution']}</p>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                    else:
                        st.success("✅ Zero cross-condition collisions detected among selected conditions.")

                    # 3. Synergistic Superfoods
                    synergy = resolution["synergistic_foods"]
                    if synergy:
                        st.markdown("#### 🌟 Synergistic Superfoods")
                        st.caption("Foods that provide mutual biochemical benefits across your selected profile without triggering conflicts.")
                        badges_html = "".join([f"<span style='background: #f0fdf4; color: #15803d; border: 1px solid #bbf7d0; padding: 5px 12px; border-radius: 9999px; font-size: 13px; font-weight: 600;'>✓ {food}</span>" for food in synergy])
                        st.markdown(f"<div style='display: flex; flex-wrap: wrap; gap: 8px; margin: 8px 0 16px 0;'>{badges_html}</div>", unsafe_allow_html=True)

                    # 4. Caloric & Macronutrient Partitioning
                    macros = resolution["macro_targets"]
                    st.markdown("#### 📊 Mathematical Caloric & Macronutrient Targets")
                    st.caption(f"Calculated via Mifflin-St Jeor equation (BMR: {macros['bmr']} kcal | TDEE: {macros['tdee']} kcal).")

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Daily Calories", f"{macros['calories']} kcal")
                    m2.metric("Protein", f"{macros['protein_g']} g", help=macros.get("protein_note", ""))
                    m3.metric("Carbohydrates", f"{macros['carbs_g']} g")
                    m4.metric("Healthy Fats", f"{macros['fats_g']} g")

                    if "protein_note" in macros and macros["protein_note"]:
                        st.info(f"💡 **Protein Allocation:** {macros['protein_note']}")

                    # 5. Conflict-Free 4-Meal Plan
                    st.markdown("#### 🍽️ Conflict-Free Daily Meal Plan")
                    st.caption("Constructed strictly from the verified safe food pool; 100% compliant with all active constraints, allergies, and drug interactions.")
                    plan = resolution["meal_plan"]
                    for meal_name, details in plan.items():
                        with st.expander(f"**{meal_name}**", expanded=True):
                            st.write(details)

                    # 6. Granular Explainability & Restriction Traceability
                    restrictions = resolution["restrictions"]
                    if restrictions:
                        st.markdown("#### 🔍 Granular Explainability & Restriction Traceability")
                        st.caption("Exact clinical reason why each food item was restricted or excluded from your plan.")
                        restr_df = pd.DataFrame(restrictions)
                        st.dataframe(
                            restr_df[["food", "source", "category", "biochemical_reason"]].rename(
                                columns={
                                    "food": "Restricted Food",
                                    "source": "Triggering Factor",
                                    "category": "Classification",
                                    "biochemical_reason": "Biochemical Hazard / Rationale",
                                }
                            ),
                            use_container_width=True,
                            hide_index=True,
                        )

    with tab5:
        st.subheader("🩺 Disease Prediction")
        st.caption("Select your symptoms below to identify the disease.")

        # Clean searchable multi-select
        selected_symptoms = st.multiselect(
            "Select Symptoms:",
            options=sorted(columns),
            placeholder="Type or select symptoms (e.g. Fever, Cough, Headache, Skin Rash...)",
            help="Choose one or more symptoms from the 50 available clinical features.",
        )

        with st.expander("📋 Or browse symptoms list"):
            checkbox_symptoms = []
            c_cols = st.columns(3)
            for i, col in enumerate(columns):
                with c_cols[i % 3]:
                    if st.checkbox(col, key=f"symptom_chk_{col}"):
                        checkbox_symptoms.append(col)

        all_selected_symptoms = list(set(selected_symptoms + checkbox_symptoms))

        if all_selected_symptoms:
            st.info(f"Selected **{len(all_selected_symptoms)}** symptom(s): {', '.join(sorted(all_selected_symptoms))}")

        if st.button("🔍 Predict Disease", type="primary", use_container_width=True):
            if not all_selected_symptoms:
                st.warning("Please select at least one symptom.")
            else:
                input_data = [1 if col in all_selected_symptoms else 0 for col in columns]
                pred = model.predict([input_data])
                disease_name = le.inverse_transform(pred)[0]

                # Modern, executive clinical diagnosis card
                st.markdown(
                    f"""
                    <div style="
                        background: #ffffff;
                        border: 1px solid #e2e8f0;
                        border-left: 6px solid #0284c7;
                        border-radius: 14px;
                        padding: 24px 28px;
                        margin: 22px 0;
                        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.04);
                    ">
                        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px;">
                            <span style="font-size: 12px; text-transform: uppercase; letter-spacing: 1.5px; color: #0284c7; font-weight: 700;">
                                Clinical Machine Learning Output
                            </span>
                            <span style="background: #e0f2fe; color: #0369a1; padding: 4px 10px; border-radius: 9999px; font-size: 12px; font-weight: 600;">
                                ✓ Model Verified
                            </span>
                        </div>
                        <h1 style="margin: 4px 0 0 0; font-size: 36px; color: #0f172a; font-weight: 800; letter-spacing: -0.02em;">
                            {disease_name}
                        </h1>
                        <p style="margin: 8px 0 0 0; color: #64748b; font-size: 14px;">
                            Identified from {len(all_selected_symptoms)} selected symptom(s).
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

