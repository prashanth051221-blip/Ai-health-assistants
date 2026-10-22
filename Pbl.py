import hashlib
import json
import os
import pickle
import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from pymongo import MongoClient
from rapidfuzz import fuzz, process


st.set_page_config(page_title="AI Health Assistant", layout="wide")

LOCAL_DB_PATH = Path("data/local_health_app.json")


def set_bg():
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(135deg, #d8f3dc 0%, #74c69d 45%, #277da1 100%);
        }
        .block-container {
            background: rgba(10, 20, 30, 0.78);
            padding: 1.5rem;
            border-radius: 8px;
        }
        h1, h2, h3, label, p, div, span {
            color: #f8fafc;
        }
        .stDataFrame div, .stDataFrame span {
            color: inherit;
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

set_bg()
st.title("AI Health Assistant")
st.caption("Disease food guidance, BMI tracking, diet planning, and symptom prediction.")

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
                        st.markdown("##### 🟢 Foods to Eat")
                        eat_items = [item.strip() for item in re.split(r'[;,]', str(row["Foods_to_Eat"])) if item.strip()]
                        for item in eat_items:
                            st.write(f"• {item}")

                    with right:
                        st.markdown("##### 🔴 Foods to Avoid")
                        avoid_items = [item.strip() for item in re.split(r'[;,]', str(row["Foods_to_Avoid"])) if item.strip()]
                        for item in avoid_items:
                            st.write(f"• {item}")

                    st.info(f"💡 **Note:** {row['Note']}")
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

            fig, ax = plt.subplots()
            ax.plot(history_df["created_at"], history_df["bmi"], marker="o")
            ax.set_xlabel("Date")
            ax.set_ylabel("BMI")
            ax.set_title("BMI Progress")
            plt.xticks(rotation=30, ha="right")
            st.pyplot(fig)
        else:
            st.info("No BMI records yet. Calculate your BMI to start tracking.")

    with tab4:
        st.subheader("Diet Planner")
        goal = st.selectbox("Goal", ["Weight Loss", "Muscle Gain", "Healthy"])
        if st.button("Generate Diet Plan"):
            plan = build_diet_plan(goal)
            for meal, details in plan.items():
                st.markdown(f"**{meal}**")
                st.write(details)

    with tab5:
        st.subheader("Symptom Prediction")
        input_dict = {}
        cols_ui = st.columns(2)

        for i, col in enumerate(columns):
            with cols_ui[i % 2]:
                input_dict[col] = st.checkbox(col)

        if st.button("Predict Disease"):
            input_data = [1 if input_dict[col] else 0 for col in columns]
            pred = model.predict([input_data])
            disease_name = le.inverse_transform(pred)[0]
            st.success(f"Predicted disease: **{disease_name}**")

            pred_matches = df[df["Disease"].str.lower() == disease_name.lower()]
            if not pred_matches.empty:
                pred_row = pred_matches.iloc[0]
                st.markdown("---")
                st.subheader(f"Diet Guidance for {disease_name}")
                st.caption(f"*{pred_row['Description']}*")
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("##### 🟢 Foods to Eat")
                    for item in re.split(r'[;,]', str(pred_row["Foods_to_Eat"])):
                        if item.strip():
                            st.write(f"• {item.strip()}")
                with c2:
                    st.markdown("##### 🔴 Foods to Avoid")
                    for item in re.split(r'[;,]', str(pred_row["Foods_to_Avoid"])):
                        if item.strip():
                            st.write(f"• {item.strip()}")
                st.info(f"💡 **Note:** {pred_row['Note']}")
