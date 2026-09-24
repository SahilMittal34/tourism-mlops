import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Wellness Tourism Package Predictor",
    page_icon="✈️",
    layout="wide",
)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "deployment" / "model.joblib"
METADATA_PATH = ROOT / "deployment" / "model_metadata.json"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata():
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


model = load_model()
metadata = load_metadata()

st.title("✈️ Wellness Tourism Package Purchase Predictor")
st.markdown(
    "Use the customer and interaction details below to estimate the probability "
    "that a customer will purchase the Wellness Tourism Package."
)

with st.expander("About the model", expanded=False):
    st.write(f"**Production model:** {metadata['model_name'].replace('_', ' ')}")
    st.write("The model uses the same preprocessing pipeline used during training, including numerical imputation/scaling and categorical one-hot encoding.")
    st.caption("This application is for project demonstration purposes; it should not be treated as a production marketing decision without business validation.")

with st.form("prediction_form"):
    st.subheader("Customer details")
    c1, c2, c3 = st.columns(3)

    with c1:
        age = st.number_input("Age", min_value=18, max_value=80, value=35, step=1)
        city_tier = st.selectbox("City Tier", [1, 2, 3], index=1)
        occupation = st.selectbox(
            "Occupation",
            ["Salaried", "Free Lancer", "Small Business", "Large Business"],
        )
        gender = st.selectbox("Gender", ["Male", "Female", "Fe Male"])
        marital = st.selectbox(
            "Marital Status", ["Married", "Single", "Divorced", "Unmarried"]
        )
        designation = st.selectbox(
            "Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
        )

    with c2:
        contact = st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry"])
        persons = st.number_input(
            "Number of Persons Visiting", min_value=1, max_value=10, value=2, step=1
        )
        children = st.number_input(
            "Number of Children Visiting", min_value=0, max_value=5, value=0, step=1
        )
        property_star = st.selectbox("Preferred Property Star", [3, 4, 5], index=1)
        trips = st.number_input("Number of Trips", min_value=0, max_value=30, value=3, step=1)
        passport = st.selectbox("Passport", [0, 1], format_func=lambda x: "Yes" if x else "No")

    with c3:
        own_car = st.selectbox("Own Car", [0, 1], format_func=lambda x: "Yes" if x else "No")
        pitch_score = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
        product = st.selectbox(
            "Product Pitched", ["Basic", "Deluxe", "King", "Standard", "Super Deluxe"]
        )
        followups = st.number_input("Number of Followups", min_value=0, max_value=10, value=3, step=1)
        pitch_duration = st.number_input(
            "Duration of Pitch (minutes)", min_value=0, max_value=150, value=15, step=1
        )
        income = st.number_input(
            "Monthly Income", min_value=5000.0, max_value=200000.0, value=25000.0, step=500.0
        )

    submitted = st.form_submit_button("🔮 Predict Purchase Probability", use_container_width=True)

if submitted:
    row = pd.DataFrame(
        [
            {
                "Age": age,
                "TypeofContact": contact,
                "CityTier": city_tier,
                "DurationOfPitch": pitch_duration,
                "Occupation": occupation,
                "Gender": gender,
                "NumberOfPersonVisiting": persons,
                "NumberOfFollowups": followups,
                "ProductPitched": product,
                "PreferredPropertyStar": property_star,
                "MaritalStatus": marital,
                "NumberOfTrips": trips,
                "Passport": passport,
                "PitchSatisfactionScore": pitch_score,
                "OwnCar": own_car,
                "NumberOfChildrenVisiting": children,
                "Designation": designation,
                "MonthlyIncome": income,
            }
        ]
    )

    probability = float(model.predict_proba(row)[:, 1][0])
    prediction = int(probability >= 0.50)

    st.divider()
    r1, r2 = st.columns(2)
    with r1:
        st.metric("Purchase probability", f"{probability:.1%}")
    with r2:
        st.metric("Decision threshold", "50%")

    if prediction == 1:
        st.success("Prediction: Customer is likely to purchase the Wellness Tourism Package.")
    else:
        st.info("Prediction: Customer is less likely to purchase the Wellness Tourism Package.")

    st.progress(probability)
    st.caption("The 50% threshold is a demonstration threshold. A production campaign should tune the threshold using campaign capacity and the business costs of false positives and false negatives.")

st.sidebar.header("Project")
st.sidebar.write("Visit with Us — Wellness Tourism Package")
st.sidebar.write(f"Model: {metadata['model_name'].replace('_', ' ')}")
st.sidebar.write("MLOps: GitHub Actions → Streamlit Community Cloud")
