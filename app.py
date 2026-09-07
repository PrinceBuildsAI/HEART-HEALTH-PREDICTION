import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------
st.markdown(
    """
<style>

.main {
    background-color: #f8fafc;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

/* Main title */
.main-title {
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 18px;
    color: #64748b;
    margin-bottom: 25px;
}

/* Cards */
.info-card {
    padding: 20px;
    border-radius: 15px;
    background: white;
    border: 1px solid #e2e8f0;
    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    margin-bottom: 15px;
}

.metric-card {
    padding: 18px;
    border-radius: 15px;
    background: white;
    border: 1px solid #e2e8f0;
    text-align: center;
}

/* Result */
.result-card {
    padding: 30px;
    border-radius: 20px;
    text-align: center;
    margin-top: 25px;
}

/* Buttons */
.stButton > button {
    width: 100%;
    height: 50px;
    border-radius: 12px;
    font-size: 17px;
    font-weight: 700;
}

/* Section headers */
.section-title {
    font-size: 24px;
    font-weight: 700;
    margin-top: 15px;
    margin-bottom: 10px;
}

/* Disclaimer */
.disclaimer {
    padding: 15px;
    border-radius: 12px;
    background: #fff7ed;
    border: 1px solid #fed7aa;
    color: #9a3412;
    font-size: 14px;
}

</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource
def load_model():

    model = joblib.load(os.path.join(BASE_DIR, "KNN_heart.pkl"))

    scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

    expected_columns = joblib.load(os.path.join(BASE_DIR, "columns.pkl"))

    return model, scaler, expected_columns


model, scaler, expected_columns = load_model()
# ---------------------------------------------------
# HEADER
# ---------------------------------------------------
st.markdown(
    '<div class="main-title">❤️ Heart Disease Risk Predictor</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "AI-powered heart disease risk assessment using a KNN machine learning model."
    "</div>",
    unsafe_allow_html=True,
)


# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
with st.sidebar:

    st.header("⚙️ Prediction Settings")

    st.markdown("""
    **Risk Levels**

    🟢 **Low Risk**  
    Probability below 30%

    🟡 **Moderate Risk**  
    Probability between 30%–60%

    🔴 **High Risk**  
    Probability above 60%
    """)

    st.divider()

    st.info(
        "Enter the patient's information carefully for the model to generate a prediction."
    )


# ---------------------------------------------------
# PATIENT INFORMATION
# ---------------------------------------------------
st.markdown(
    '<div class="section-title">👤 Patient Information</div>', unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", min_value=18, max_value=100, value=45)

with col2:
    sex = st.selectbox("Sex", ["M", "F"])

with col3:
    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])


# ---------------------------------------------------
# VITALS
# ---------------------------------------------------
st.markdown(
    '<div class="section-title">🩺 Vital & Laboratory Information</div>',
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)

with col1:
    resting_bp = st.number_input(
        "Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120
    )

with col2:
    cholesterol = st.number_input(
        "Cholesterol (mg/dL)", min_value=100, max_value=600, value=200
    )

with col3:
    fasting_bs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dL",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No",
    )


# ---------------------------------------------------
# HEART TEST RESULTS
# ---------------------------------------------------
st.markdown(
    '<div class="section-title">❤️ Heart Test Results</div>', unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])

with col2:
    max_hr = st.slider("Maximum Heart Rate", min_value=60, max_value=220, value=150)

with col3:
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])


# ---------------------------------------------------
# ADDITIONAL PARAMETERS
# ---------------------------------------------------
with st.expander("🔬 Additional Heart Parameters"):

    col1, col2 = st.columns(2)

    with col1:
        oldpeak = st.slider(
            "Oldpeak (ST Depression)", min_value=0.0, max_value=6.0, value=1.0, step=0.1
        )

    with col2:
        st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])


# ---------------------------------------------------
# PREDICT BUTTON
# ---------------------------------------------------
st.divider()

predict_col, reset_col = st.columns(2)

with predict_col:
    predict_button = st.button("🔍 Analyze Heart Disease Risk", type="primary")

with reset_col:
    reset_button = st.button("🔄 Reset")


# ---------------------------------------------------
# RESET
# ---------------------------------------------------
if reset_button:
    st.rerun()


# ---------------------------------------------------
# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------

if predict_button:

    with st.spinner("Analyzing patient information..."):

        # Original input
        raw_input = {
            "Age": age,
            "RestingBP": resting_bp,
            "Cholesterol": cholesterol,
            "FastingBS": fasting_bs,
            "MaxHR": max_hr,
            "Oldpeak": oldpeak,
            "Sex": sex,
            "ChestPainType": chest_pain,
            "RestingECG": resting_ecg,
            "ExerciseAngina": exercise_angina,
            "ST_Slope": st_slope,
        }

        input_df = pd.DataFrame([raw_input])

        # One-hot encoding
        input_df = pd.get_dummies(input_df)

        # Match training columns exactly
        input_df = input_df.reindex(columns=expected_columns, fill_value=0)

        # Scaling
        scaled_input = scaler.transform(input_df)

        # Prediction
        prediction = model.predict(scaled_input)[0]

        # IMPORTANT:
        # Initialize probability before using it
        probability = None

        # Probability
        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(scaled_input)[0]

            # Find probability of class 1
            if 1 in model.classes_:

                high_risk_index = list(model.classes_).index(1)

                probability = probabilities[high_risk_index] * 100

    # ------------------------------------------------
    # RISK CLASSIFICATION
    # ------------------------------------------------

    if prediction == 0:

        risk_level = "LOW RISK"
        risk_icon = "🟢"
        risk_message = "The model classified this input as lower risk."

    else:

        if probability is not None:

            if probability <= 60:

                risk_level = "MODERATE RISK"
                risk_icon = "🟡"
                risk_message = "The model estimates a moderate level of risk."

            else:

                risk_level = "HIGH RISK"
                risk_icon = "🔴"
                risk_message = "The model estimates a higher level of risk."

        else:

            risk_level = "HIGH RISK"
            risk_icon = "🔴"
            risk_message = "The model classified this input as higher risk."

            probability = 100

    # ------------------------------------------------
    # RESULT DISPLAY
    # ------------------------------------------------

    st.markdown("## 📊 Prediction Result")

    st.markdown(
        f"""
    <div class="result-card">
        <div style="font-size:50px;">{risk_icon}</div>
        <h1>{risk_level}</h1>
        <p style="font-size:18px;">{risk_message}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # ------------------------------------------------
    # METRICS
    # ------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Risk Probability", f"{probability:.1f}%")

    with col2:
        st.metric("Age", f"{age} years")

    with col3:
        st.metric("Maximum Heart Rate", f"{max_hr} bpm")

    # ------------------------------------------------
    # PROGRESS BAR
    # ------------------------------------------------

    st.markdown("### Risk Probability")

    st.progress(min(int(probability), 100))

    # ------------------------------------------------
    # INTERPRETATION
    # ------------------------------------------------

    if risk_level == "LOW RISK":

        st.success("🟢 The model places this input in the low-risk category.")

    elif risk_level == "MODERATE RISK":

        st.warning(
            "🟡 The model places this input in the "
            "moderate-risk category. Consider professional "
            "medical evaluation."
        )

    else:

        st.error(
            "🔴 The model places this input in the "
            "high-risk category. Professional medical "
            "evaluation is recommended."
        )
