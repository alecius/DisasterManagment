import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="AI Disaster Management System",
    page_icon="🌍",
    layout="wide"
)

# ==========================================
# 2. LOAD TRAINED MODELS
# ==========================================

MODEL_DIR = Path(__file__).resolve().parent / "models"


@st.cache_resource
def load_models():

    damage_path = MODEL_DIR / "damage_model.pkl"
    report_path = MODEL_DIR / "report_model.pkl"
    priority_path = MODEL_DIR / "priority_model.pkl"

    missing = [
        name
        for name, path in [
            ("Damage model", damage_path),
            ("Report model", report_path),
            ("Priority model", priority_path)
        ]
        if not path.exists()
    ]

    if missing:
        return None, missing

    damage_model = joblib.load(damage_path)
    report_model = joblib.load(report_path)
    priority_model = joblib.load(priority_path)

    return {
        "damage": damage_model,
        "report": report_model,
        "priority": priority_model
    }, []


models, missing_models = load_models()

# ==========================================
# 3. CUSTOM CSS
# ==========================================

st.markdown("""
<style>
    .main {
        background-color: #f5f7fb;
    }

    .main-title {
        font-size: 36px;
        font-weight: bold;
        color: #15304b;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #64748b;
        font-size: 16px;
        margin-bottom: 30px;
    }

    .result-box {
        padding: 18px;
        border-radius: 12px;
        background-color: #eef4ff;
        border: 1px solid #cbdcff;
        margin: 10px 0;
    }

    .small-text {
        color: #64748b;
        font-size: 13px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 4. HEADER
# ==========================================

st.markdown(
    '<div class="main-title">'
    '🌍 AI-Based Disaster Management System'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Damage Severity Prediction and Response Prioritization'
    '</div>',
    unsafe_allow_html=True
)

st.warning(
    "Educational prototype: the models use small synthetic "
    "datasets. Predictions are not validated for real "
    "emergency-response decisions."
)

# ==========================================
# 5. SIDEBAR
# ==========================================

with st.sidebar:

    st.title("🌍 Disaster AI")

    st.write("Navigation")

    page = st.radio(
        "Choose a page",
        [
            "Disaster Assessment",
            "About Project"
        ]
    )

    st.divider()

    st.write("### Model Status")

    if models is not None:
        st.success("Three model files loaded")
    else:
        st.error("Some model files are missing")

        for name in missing_models:
            st.write(f"• {name}")

# ==========================================
# 6. DISASTER ASSESSMENT PAGE
# ==========================================

if page == "Disaster Assessment":

    st.header("📋 Enter Disaster Information")

    left, right = st.columns(2)

    with left:

        disaster_type = st.selectbox(
            "Disaster Type",
            [
                "Earthquake",
                "Flood",
                "Landslide",
                "Cyclone",
                "Fire"
            ]
        )

        magnitude = st.slider(
            "Disaster Magnitude / Intensity",
            min_value=1.0,
            max_value=10.0,
            value=4.0,
            step=0.1
        )

        people_affected = st.number_input(
            "Estimated People Affected",
            min_value=0,
            value=100,
            step=10
        )

    with right:

        buildings_damaged = st.number_input(
            "Estimated Buildings Damaged",
            min_value=0,
            value=10,
            step=1
        )

        infrastructure_damage = st.slider(
            "Infrastructure Damage Level",
            min_value=1,
            max_value=10,
            value=3
        )

        location = st.text_input(
            "Affected Location",
            placeholder="Enter district or area"
        )

    report = st.text_area(
        "Disaster Report",
        placeholder=(
            "Example: An earthquake damaged several buildings "
            "and blocked a nearby road."
        ),
        height=120
    )

    st.caption(
        "Enter the best available estimates. Do not enter "
        "unverified information as confirmed facts."
    )

    analyze_button = st.button(
        "🔍 Analyze Disaster",
        use_container_width=True,
        type="primary"
    )

    if analyze_button:

        if models is None:

            st.error(
                "Models are missing. Run "
                "'python train_model.py' in the terminal first."
            )

        elif not report.strip():

            st.error("Please enter a disaster report.")

        else:

            # Prepare input in the same column order
            # used during training.
            input_data = pd.DataFrame(
                [[
                    magnitude,
                    people_affected,
                    buildings_damaged,
                    infrastructure_damage
                ]],
                columns=[
                    "magnitude",
                    "people_affected",
                    "buildings_damaged",
                    "infrastructure_damage"
                ]
            )

            # ------------------------------
            # MODEL 1: DAMAGE SEVERITY
            # ------------------------------

            damage_prediction = models["damage"].predict(
                input_data
            )[0]

            # ------------------------------
            # MODEL 2: REPORT CLASSIFICATION
            # ------------------------------

            report_prediction = models["report"].predict(
                [report]
            )[0]

            # ------------------------------
            # MODEL 3: RESPONSE PRIORITY
            # ------------------------------

            priority_prediction = models["priority"].predict(
                input_data
            )[0]

            # ------------------------------
            # DISPLAY RESULTS
            # ------------------------------

            st.divider()

            st.header("📊 Assessment Results")

            st.write(f"**Disaster Type:** {disaster_type}")

            st.write(
                f"**Location:** {location or 'Not provided'}"
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Damage Severity",
                    str(damage_prediction)
                )

            with col2:

                st.metric(
                    "Report Classification",
                    str(report_prediction)
                )

            with col3:

                st.metric(
                    "Response Priority",
                    str(priority_prediction)
                )

            # ------------------------------
            # EXPLANATION
            # ------------------------------

            st.subheader("📝 Report Summary")

            st.write(report)

            st.subheader("💡 General Guidance")

            if str(priority_prediction).lower() == "urgent":

                st.error(
                    "The model assigned an Urgent priority. "
                    "This is an experimental prediction, not "
                    "a verified emergency assessment. "
                    "Have qualified responders review the incident."
                )

            elif str(priority_prediction).lower() == "high":

                st.warning(
                    "The model assigned a High priority. "
                    "Verify the reported conditions and seek "
                    "professional assessment where appropriate."
                )

            elif str(priority_prediction).lower() == "medium":

                st.info(
                    "The model assigned a Medium priority. "
                    "Review the available information and "
                    "verify the reported damage."
                )

            else:

                st.info(
                    "The model assigned a Low priority. "
                    "This does not establish that the location "
                    "is safe or that assistance is unnecessary."
                )

            st.divider()

            st.subheader("⚠️ Important Limitations")

            st.write(
                "- The models were trained on tiny demonstration datasets."
            )

            st.write(
                "- The predictions are not validated against real disasters."
            )

            st.write(
                "- Disaster type and location are displayed but are not "
                "used as model inputs in this version."
            )

            st.write(
                "- The three models make separate predictions; "
                "they are not combined into a trained fusion model."
            )

            st.write(
                "- These predictions must not determine real emergency "
                "dispatch, evacuation, or rescue decisions."
            )

# ==========================================
# 7. ABOUT PROJECT PAGE
# ==========================================

elif page == "About Project":

    st.header("📚 About Our Project")

    st.write("""
    The AI-Based Disaster Management and Damage Assessment
    System is a college mini-project demonstrating how
    Machine Learning can be applied to disaster-related data.
    """)

    st.subheader("Project Objectives")

    st.write("""
    1. Predict damage severity from numerical information.
    2. Classify disaster reports using text classification.
    3. Predict a preliminary response-priority category.
    4. Present model outputs in a simple web interface.
    """)

    st.subheader("Machine Learning Algorithms")

    st.write("""
    **Decision Tree:** Predicts damage severity from
    numerical features.

    **Naive Bayes:** Classifies text reports into categories.

    **Random Forest:** Predicts response priority from
    numerical features.
    """)

    st.subheader("Technologies Used")

    st.write("""
    - Python
    - Pandas
    - Scikit-learn
    - Joblib
    - Streamlit
    """)

    st.subheader("Project Workflow")

    st.write("""
    Disaster Information → Feature Preparation →
    Machine Learning Models → Predictions → Dashboard
    """)

    st.info(
        "This project is an educational demonstration, "
        "not a validated disaster-management service."
    )

# ==========================================
# 8. FOOTER
# ==========================================

st.divider()

st.caption(
    "AI-Based Disaster Management System | College Mini Project"
)