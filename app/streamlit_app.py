
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Weather Intelligence Platform",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "best_weather_model.pkl"
)

FEATURE_PATH = os.path.join(
    BASE_DIR,
    "models",
    "feature_columns.pkl"
)

SUMMARY_PATH = os.path.join(
    BASE_DIR,
    "final_model_summary.csv"
)

SHAP_PATH = os.path.join(
    BASE_DIR,
    "shap_feature_importance.csv"
)

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = joblib.load(
        MODEL_PATH
    )

    features = joblib.load(
        FEATURE_PATH
    )

    return model, features


model, feature_columns = load_model()

# ============================================================
# LOAD RESULTS
# ============================================================

@st.cache_data
def load_results():

    if os.path.exists(SUMMARY_PATH):

        summary = pd.read_csv(
            SUMMARY_PATH
        )

    else:

        summary = pd.DataFrame()

    if os.path.exists(SHAP_PATH):

        shap_data = pd.read_csv(
            SHAP_PATH
        )

    else:

        shap_data = pd.DataFrame()

    return summary, shap_data


model_summary, shap_data = load_results()

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("🌧️ Weather Intelligence")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "🔮 Rainfall Prediction",
        "📊 Model Performance",
        "🧠 Explainable AI",
        "📚 About Project"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "MacroEdtech GenAI Research 2026"
)

st.sidebar.caption(
    "Phase 02 — Weather Intelligence Platform"
)

# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.title(
        "🌧️ Weather Intelligence Platform"
    )

    st.subheader(
        "AI-Based Southwest Monsoon Rainfall Prediction"
    )

    st.write(
        "An AI-powered weather analysis and prediction "
        "platform developed using machine learning, "
        "time-series forecasting and explainable AI."
    )

    st.divider()

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Dataset Records",
            "4,116"
        )

    with col2:

        st.metric(
            "Features",
            len(feature_columns)
        )

    with col3:

        st.metric(
            "Prediction Target",
            "Rainfall"
        )

    with col4:

        st.metric(
            "Platform",
            "AI / ML"
        )

    st.divider()

    # --------------------------------------------------------
    # PROJECT PIPELINE
    # --------------------------------------------------------

    st.header(
        "🔄 AI Weather Prediction Pipeline"
    )

    pipeline = """

    Data Acquisition
            ↓
    Data Preprocessing
            ↓
    Exploratory Data Analysis
            ↓
    Feature Engineering
            ↓
    Machine Learning
            ↓
    Time-Series Forecasting
            ↓
    Deep Learning
            ↓
    Model Evaluation
            ↓
    SHAP Explainability
            ↓
    FastAPI Backend
            ↓
    Streamlit Application
    """

    st.code(
        pipeline,
        language="text"
    )

    st.success(
        "The core Weather Prediction Engine has been implemented successfully."
    )

# ============================================================
# RAINFALL PREDICTION
# ============================================================

elif page == "🔮 Rainfall Prediction":

    st.title(
        "🔮 Rainfall Prediction"
    )

    st.write(
        "Enter the model input parameters below "
        "to generate a rainfall prediction."
    )

    st.divider()

    input_values = {}

    # --------------------------------------------------------
    # INPUTS
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    for index, feature in enumerate(
        feature_columns
    ):

        column = (
            col1
            if index % 2 == 0
            else col2
        )

        with column:

            if feature == "YEAR":

                input_values[feature] = st.number_input(
                    "Year",
                    min_value=1900,
                    max_value=2100,
                    value=2015,
                    step=1
                )

            else:

                input_values[feature] = st.number_input(
                    feature,
                    value=0.0,
                    format="%.4f"
                )

    st.divider()

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    if st.button(
        "🔮 Predict Rainfall",
        type="primary",
        use_container_width=True
    ):

        input_df = pd.DataFrame(
            [input_values]
        )

        input_df = input_df[
            feature_columns
        ]

        try:

            prediction = model.predict(
                input_df
            )

            rainfall = float(
                prediction[0]
            )

            st.success(
                "Prediction completed successfully."
            )

            st.metric(
                "Predicted Rainfall",
                f"{rainfall:,.2f} mm"
            )

            if rainfall < 500:

                st.info(
                    "Prediction indicates relatively low rainfall."
                )

            elif rainfall < 1000:

                st.info(
                    "Prediction indicates moderate rainfall."
                )

            elif rainfall < 2000:

                st.warning(
                    "Prediction indicates high rainfall."
                )

            else:

                st.error(
                    "Prediction indicates very high rainfall."
                )

        except Exception as e:

            st.error(
                f"Prediction failed: {e}"
            )

# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "📊 Model Performance":

    st.title(
        "📊 Model Performance"
    )

    st.write(
        "Comparison of the forecasting models "
        "evaluated during the project."
    )

    if not model_summary.empty:

        st.dataframe(
            model_summary,
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        # ----------------------------------------------------
        # RMSE
        # ----------------------------------------------------

        if "RMSE" in model_summary.columns:

            st.subheader(
                "RMSE Comparison"
            )

            rmse_data = model_summary[
                ["Model", "RMSE"]
            ].copy()

            rmse_data = rmse_data.set_index(
                "Model"
            )

            st.bar_chart(
                rmse_data
            )

        # ----------------------------------------------------
        # R2
        # ----------------------------------------------------

        if "R2" in model_summary.columns:

            st.subheader(
                "R² Comparison"
            )

            r2_data = model_summary[
                ["Model", "R2"]
            ].copy()

            r2_data = r2_data.set_index(
                "Model"
            )

            st.bar_chart(
                r2_data
            )

    else:

        st.warning(
            "Model comparison results are not available."
        )

# ============================================================
# EXPLAINABLE AI
# ============================================================

elif page == "🧠 Explainable AI":

    st.title(
        "🧠 Explainable AI"
    )

    st.write(
        "SHAP (SHapley Additive exPlanations) is used "
        "to understand the contribution of individual "
        "features to the model predictions."
    )

    if not shap_data.empty:

        st.subheader(
            "Top Feature Importance"
        )

        st.dataframe(
            shap_data.head(15),
            use_container_width=True,
            hide_index=True
        )

        if "Feature" in shap_data.columns:

            importance = shap_data.head(15)

            importance_chart = (
                importance
                .set_index("Feature")
                ["Mean_Absolute_SHAP"]
            )

            st.subheader(
                "SHAP Feature Importance"
            )

            st.bar_chart(
                importance_chart
            )

        # ----------------------------------------------------
        # SHAP IMAGE
        # ----------------------------------------------------

        shap_image = os.path.join(
            BASE_DIR,
            "figures",
            "shap_feature_importance.png"
        )

        if os.path.exists(shap_image):

            st.subheader(
                "SHAP Visualization"
            )

            st.image(
                shap_image,
                caption="SHAP Feature Importance"
            )

    else:

        st.warning(
            "SHAP results are not available."
        )

# ============================================================
# ABOUT PROJECT
# ============================================================

elif page == "📚 About Project":

    st.title(
        "📚 About the Project"
    )

    st.header(
        "Project Objective"
    )

    st.write(
        "The objective of this project is to develop "
        "an AI-powered Weather Intelligence and Climate "
        "Decision Support Platform for analyzing and "
        "predicting the Indian Southwest Monsoon."
    )

    st.header(
        "Technologies Used"
    )

    technologies = [
        "Python",
        "Pandas",
        "NumPy",
        "Scikit-learn",
        "Statsmodels",
        "TensorFlow / Keras",
        "SHAP",
        "FastAPI",
        "Streamlit",
        "Git",
        "GitHub"
    ]

    for technology in technologies:

        st.write(
            f"• {technology}"
        )

    st.header(
        "Project Components"
    )

    components = [
        "Data Acquisition",
        "Data Preprocessing",
        "Exploratory Data Analysis",
        "Feature Engineering",
        "Machine Learning",
        "Time-Series Forecasting",
        "Deep Learning",
        "Model Evaluation",
        "Explainable AI",
        "FastAPI Backend",
        "Streamlit User Interface"
    ]

    for component in components:

        st.write(
            f"✓ {component}"
        )

    st.divider()

    st.info(
        "This application represents the Part 01 "
        "Weather Prediction Engine of the MacroEdtech "
        "GenAI Research 2026 Phase 02 project."
    )

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Weather Intelligence Platform | "
    "MacroEdtech GenAI Research 2026"
)
