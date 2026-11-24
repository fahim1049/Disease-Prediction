
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# -------------------------------
# PAGE CONFIGURATION
# -------------------------------

st.set_page_config(
    page_title="AI Disease Prediction System",
    page_icon="🩺",
    layout="wide"
)

# -------------------------------
# LOAD MODEL
# -------------------------------

@st.cache_resource
def load_bundle():
    return joblib.load("disease_prediction_bundle.joblib")

bundle = load_bundle()

model = bundle["model"]
feature_names = bundle["feature_names"]
severity_df = bundle["symptom_severity"]
description_df = bundle["disease_description"]
precaution_df = bundle["disease_precaution"]

@st.cache_resource
def load_explainer():
    return shap.TreeExplainer(model)

explainer = load_explainer()

# -------------------------------
# HEADER
# -------------------------------

st.title("🩺 AI-Based Disease Prediction System")

st.markdown(
    """
    Select your symptoms to explore the machine learning
    model's prediction and its SHAP-based explanation.
    """
)

st.warning(
    "Educational research prototype only. "
    "This application does not provide medical diagnosis "
    "or replace professional medical advice."
)

st.divider()

# -------------------------------
# SYMPTOM SELECTION
# -------------------------------

st.subheader("Select Your Symptoms")

selected_symptoms = st.multiselect(
    "Search and select symptoms:",
    options=feature_names,
    placeholder="Type to search symptoms..."
)

st.write("Selected Symptoms:", len(selected_symptoms))

# -------------------------------
# PREDICTION
# -------------------------------

if st.button("Predict Disease", type="primary"):

    if not selected_symptoms:
        st.warning("Please select at least one symptom.")

    else:

        # Create input feature vector
        input_data = pd.DataFrame(
            np.zeros((1, len(feature_names))),
            columns=feature_names
        )

        for symptom in selected_symptoms:
            input_data.loc[0, symptom] = 1

        # Model prediction
        prediction = model.predict(input_data)[0]

        probabilities = model.predict_proba(input_data)[0]

        top_indices = np.argsort(probabilities)[::-1][:3]

        # ---------------------------
        # PREDICTION RESULT
        # ---------------------------

        st.divider()
        st.subheader("Prediction Result")

        st.success(f"Model Prediction: {prediction}")

        st.subheader("Top 3 Model Scores")

        for idx in top_indices:
            disease = model.classes_[idx]
            score = probabilities[idx]

            st.write(
                f"**{disease}** — {score * 100:.2f}%"
            )

            st.progress(float(score))

        # ---------------------------
        # DESCRIPTION
        # ---------------------------

        st.divider()
        st.subheader("Disease Information")

        disease_info = description_df[
            description_df["Disease"].str.strip() == prediction
        ]

        if not disease_info.empty:
            st.write(disease_info.iloc[0]["Description"])
        else:
            st.info("Description not available.")

        # ---------------------------
        # PRECAUTIONS
        # ---------------------------

        st.subheader("General Precautions")

        precaution_info = precaution_df[
            precaution_df["Disease"].str.strip() == prediction
        ]

        if not precaution_info.empty:

            precaution_columns = [
                col for col in precaution_df.columns
                if col.startswith("Precaution_")
            ]

            for item in precaution_info.iloc[0][
                precaution_columns
            ]:
                if pd.notna(item) and str(item).strip():
                    st.write("•", item)

        else:
            st.info("Precautions not available.")

        # ---------------------------
        # SHAP EXPLANATION
        # ---------------------------

        st.divider()
        st.subheader("Explainable AI - SHAP")

        st.write(
            "The following chart explains how symptom features "
            "contribute to the selected model prediction."
        )

        class_idx = list(model.classes_).index(prediction)

        shap_values = explainer.shap_values(input_data)

        # Current multiclass SHAP shape:
        # (samples, features, classes)
        if isinstance(shap_values, np.ndarray):
            if shap_values.ndim == 3:
                sample_shap = shap_values[0, :, class_idx]
            else:
                sample_shap = shap_values[0]
        else:
            sample_shap = shap_values[class_idx][0]

        shap_df = pd.DataFrame({
            "Symptom": feature_names,
            "SHAP Value": sample_shap
        })

        shap_df["Absolute SHAP"] = (
            shap_df["SHAP Value"].abs()
        )

        shap_df = shap_df.sort_values(
            "Absolute SHAP",
            ascending=False
        ).head(15)

        shap_df = shap_df.sort_values(
            "SHAP Value"
        )

        fig, ax = plt.subplots(figsize=(9, 6))

        ax.barh(
            shap_df["Symptom"],
            shap_df["SHAP Value"]
        )

        ax.set_xlabel("SHAP Value")
        ax.set_title(
            f"SHAP Explanation - {prediction}"
        )

        plt.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "SHAP explains model behavior, not medical causation "
            "or actual disease probability."
        )

        st.divider()

        st.info(
            "This result is generated by a machine learning model. "
            "Consult a qualified healthcare professional for "
            "medical evaluation."
        )
