import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load serialized models
with open("model.pkl", "rb") as f:
    mlp_model = pickle.load(f)  # For Primary Diagnosis prediction
with open("binary_classifier.pkl", "rb") as f:
    logreg_model = pickle.load(f)  # For Grade prediction

# Set up mappings
grade_mapping = {0: "LGG", 1: "GBM"}
diagnosis_mapping = {
    0: "Astrocytoma, NOS",
    1: "Astrocytoma, anaplastic",
    2: "Glioblastoma",
    3: "Mixed glioma",
    4: "Oligodendroglioma, NOS",
    5: "Oligodendroglioma, anaplastic",
}

# Set up the layout and title
st.title("Patient Diagnosis and Grade Prediction")
st.write(
    "This app predicts the Primary Diagnosis and Grade based on patient data using MLP and Logistic Regression models."
)

# Define input form for the required variables
with st.form("patient_form"):
    st.subheader("Patient Data Input")
    gender = st.selectbox("Gender", ["Male", "Female"])
    age_at_diagnosis = st.number_input("Age at Diagnosis", min_value=0)

    # Boolean inputs for gene mutations
    IDH1 = st.checkbox("IDH1 Mutation", value=False)
    TP53 = st.checkbox("TP53 Mutation", value=False)
    ATRX = st.checkbox("ATRX Mutation", value=False)
    PTEN = st.checkbox("PTEN Mutation", value=False)
    EGFR = st.checkbox("EGFR Mutation", value=False)
    CIC = st.checkbox("CIC Mutation", value=False)
    MUC16 = st.checkbox("MUC16 Mutation", value=False)
    PIK3CA = st.checkbox("PIK3CA Mutation", value=False)
    NF1 = st.checkbox("NF1 Mutation", value=False)
    PIK3R1 = st.checkbox("PIK3R1 Mutation", value=False)
    FUBP1 = st.checkbox("FUBP1 Mutation", value=False)
    RB1 = st.checkbox("RB1 Mutation", value=False)
    NOTCH1 = st.checkbox("NOTCH1 Mutation", value=False)
    BCOR = st.checkbox("BCOR Mutation", value=False)
    CSMD3 = st.checkbox("CSMD3 Mutation", value=False)
    SMARCA4 = st.checkbox("SMARCA4 Mutation", value=False)
    GRIN2A = st.checkbox("GRIN2A Mutation", value=False)
    IDH2 = st.checkbox("IDH2 Mutation", value=False)
    FAT4 = st.checkbox("FAT4 Mutation", value=False)
    PDGFRA = st.checkbox("PDGFRA Mutation", value=False)

    # Model selection
    model_choice = st.radio(
        "Select the Model",
        ["MLP for Primary Diagnosis", "Logistic Regression for Grade"],
    )

    # Submit button
    submit = st.form_submit_button("Predict")

# On submit, perform predictions based on model choice
if submit:
    # Encode Gender (Assume Male = 0, Female = 1)
    gender_encoded = 1 if gender == "Female" else 0

    # Convert booleans to integer values for gene mutations (True -> 1, False -> 0)
    input_data = pd.DataFrame(
        [
            [
                gender_encoded,
                age_at_diagnosis,
                int(IDH1),
                int(TP53),
                int(ATRX),
                int(PTEN),
                int(EGFR),
                int(CIC),
                int(MUC16),
                int(PIK3CA),
                int(NF1),
                int(PIK3R1),
                int(FUBP1),
                int(RB1),
                int(NOTCH1),
                int(BCOR),
                int(CSMD3),
                int(SMARCA4),
                int(GRIN2A),
                int(IDH2),
                int(FAT4),
                int(PDGFRA),
            ]
        ],
        columns=[
            "Gender",
            "Age_at_diagnosis",
            "IDH1",
            "TP53",
            "ATRX",
            "PTEN",
            "EGFR",
            "CIC",
            "MUC16",
            "PIK3CA",
            "NF1",
            "PIK3R1",
            "FUBP1",
            "RB1",
            "NOTCH1",
            "BCOR",
            "CSMD3",
            "SMARCA4",
            "GRIN2A",
            "IDH2",
            "FAT4",
            "PDGFRA",
        ],
    )

    # Choose and apply the model
    if model_choice == "MLP for Primary Diagnosis":
        # Predict using the MLP model and get the probabilities
        probabilities = mlp_model.predict_proba(input_data)[0]
        prediction = (
            probabilities.argmax()
        )  # Extract the index of the highest probability

        # Format probabilities as percentages
        confidence_percentages = {
            diagnosis_mapping.get(i, "Unknown"): f"{prob * 100:.2f}%"
            for i, prob in enumerate(probabilities)
        }

        # Display results
        st.write(
            "Predicted Primary Diagnosis:", diagnosis_mapping.get(prediction, "Unknown")
        )
        st.write("Confidence Probabilities:", confidence_percentages)

    elif model_choice == "Logistic Regression for Grade":
        # Predict using Logistic Regression model and map the result
        prediction = logreg_model.predict(input_data)[0]
        confidence = logreg_model.predict_proba(input_data)
        st.write("Predicted Grade:", grade_mapping.get(prediction, "Unknown"))
        st.write(f"Confidence Percentage(LGG): {confidence[0][0]*100:.2f}%")
        st.write(f"Confidence Percentage(GBM): {100 - confidence[0][0]*100:.2f}%")
