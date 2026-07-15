import streamlit as st
import pickle
import numpy as np
import shap
import matplotlib.pyplot as plt

# Load model and scaler
with open("ensemble_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("Bank Customer Churn Prediction")

# Load XGBoost model for SHAP explanations
with open("xgb_model.pkl", "rb") as f:
    xgb_model = pickle.load(f)

# SHAP explainer (cache for speed)
explainer = shap.TreeExplainer(xgb_model)

# ===== USER INPUTS =====
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
age = st.number_input("Age", min_value=18, max_value=100, value=40)
tenure = st.slider("Tenure (years)", 0, 10, 5)
balance = st.number_input("Balance", min_value=0.0, value=50000.0)
num_products = st.slider("Number of Products", 1, 4, 1)

has_cr_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
is_active = st.selectbox("Is Active Member?", ["Yes", "No"])

estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=60000.0)

geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Female", "Male"])

# ===== ENCODING (MUST MATCH TRAINING) =====
has_cr_card = 1 if has_cr_card == "Yes" else 0
is_active = 1 if is_active == "Yes" else 0

geo_germany = 1 if geography == "Germany" else 0
geo_spain = 1 if geography == "Spain" else 0

gender_male = 1 if gender == "Male" else 0


# ===== PREDICTION =====
if st.button("Predict"):

    input_data = np.array([[
        credit_score,
        age,
        tenure,
        balance,
        num_products,
        has_cr_card,
        is_active,
        estimated_salary,
        geo_germany,
        geo_spain,
        gender_male
    ]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    # ===== RESULT MESSAGE =====
    if prediction == 1:
        st.error(f" Customer is likely to CHURN (Risk: {probability:.2%})")
    else:
        st.success(f" Customer is likely to STAY (Risk: {probability:.2%})")

    # ===== BANK EMPLOYEE FRIENDLY EXPLANATION =====
    if prediction == 1:  # CHURN CASE
        st.subheader(" Why this customer may leave the bank")

        reasons = []

        if num_products <= 1:
            reasons.append("Customer is using only one banking product")

        if is_active == 0:
            reasons.append("Customer engagement level is low")

        if credit_score < 650:
            reasons.append("Customer has a moderate credit score")

        if age >= 40:
            reasons.append("Customer belongs to a high-risk age group")

        if geography in ["Germany", "France"]:
            reasons.append("Customer location shows higher churn tendency")

        st.markdown("**Key reasons for churn risk:**")
        for r in reasons:
            st.write(f"• {r}")

        st.subheader(" Recommended retention actions")
        actions = [
            "Offer personalized retention discounts",
            "Cross-sell additional banking products",
            "Assign a relationship manager for follow-up",
            "Provide loyalty rewards or premium benefits"
        ]

        for a in actions:
            st.write(f"• {a}")

    else:  # STAY CASE
        st.subheader(" Why this customer is likely to stay")

        st.markdown("**Positive indicators:**")
        positives = []

        if num_products > 1:
            positives.append("Customer is using multiple banking products")

        if is_active == 1:
            positives.append("Customer shows good engagement with the bank")

        if credit_score >= 650:
            positives.append("Customer has a good credit score")

        positives.append("Overall churn risk is low")

        for p in positives:
            st.write(f"• {p}")

        st.subheader(" Suggested growth opportunities")
        growth_actions = [
            "Offer premium product upgrades",
            "Introduce loyalty or referral programs",
            "Maintain regular engagement to retain the customer"
        ]

        for g in growth_actions:
            st.write(f"• {g}")

        # =====================================================
#  BULK CUSTOMER CHURN PREDICTION (CSV UPLOAD)
# =====================================================

st.markdown("---")
st.header(" Bulk Customer Churn Prediction")

uploaded_file = st.file_uploader(
    "Upload CSV file for bulk prediction",
    type=["csv"]
)

if uploaded_file is not None:
    import pandas as pd

    df_upload = pd.read_csv(uploaded_file)

    st.subheader("Preview of Uploaded Data")
    st.dataframe(df_upload.head())

    try:
        required_cols = [
            'CreditScore','Age','Tenure','Balance','NumOfProducts',
            'HasCrCard','IsActiveMember','EstimatedSalary',
            'Geography','Gender'
        ]

        if not all(col in df_upload.columns for col in required_cols):
            st.error("❌ Uploaded file is missing required columns.")
        else:
            # Encoding same as training
            df_upload['Geography_Germany'] = (df_upload['Geography'] == "Germany").astype(int)
            df_upload['Geography_Spain'] = (df_upload['Geography'] == "Spain").astype(int)
            df_upload['Gender_Male'] = (df_upload['Gender'] == "Male").astype(int)

            df_model = df_upload[[
                'CreditScore','Age','Tenure','Balance','NumOfProducts',
                'HasCrCard','IsActiveMember','EstimatedSalary',
                'Geography_Germany','Geography_Spain','Gender_Male'
            ]]

            scaled_data = scaler.transform(df_model)

            predictions = model.predict(scaled_data)
            probabilities = model.predict_proba(scaled_data)[:, 1]

            df_upload["Churn_Prediction"] = predictions
            df_upload["Churn_Risk_%"] = (probabilities * 100).round(2)

            reasons_list = []
            suggestions_list = []

            for i, row in df_upload.iterrows():

                reasons = []
                suggestions = []

                if row['NumOfProducts'] <= 1:
                    reasons.append("Only one banking product")

                if row['IsActiveMember'] == 0:
                    reasons.append("Low engagement")

                if row['CreditScore'] < 650:
                    reasons.append("Moderate credit score")

                if row['Age'] >= 40:
                    reasons.append("High-risk age group")

                if predictions[i] == 1:
                    suggestions = [
                        "Offer retention discount",
                        "Cross-sell additional products",
                        "Assign relationship manager"
                    ]
                else:
                    suggestions = [
                        "Offer premium upgrades",
                        "Maintain engagement"
                    ]

                reasons_list.append(", ".join(reasons))
                suggestions_list.append(", ".join(suggestions))

            df_upload["Reasons"] = reasons_list
            df_upload["Suggestions"] = suggestions_list

            st.success(" Bulk prediction completed!")
            st.dataframe(df_upload)

    except Exception as e:
        st.error(f"Error during bulk prediction: {e}")