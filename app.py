import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.title("Customer Churn Predictor")
st.write("""
This app predicts the likelihood of customer churn using an XGBoost model trained on the Telco Customer Churn dataset.
Provide customer details in the sidebar and click Predict to see the result.
""")

# Load model and encoder
try:
    model = joblib.load("models/final_xgboost_model.joblib")
    # Disable feature name validation for XGBoost
    model.get_booster().feature_names = None
except:
    st.error("Error loading model")
    
encoder = joblib.load("models/input_encoder.pkl")

st.sidebar.header("Customer Data Input")

def user_input_features():
    gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
    SeniorCitizen = st.sidebar.selectbox("Senior Citizen", [0, 1])
    Partner = st.sidebar.selectbox("Partner", ["No", "Yes"])
    Dependents = st.sidebar.selectbox("Dependents", ["No", "Yes"])
    tenure = st.sidebar.slider("Tenure (months)", 0, 72, 1)
    PhoneService = st.sidebar.selectbox("Phone Service", ["No", "Yes"])
    MultipleLines = st.sidebar.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    InternetService = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.sidebar.selectbox("Online Security", ["No", "Yes", "No internet service"])
    OnlineBackup = st.sidebar.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    DeviceProtection = st.sidebar.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    TechSupport = st.sidebar.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    StreamingTV = st.sidebar.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    StreamingMovies = st.sidebar.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    Contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.sidebar.selectbox("Paperless Billing", ["No", "Yes"])
    PaymentMethod = st.sidebar.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    MonthlyCharges = st.sidebar.slider("Monthly Charges", 0.0, 120.0, 20.0)
    TotalCharges = st.sidebar.slider("Total Charges", 0.0, 9000.0, 100.0)
    data = {
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()


# Match feature order to training (from processed_data.csv)
feature_order = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges"
]

# Encode categorical features except SeniorCitizen and numerical columns
cat_cols = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod"
]
input_df[cat_cols] = encoder.transform(input_df[cat_cols])

# Scale numerical features
from sklearn.preprocessing import StandardScaler
num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
scaler = StandardScaler()
input_df[num_cols] = scaler.fit_transform(input_df[num_cols])

# Ensure correct feature order and only features used in training
input_df = input_df[feature_order]

if st.button("Predict Churn"):
    # Convert to numpy array like in training (X = df.iloc[:,:-1].to_numpy())
    X_input = input_df.to_numpy()
    
    prediction = model.predict(X_input)
    proba = model.predict_proba(X_input)[0][1]
    st.subheader("Prediction Result")
    st.write(f"Churn Probability: {proba:.2f}")
    if prediction[0] == 1:
        st.error("This customer is likely to churn.")
    else:
        st.success("This customer is unlikely to churn.")
     