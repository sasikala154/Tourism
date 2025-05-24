import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Set page title
st.set_page_config(page_title="Tourism Experience Rating Predictor", layout="centered")
st.title("🌍 Tourism Experience Rating Predictor")

# Load models and encoders
try:
    model_path = r"D:\Tourism\Experience model.pkl"
    ohe_path = r"D:\Tourism\experience_ohe.pkl"
    target_enc_path = r"D:\Tourism\experience_target_enc.pkl"

    model = joblib.load(model_path)
    ohe = joblib.load(ohe_path)
    target_enc = joblib.load(target_enc_path)

except FileNotFoundError as e:
    st.error(f"Model or encoder file not found: {e}")
    st.stop()

# Sidebar for user input
st.sidebar.header("Enter Attraction Details")

VisitModeName = st.sidebar.text_input("Visit Mode")
AttractionId = st.sidebar.text_input("Attraction ID")
Attraction = st.sidebar.text_input("Attraction Name")
AttractionType = st.sidebar.text_input("Attraction Type")
CountryId = st.sidebar.text_input("Country ID")
RegionId = st.sidebar.text_input("Region ID")

# Combine inputs into DataFrame
input_data = pd.DataFrame([{
    "VisitModeName": VisitModeName,
    "AttractionId": AttractionId,
    "Attraction": Attraction,
    "AttractionType": AttractionType,
    "CountryId": CountryId,
    "RegionId": RegionId
}])

st.subheader("🗃️ Input Data Preview")
st.dataframe(input_data)

# Prediction
if st.button("🔮 Predict Experience Rating"):
    try:
        # One-hot encode categorical variables
        encoded_cats = ohe.transform(input_data[["VisitModeName", "AttractionType"]])
        encoded_df = pd.DataFrame(encoded_cats, columns=ohe.get_feature_names_out(["VisitModeName", "AttractionType"]))

        # Target encode attraction name
        input_data["Attraction"] = target_enc.transform(input_data["Attraction"])

        # Factorize ID fields
        input_data["AttractionId"] = pd.factorize(input_data["AttractionId"])[0]
        input_data["CountryId"] = pd.factorize(input_data["CountryId"])[0]
        input_data["RegionId"] = pd.factorize(input_data["RegionId"])[0]

        # Drop encoded columns and concatenate final features
        input_data = input_data.drop(columns=["VisitModeName", "AttractionType"])
        final_input = pd.concat([input_data, encoded_df], axis=1)

        # Make prediction
        prediction = model.predict(final_input)[0]
        st.success(f"⭐ Predicted Experience Rating: {prediction:.2f}")

    except Exception as e:
        st.error(f"Prediction error: {e}")

# Optionally display dataset preview
if st.checkbox("📂 Show Sample Dataset"):
    try:
        df = pd.read_csv("tor.csv")
        st.dataframe(df.head(10))
    except FileNotFoundError:
        st.warning("Sample dataset 'tor.csv' not found.")
