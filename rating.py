import streamlit as st
import pandas as pd
import joblib
import numpy as np

model_path = r"D:\Tourism\Rating model.pkl"
ohe_path = r"D:\Tourism\ohe.pkl"
target_enc_path = r"D:\Tourism\target_enc.pkl"

best_xgb = joblib.load(model_path)
ohe = joblib.load(ohe_path)
target_enc = joblib.load(target_enc_path)

st.title("Tourism Attraction Rating Prediction")

st.sidebar.header("Enter Attraction Details")

VisitModeName = st.sidebar.text_input("Visit Mode")
AttractionId = st.sidebar.text_input("Attraction ID")
Attraction = st.sidebar.text_input("Attraction Name")
AttractionType = st.sidebar.text_input("Attraction Type")
CountryId = st.sidebar.text_input("Country ID")
RegionId = st.sidebar.text_input("Region ID")

input_data = pd.DataFrame([{
    "VisitModeName": VisitModeName,
    "AttractionId": AttractionId,
    "Attraction": Attraction,
    "AttractionType": AttractionType,
    "CountryId": CountryId,
    "RegionId": RegionId
}])

st.subheader("Input Data Preview")
df = pd.read_csv("tor.csv")
st.dataframe(df)

try:
    encoded_features = ohe.transform(input_data[["VisitModeName", "AttractionType"]])
    encoded_df = pd.DataFrame(encoded_features, columns=ohe.get_feature_names_out(["VisitModeName", "AttractionType"]))

    input_data["Attraction"] = target_enc.transform(input_data["Attraction"])

    input_data["AttractionId"] = pd.factorize(input_data["AttractionId"])[0]
    input_data["CountryId"] = pd.factorize(input_data["CountryId"])[0]
    input_data["RegionId"] = pd.factorize(input_data["RegionId"])[0]

    input_data = input_data.drop(columns=["VisitModeName", "AttractionType"])
    input_data = pd.concat([input_data, encoded_df], axis=1)

    prediction = best_xgb.predict(input_data)
    st.success(f"Predicted Tourism Rating: {prediction[0]:.2f}")

except Exception as e:
    st.error(f"Prediction failed: {e}")