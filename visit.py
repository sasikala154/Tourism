import streamlit as st
import pandas as pd
import numpy as np
import joblib

from category_encoders import TargetEncoder

model = joblib.load(r"D:\Tourism\model2.pkl")
one_hot_encoder = joblib.load(r"D:\Tourism\one_hot.pkl")
label_encoder = joblib.load(r"D:\Tourism\label_encoder.pkl")
target_encoder = joblib.load(r"D:\Tourism\target_encoder.pkl")

st.title("Visit Mode Prediction App")

user_id = st.number_input("User ID", min_value=1)
visit_mode = st.selectbox("Visit Mode (Encoded)", [0, 1, 2])
attraction_id = st.number_input("Attraction ID", min_value=1)
continent_id = st.number_input("Continent ID", min_value=1)
region_id = st.number_input("Region ID", min_value=1)
attraction = st.text_input("Attraction Name")
attraction_type = st.selectbox("Attraction Type", ['Natural', 'Cultural', 'Recreational'])
attraction_type_id = st.number_input("Attraction Type ID", min_value=1)

if st.button("Predict Visit Mode"):
    input_dataframe = pd.DataFrame([{
        "UserId": user_id,
        "VisitMode": visit_mode,
        "AttractionId": attraction_id,
        "ContinentId": continent_id,
        "RegionId": region_id,
        "Attraction": attraction,
        "AttractionType": attraction_type,
        "AttractionTypeId": attraction_type_id
    }])

    categorical_columns = ["VisitMode", "AttractionType"]
    encoded_categorical = one_hot_encoder.transform(input_dataframe[categorical_columns])
    encoded_categorical_df = pd.DataFrame(encoded_categorical, columns=one_hot_encoder.get_feature_names_out(categorical_columns))

    input_dataframe["Attraction"] = target_encoder.transform(input_dataframe["Attraction"])
    input_dataframe = input_dataframe.drop(columns=categorical_columns)
    final_input = pd.concat([input_dataframe.reset_index(drop=True), encoded_categorical_df], axis=1)

    prediction_encoded = model.predict(final_input)[0]
    prediction_decoded = label_encoder.inverse_transform([prediction_encoded])[0]

    st.success(f"Predicted Visit Mode: {prediction_decoded}")