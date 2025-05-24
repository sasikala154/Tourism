import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ------------------ Page Config ------------------
st.set_page_config(page_title="Tourism Experience Analytics", layout="centered")
st.title("🌍 Tourism Experience Analytics")

# ------------------ Helper Function ------------------
def load_model(path, name):
    if os.path.exists(path):
        return joblib.load(path)
    else:
        st.error(f"{name} not found at: {path}")
        st.stop()

# ------------------ Load Models ------------------
# Visit Mode Prediction
visit_model = load_model(r"D:\Tourism\model2.pkl", "Visit Mode Model")
visit_ohe = load_model(r"D:\Tourism\one_hot.pkl", "Visit OHE")
visit_label_enc = load_model(r"D:\Tourism\label_encoder.pkl", "Visit Label Encoder")
visit_target_enc = load_model(r"D:\Tourism\target_encoder.pkl", "Visit Target Encoder")

# Rating Prediction
rating_model = load_model(r"D:\Tourism\Rating model.pkl", "Rating Model")
rating_ohe = load_model(r"D:\Tourism\ohe.pkl", "Rating OHE")
rating_target_enc = load_model(r"D:\Tourism\target_enc.pkl", "Rating Target Encoder")

# Recommendation System
knn_model = load_model(r"D:\Tourism\knn_model.pkl", "KNN Recommendation Model")
user_matrix = load_model(r"D:\Tourism\user_attraction_matrix.pkl", "User-Attraction Matrix")
user_matrix_reduced = load_model(r"D:\Tourism\user_attraction_matrix_reduced.pkl", "Reduced Matrix")

# ------------------ Navigation ------------------
option = st.sidebar.selectbox("Select Feature", [
    "Visit Mode Prediction", 
    "Tourist Rating Prediction", 
    "Attraction Recommendation"
])

# ------------------ Visit Mode Prediction ------------------
if option == "Visit Mode Prediction":
    st.header("🌐 Predict Visit Mode")

    user_id = st.number_input("User ID", min_value=1)
    visit_mode = st.selectbox("Visit Mode (Encoded)", [0, 1, 2])
    attraction_id = st.number_input("Attraction ID", min_value=1)
    continent_id = st.number_input("Continent ID", min_value=1)
    region_id = st.number_input("Region ID", min_value=1)
    attraction = st.text_input("Attraction Name")
    attraction_type = st.selectbox("Attraction Type", ['Natural', 'Cultural', 'Recreational'])
    attraction_type_id = st.number_input("Attraction Type ID", min_value=1)

    if st.button("Predict Visit Mode"):
        visit_input = pd.DataFrame([{
            "UserId": user_id,
            "VisitMode": visit_mode,
            "AttractionId": attraction_id,
            "ContinentId": continent_id,
            "RegionId": region_id,
            "Attraction": attraction,
            "AttractionType": attraction_type,
            "AttractionTypeId": attraction_type_id
        }])

        encoded = visit_ohe.transform(visit_input[["VisitMode", "AttractionType"]])
        visit_input["Attraction"] = visit_target_enc.transform(visit_input["Attraction"])
        visit_input = visit_input.drop(columns=["VisitMode", "AttractionType"])
        final_visit_input = pd.concat([visit_input.reset_index(drop=True), pd.DataFrame(encoded, columns=visit_ohe.get_feature_names_out())], axis=1)

        result = visit_model.predict(final_visit_input)[0]
        decoded_result = visit_label_enc.inverse_transform([result])[0]
        st.success(f"Predicted Visit Mode: {decoded_result}")

# ------------------ Tourist Rating Prediction ------------------
elif option == "Tourist Rating Prediction":
    st.header("🌟 Predict Tourist Rating")

    vm = st.text_input("Visit Mode")
    aid = st.text_input("Attraction ID")
    aname = st.text_input("Attraction Name")
    atype = st.text_input("Attraction Type")
    cid = st.text_input("Country ID")
    rid = st.text_input("Region ID")

    if st.button("Predict Rating"):
        rating_input = pd.DataFrame([{
            "VisitModeName": vm,
            "AttractionId": aid,
            "Attraction": aname,
            "AttractionType": atype,
            "CountryId": cid,
            "RegionId": rid
        }])

        enc = rating_ohe.transform(rating_input[["VisitModeName", "AttractionType"]])
        rating_input["Attraction"] = rating_target_enc.transform(rating_input["Attraction"])
        rating_input["AttractionId"] = pd.factorize(rating_input["AttractionId"])[0]
        rating_input["CountryId"] = pd.factorize(rating_input["CountryId"])[0]
        rating_input["RegionId"] = pd.factorize(rating_input["RegionId"])[0]
        rating_input = rating_input.drop(columns=["VisitModeName", "AttractionType"])
        final_rating_input = pd.concat([rating_input.reset_index(drop=True), pd.DataFrame(enc, columns=rating_ohe.get_feature_names_out())], axis=1)

        prediction = rating_model.predict(final_rating_input)
        st.success(f"Predicted Rating: {prediction[0]:.2f}")

# ------------------ Attraction Recommendation ------------------
elif option == "Attraction Recommendation":
    st.header("🧭 Attraction Recommendation")

    user_id_list = user_matrix.index.tolist()
    selected_user = st.selectbox("Select User ID", user_id_list)

    def recommend_attractions(uid, top_n=5):
        idx = user_matrix.index.get_loc(uid)
        reduced_vector = user_matrix_reduced[idx].reshape(1, -1)

        # Prevent exceeding available samples
        max_neighbors = min(top_n + 1, knn_model.n_samples_fit_)
        if max_neighbors < 2:
            st.warning("Not enough data to generate recommendations.")
            return []

        distances, indices = knn_model.kneighbors(reduced_vector, n_neighbors=max_neighbors)
        neighbors = indices.flatten()[1:]  # exclude the user

        neighbor_ratings = user_matrix.iloc[neighbors]
        user_ratings = user_matrix.iloc[idx]
        unseen = user_ratings[user_ratings == 0].index

        scores = {attr: neighbor_ratings[attr].sum() for attr in unseen}
        sorted_attrs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [attr for attr, _ in sorted_attrs[:top_n]]

    if st.button("Get Recommendations"):
        results = recommend_attractions(selected_user)
        if results:
            st.subheader("Recommended Attractions:")
            for i, r in enumerate(results, 1):
                st.write(f"{i}. {r}")
        else:
            st.info("No recommendations available for this user.")
