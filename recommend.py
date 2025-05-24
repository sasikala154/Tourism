import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Set page title and layout
st.set_page_config(page_title="Tourism Attraction Recommender", layout="centered")
st.title("🏝️ Tourism Attraction Recommender System")

# File paths
knn_model_path = r"D:\Tourism\knn_model.pkl"
user_attraction_matrix_path = r"D:\Tourism\user_attraction_matrix.pkl"
user_attraction_reduced_path = r"D:\Tourism\user_attraction_matrix_reduced.pkl"  # Optional

# Load models and matrices
try:
    knn_model = joblib.load(knn_model_path)
    user_attraction_matrix = joblib.load(user_attraction_matrix_path)
    user_attraction_reduced = joblib.load(user_attraction_reduced_path)
except FileNotFoundError as e:
    st.error(f"❌ File not found: {e.filename}")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading models: {e}")
    st.stop()

# UI: User selection
user_id_list = user_attraction_matrix.index.tolist()
selected_user_id = st.selectbox("Choose a User ID", user_id_list)

def recommend_attractions(user_id, top_n=5):
    """
    Recommend top N attractions for the given user_id based on KNN collaborative filtering.
    """
    total_users = user_attraction_reduced.shape[0]
    
    # Ensure we don’t exceed available neighbors
    if total_users <= 1:
        return []

    n_neighbors = min(top_n + 1, total_users)

    user_index = user_attraction_matrix.index.get_loc(user_id)
    user_vector_reduced = user_attraction_reduced[user_index].reshape(1, -1)

    try:
        distances, indices = knn_model.kneighbors(user_vector_reduced, n_neighbors=n_neighbors)
    except ValueError as e:
        st.error(f"KNN error: {e}")
        return []

    # Exclude the user itself
    neighbor_indices = indices.flatten()
    neighbor_indices = [i for i in neighbor_indices if i != user_index]

    neighbors_ratings = user_attraction_matrix.iloc[neighbor_indices]
    user_ratings = user_attraction_matrix.iloc[user_index]

    unseen_attractions = user_ratings[user_ratings == 0].index

    scores = {
        attraction: neighbors_ratings[attraction].sum()
        for attraction in unseen_attractions
    }

    recommended = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [attraction for attraction, _ in recommended[:top_n]]

# Run recommendation
if st.button("Get Recommendations"):
    recommended_attractions = recommend_attractions(selected_user_id, top_n=5)
    if recommended_attractions:
        st.subheader("🎯 Recommended Attractions:")
        for i, attraction in enumerate(recommended_attractions, start=1):
            st.markdown(f"**{i}. {attraction}**")
    else:
        st.info("No recommendations available for this user or not enough data.")

# Dataset Preview (optional)
try:
    df = pd.read_csv("tor.csv")
    st.subheader("📊 Sample Tourism Dataset Preview")
    st.dataframe(df.head(10))
except FileNotFoundError:
    st.warning("Sample dataset file 'tor.csv' not found.")
