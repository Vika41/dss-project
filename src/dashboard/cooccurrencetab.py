import streamlit as st

from src.models.cooccurrence import build_co_matrix, get_popularity, predict

from src.utils.getseed import get_seed

def render_co_tab(playlists):
    st.header("🎯 Co-occurrence-Based Recommendation")
    st.markdown("Predict tracks based on co-listening patterns across playlists. Ideal for capturing track-to-track relationships.")

    co_matrix = build_co_matrix(playlists)
    popularity = get_popularity(playlists)

    for i, playlist in enumerate(playlists[:10000]):
        category = (i // 1000) + 1
        seed_tracks = get_seed(playlist, category)
        predictions = predict(seed_tracks, co_matrix, popularity)

    st.write("**Predicted Tracks:**")
    for track in predictions[:10]:
        st.markdown(f"- {track}")