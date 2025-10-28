import streamlit as st

from src.config import FOLDER_PATH

from src.models.contentbased import recommend_content_based

from src.preprocessing.buildtrackfeatures import build_track_features
from src.preprocessing.normalizefeatures import normalize_features

from src.utils.getseed import get_seed

def render_content_tab(playlists):
    st.header("🧬 Content-Based Recommendation")
    st.markdown("Recommend tracks using metadata and audio features like duration, position, and popularity — perfect for cold-start scenarios.")

    df = build_track_features(FOLDER_PATH)
    feature_cols = ["avg_duration_ms", "avg_position", "playlist_count"]
    track_features = normalize_features(df, feature_cols)

    for i, playlist in enumerate(playlists):
        category = (i // 1000) + 1
        seed_tracks = get_seed(playlist, category)
        predictions = recommend_content_based(seed_tracks, track_features)

    st.write("Top Content-Based Recommendations")
    for track in predictions[:10]:
        st.markdown(f"- {track}")
