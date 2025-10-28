import streamlit as st

from models.old_collaborative import build_interaction_df, build_interaction_matrix, train_als_model

from src.utils.getseed import get_seed
from src.utils.splitdata import load_all_playlists, split_playlists, load_all

def render_cf_tab():
    st.header("👥 Collaborative Filtering Recommendation")
    st.markdown("Leverage latent patterns in playlist-track interactions using matrix factorization. Great for capturing user behavior.")

    #all_playlists = load_all_playlists('path/to/json')
    all_playlists = load_all('path/to/')
    train_playlists, challenge_playlists = split_playlists(all_playlists)
    #filtered = [
    #    p for p in train_playlists
    #    if any("track_uri" in t and t["track_uri"] for t in p.get("tracks", []))
    #]
    matrix, track_ids = build_interaction_matrix(train_playlists)
    model = train_cf_model(matrix)
    reverse_track_ids = {v: k for k, v in track_ids.items()}

    category = st.selectbox("Choose Challenge Category", list(range(1, 11)))
    filtered = [p for i, p in enumerate(challenge_playlists) if (i // 1000 + 1) == category]
    sample = st.slider("Select Playlist Index", 0, len(filtered) - 1)
    playlist = filtered[sample]

    seed_tracks, title = get_seed(playlist, category)
    st.write(f"**Title:** {title}")
    st.write(f"**Seed Tracks:** {seed_tracks}")

    predictions = recommend_cf_for_playlist(seed_tracks, model, track_ids, reverse_track_ids)

    st.write("**Top Collaborative Recommendations:**")
    for track in predictions[:10]:
        st.markdown(f"- {track}")
