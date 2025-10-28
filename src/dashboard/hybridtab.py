import streamlit as st

from sklearn.metrics.pairwise import cosine_similarity

from src.config import FOLDER_PATH

from src.models.hybrid import simulate_user_vector, get_cf_scores, get_cb_scores, normalize_dict, blend_multi_scores

from src.preprocessing.buildtrackfeatures import build_track_features
from src.preprocessing.normalizefeatures import normalize_features


def render_hybrid_tab(playlists, playlists_cf, track_meta, item_sim_matrix, als_model, track_index, playlist_index, popularity_dict):
    st.header("🔀 Hybrid Recommendation System")
    st.markdown("Blend collaborative filtering with content-based similarity for a balanced, robust recommendation strategy.")

    track_features_raw = build_track_features(FOLDER_PATH)
    track_features = normalize_features(track_features_raw, ["avg_duration_ms", "avg_position", "playlist_count"])

    method = st.selectbox("Choose hybrid strategy", ["ALS + CBF", "ALS + ItemCF + CBF"])
    seed_input = st.text_area("Enter seed track URIs (comma-separated)")
    seed_tracks = [uri.strip() for uri in seed_input.split(",") if uri.strip()]

    if st.button("Generate Recommendations") and seed_tracks:
        user_vec = simulate_user_vector(seed_tracks, track_index)
        cf_scores_als = normalize_dict(get_cf_scores(als_model, user_vec, {v: k for k, v in track_index.items()}))
        cb_scores = normalize_dict(get_cb_scores([seed_tracks], track_features))

        if method == 'ALS + CBF':
            blended = blend_multi_scores([cf_scores_als, cb_scores], weights=[0.7, 0.3], top_k=50)

        elif method == 'ALS + ItemCF + CBF':
            seed_indices = [track_index[t] for t in seed_tracks if t in track_index]
            item_scores = cosine_similarity(item_sim_matrix[seed_indices], item_sim_matrix).mean(axis=0)
            cf_scores_item = normalize_dict({uri: item_scores[i] for uri, i in track_index.items()})
            blended = blend_multi_scores([cf_scores_als, cf_scores_item, cb_scores], weights=[0.5, 0.2, 0.3], top_k=50)

    st.subheader("Top Recommendations")
    for uri, score in blended:
        meta = track_meta.get(uri, {})
        st.markdown(f"**{meta.get('track_name', 'Unknown')}** by *{meta.get('artist_name', 'Unknown')}*")
        st.caption(f"Album: {meta.get('album_name', 'Unknown')} | URI: {uri} | Score: {score:.4f}")

    #category = st.selectbox("Choose Challenge Category", list(range(1, 11)))
    #filtered = [p for i, p in enumerate(playlists) if (i // 1000 + 1) == category]
    #sample = st.slider("Select Playlist Index", 0, len(filtered) - 1)
    #playlist = filtered[sample]
    #seed_tracks, title = get_seed(playlist, category)

    #user_vec = simulate_user_vector(seed_tracks, track_ids)
    #cf_scores_als = get_cf_scores(als_model, user_vec, reverse_track_ids)
    #cf_scores_new 
    #cb_scores = get_cbf_scores(seed_tracks, track_features)

    #cf_scores_als = normalize_dict(cf_scores_als)
    #cf_scores_new
    #cb_scores = normalize_dict(cb_scores)

    #hybrid_preds = blend_scores(cf_scores_als, cb_scores, alpha=0.7)

    #st.write(f"**Title:** {title}")
    #st.write(f"**Seed Tracks:** {seed_tracks}")
    #st.write("**Top Hybrid Recommendations:**")
    #for track, score in hybrid_preds[:10]:
    #    st.markdown(f"- {track} ({score:.3f})")

