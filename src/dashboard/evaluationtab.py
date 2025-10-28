import streamlit as st

from sklearn.metrics.pairwise import cosine_similarity

from src.config import FOLDER_PATH

from src.evaluation.evaluate import evaluate, get_ground_truth_tracks, evaluate_k

from src.models.collaborative import recommend_from_als
from src.models.contentbased import recommend_content_based
from src.models.cooccurrence import get_popularity, build_co_matrix, predict_challenge
from src.models.hybrid import simulate_user_vector, get_cb_scores, get_cf_scores, blend_scores, normalize_dict, blend_multi_scores

from src.preprocessing.buildtrackfeatures import build_track_features
from src.preprocessing.normalizefeatures import normalize_features

from src.utils.getseed import get_seed

def render_evaluation_tab(playlists, playlists_cf, track_meta, item_sim_matrix, als_model, track_index, playlist_index, popularity_dict):
    st.header("📊 DSS Evaluation Dashboard")
    st.markdown("Compare Co-occurrence, Content-Based, Collaborative, and Hybrid models using R-Precision, NDCG, and Clicks.")

    # Load data
    co_matrix = build_co_matrix(playlists)
    popularity = get_popularity(playlists)
    track_features_raw = build_track_features(FOLDER_PATH)
    track_features = normalize_features(track_features_raw, ["avg_duration_ms", "avg_position", "playlist_count"])

    # Select challenge playlist
    category = st.selectbox("Challenge Category", list(range(1, 11)))
    #filtered = [p for i, p in enumerate(playlists) if (i // 1000 + 1) == category]
    #sample = st.slider("Playlist Index", 0, len(filtered) - 1)
    #playlist = filtered[sample]
    #seed_tracks, title = get_seed(playlist, category)
    #ground_truth = [t["track_uri"] for t in playlist["tracks"] if t["track_uri"] not in seed_tracks]

    playlist_id = st.number_input("Playlist ID to evaluate", min_value=0, max_value=len(playlists)-1, step=1)
    k = st.slider("Top-K", min_value=1, max_value=100, value=10)
    category = st.selectbox("Challenge Category", list(range(1, 11)))

    if st.button("Evaluate"):
        playlist = playlists[playlist_id]
        seed, title = get_seed(playlist, category)
        seed_tracks = [t["track_uri"] for t in playlist.get("tracks", [])[:5]]
        ground_truth = get_ground_truth_tracks(playlist)

        user_vec = simulate_user_vector(seed_tracks, track_index)
        seed_indices = [track_index[t] for t in seed_tracks if t in track_index]
        user_id = playlist_index[playlist]

        # Run all DSS models
        co_preds = predict_challenge(playlist, category, co_matrix, popularity)
        cb_preds = recommend_content_based(seed_tracks, track_features)
        cf_preds_als = recommend_from_als(als_model, user_id, track_index, playlist_id)
        cf_scores_als = normalize_dict(get_cf_scores(als_model, user_vec, {v: k for k, v in track_index.items()}))
        cb_scores = normalize_dict(get_cb_scores([seed_tracks], track_features))
        item_scores = cosine_similarity(item_sim_matrix[seed_indices], item_sim_matrix.mean(axis=0))
        cf_scores_item = normalize_dict({uri: item_scores[i] for uri, i in track_index.items()})
        hybrid_preds = [t for t, _ in blend_scores(cf_preds_als, cb_preds, alpha=0.7)]
        blended = blend_multi_scores([cf_scores_als, cf_scores_item, cb_scores], weights=[0.5, 0.2, 0.3], top_k=100)
        hybrid_preds_blended = [uri for uri, _ in blended]

    st.write(f"**Title:** {title}")
    st.write(f"**Seed Tracks:** {seed_tracks}")
    st.write(f"**Ground Truth Tracks:** {ground_truth[:5]}...")

    # Evaluate
    results = {
        "Co-occurrence": evaluate(co_preds, ground_truth),
        "Content-Based": evaluate(cb_preds, ground_truth),
        "Collaborative (ALS)": evaluate(cf_preds_als, ground_truth),
        "Hybrid": evaluate(hybrid_preds, ground_truth),
        "Hybrid @ k": evaluate_k(hybrid_preds_blended, ground_truth, k)
    }

    # Display
    st.subheader("📈 Evaluation Results")
    for model_name, metrics in results.items():
        st.markdown(f"**{model_name}**")
        st.write(metrics)
