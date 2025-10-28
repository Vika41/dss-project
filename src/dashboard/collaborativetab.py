import streamlit as st

from src.models.collaborative import get_recommendations

def render_cf_tab(playlists, track_meta, item_sim_matrix, als_model, track_index, playlist_index, popularity_dict):
    #st.header("🎧 Playlist Continuation")
    st.header("👥 Collaborative Filtering Recommendation")
    st.markdown("Leverage latent patterns in playlist-track interactions using matrix factorization. Great for capturing user behavior.")
    st.markdown("Generate track recommendations using collaborative filtering.")

    method = st.selectbox("Choose recommendation method", ["item", "als", "hybrid"])
    seed_uris = st.text_area("Enter seed track URIs (comma-separated)").split(",")
    seed_uris = [uri.strip() for uri in seed_uris if uri.strip()]
    seed_playlist_id = st.text_input("Optional: Seed playlist ID (for ALS)", value="")

    if st.button("Generate Recommendations"):
        recs = get_recommendations(
            seed_uris=seed_uris,
            method=method,
            item_sim_matrix=item_sim_matrix,
            als_model=als_model,
            popularity_dict=popularity_dict,
            track_index=track_index,
            playlist_index=playlist_index,
            seed_playlist_id=int(seed_playlist_id) if seed_playlist_id.isdigit() else None,
            top_k=10
        )

        if not recs:
            st.warning("No recommendations found.")
        else:
            st.subheader("Top Recommendations")
            for uri, score in recs:
                meta = track_meta.get(uri, {})
                st.markdown(f"**{meta.get('track_name', 'Unknown')}** by *{meta.get('artist_name', 'Unknown')}*")
                st.caption(f"Album: {meta.get('album_name', 'Unknown')} | URI: {uri}")
