import streamlit as st

from src.config import FOLDER_PATH, JSON_PATH

from src.dashboard.collaborativetab import render_cf_tab
from src.dashboard.contentbasedtab import render_content_tab
from src.dashboard.cooccurrencetab import render_co_tab
from src.dashboard.evaluationtab import render_evaluation_tab
from src.dashboard.hybridtab import render_hybrid_tab

from src.utils.loaders import load_playlists, load_playlists_cf, load_track_meta, build_models

st.set_page_config(page_title="Spotify MPD DSS", layout="wide")

st.title("🎧 Spotify Million Playlist DSS Dashboard")

playlists_cf = load_playlists_cf(JSON_PATH)
playlists = load_playlists(FOLDER_PATH)
track_meta = load_track_meta(playlists_cf)
item_sim_matrix, als_model, popularity_dict, track_index, playlist_index = build_models(playlists_cf)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Co-occurrence",
    "Content-Based",
    "Collaborative",
    "Hybrid",
    "Evaluation"
])

with tab1:
    render_co_tab(playlists)

with tab2:
    render_content_tab(playlists)

with tab3:
    render_cf_tab(playlists_cf, track_meta, item_sim_matrix, als_model, track_index, playlist_index, popularity_dict)

with tab4:
    render_hybrid_tab(playlists, playlists_cf, track_meta, item_sim_matrix, als_model, track_index, playlist_index, popularity_dict)

with tab5:
    render_evaluation_tab(playlists, playlists_cf, track_meta, item_sim_matrix, als_model, track_index, playlist_index, popularity_dict)