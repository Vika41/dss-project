import streamlit as st

#from src.evaluation.export import export_predictions

#from src.dashboard.collaborativetab import render_cf_tab
from src.dashboard.contentbasedtab import render_content_tab
from src.dashboard.cooccurrencetab import render_co_tab
#from src.dashboard.evaluationtab import render_evaluation_tab
#from src.dashboard.hybridtab import render_hybrid_tab

#from src.models.cooccurrence import load_playlists, build_co_matrix, get_popularity
#from src.models.predictor import predict_challenge

st.set_page_config(page_title="Spotify MPD DSS", layout="wide")

st.title("🎧 Spotify Million Playlist DSS Dashboard")

#def render_predictor_tab():
    #st.header("🧠 Playlist Predictor")
    #st.markdown("Explore track predictions across the 10 official challenge categories. Select a playlist and input type — title, seed tracks, or both — and preview how different DSS models respond.")

    #folder = 'c:/Users/victo/Documents/GitHub/dss-project/data'
    #json_dir = folder

    #playlists = load_playlists(json_dir)
    #co_matrix = build_co_matrix(playlists)
    #popularity = get_popularity(playlists)

    #filtered = [p for i, p in enumerate(playlists) if (i // 1000 + 1)]
    #sample = st.slider("Select Playlist Index", 0, len(filtered) - 1)
    #playlist = filtered[sample]

    #for i, playlist in enumerate(playlists[:10000]):
    #    category = (i // 1000) + 1
    #    predictions = predict_challenge(playlist, category, co_matrix, popularity)

    #sort = sorted(predictions, key=lambda x: x['pid'])
    #export = export_predictions(sort, 'c:/Users/victo/Documents/GitHub/dss-project/outputs/predictions_predictor.json')

    #st.write("**Predicted Tracks:**")
    #for track in predictions[:10]:
    #    st.markdown(f"- {track}")


tab1, tab2, tab3, tab4, tab5 = st.tabs([
    #"Predictor",
    "Co-occurrence",
    "Content-Based",
    "Collaborative",
    "Hybrid",
    "Evaluation"
])

with tab1:
    #render_predictor_tab()
    render_co_tab()

with tab2:
    #render_co_tab()
    render_content_tab()

#with tab3:
    #render_content_tab()
#    render_cf_tab()

#with tab4:
    #render_cf_tab()
#    render_hybrid_tab()

#with tab5:
    #render_hybrid_tab()
#    render_evaluation_tab()

#with tab6:
#   render_evaluation_tab()
