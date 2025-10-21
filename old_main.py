import glob
import json
import pandas as pd
import streamlit as st

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer

def load_all_slices(folder):
    files = glob.glob(f"{folder}/challenge_set.json")
    all_records = []
    for file in files:
        all_records.append(load_mpd_slice(file))
    return pd.concat(all_records, ignore_index=True)

def load_mpd_slice(path):
    with open(path, 'r') as f:
        data = json.load(f)
    playlists = data['playlists']
    records = []
    for p in playlists:
        for track in p['tracks']:
            records.append({
                'pid': p['pid'],
                'name': p.get('name', ''),
                'track_uri': track['track_uri'],
                'track_name': track['track_name'],
                'artist_name': track['artist_name']
            })
    return pd.DataFrame(records)

df = load_mpd_slice('challenge_set.json')
df = load_all_slices('path/to/folder')

def get_top_tracks(df, n=10):
    return df['track_name'].value_counts().head(n).index.tolist()

def build_user_track_matrix(df):
    grouped = df.groupby('pid')['track_uri'].apply(list)
    mlb = MultiLabelBinarizer()
    matrix = mlb.fit_transform(grouped)
    return pd.DataFrame(matrix, index=grouped.index, columns=mlb.classes_)

def recommend_similar_playlists(pid, matrix, df, top_n=5):
    if pid not in matrix.index:
        print(f"Playlist ID {pid} not found in matrix.")
        return []
    
    sim = cosine_similarity(matrix)
    pid_index = matrix.index.get_loc(pid)
    scores = list(enumerate(sim[pid_index]))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    scores = [s for s in scores if matrix.index[s[0]] != pid][:top_n]

    result = []
    playlist_meta = df.drop_duplicates('pid').set_index('pid')
    for i, score in scores:
        similar_pid = matrix.index[i]
        name = playlist_meta.loc[similar_pid]['name']
        tracks = df[df['pid'] == similar_pid]['track_name'].unique().tolist()

        result.append({
            'pid': similar_pid,
            'name': name,
            'similarity': round(score, 3),
            'preview_tracks': tracks
        })

    #similar_pids = [matrix.index[i] for i, _ in scores]
    #playlist_names = df.drop_duplicates('pid').set_index('pid').loc[similar_pids]['name'].tolist()
    #return list(zip(similar_pids, playlist_names))
    return result

def precision_at_k(recommended, relevant, k=10):
    recommended_k = recommended[:k]
    return len(set(recommended_k) & set(relevant)) / k

def coverage(df, recommended_lists):
    all_recommended = set([item for sublist in recommended_lists for item in sublist])
    all_items = set(df['track_uri'])
    return len(all_recommended) / len(all_items)

def split_playlist(tracks, split_ratio=0.5):
    split_point = int(len(tracks) * split_ratio)
    return tracks[:split_point], tracks[split_point:1]

st.title("Playlist Recommender DSS")
valid_pics = df['pid'].unique()
pid = st.selectbox("Choose a Playlist ID", valid_pics)

if st.button("Recommend"):
    matrix = build_user_track_matrix(df)
    if pid not in valid_pics:
        st.warning(f"Playlist ID {pid} not found. Try a different one.")
        st.write("Try one of these playlist IDs:", df['pid'].unique()[:10])
    else:
        recs = recommend_similar_playlists(pid, matrix, df)
        st.write("Similar Playlists:")
        for rec in recs:
            st.markdown(f"Playlist ID: {rec['pid']} - Name: {rec['name']}")
            st.markdown(f"Similarity Score: {rec['similarity']}")
            st.markdown("Track Preview:")
            for track in rec['preview_tracks']:
                st.write(f"- {track}")
            st.markdown("---")
        #for rec_pid, name in recs:
            #st.write(f"Playlist ID: {rec_pid} - Name: {name}")
        st.write("Top Tracks:", get_top_tracks(df))
        input_tracks, relevant_tracks = split_playlist(matrix.index[pid])
        st.write("Precision @k:", precision_at_k(recs, relevant_tracks))
        st.write("Coverage:", coverage(df, recs))
