import implicit
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix, lil_matrix
from sklearn.metrics.pairwise import cosine_similarity

def build_interaction_df(playlists):
    rows = []
    for p in playlists:
        pid = p['pid']
        for t in p.get("tracks", []):
            uri = t.get("track_uri")
            if uri:
                rows.append({
                    "playlist_id": pid,
                    "track_uri": uri,
                    "interaction": 1
                })
    return pd.DataFrame(rows)

def df_to_sparse_matrix(df):
    playlist_ids = df['playlist_id'].unique()
    track_uris = df['track_uri'].unique()

    playlist_index = {pid: i for i, pid in enumerate(playlist_ids)}
    track_index = {uri: i for i, uri in enumerate(track_uris)}

    row = df['playlist_id'].map(playlist_index)
    col = df['track_uri'].map(track_index)
    data = df['interaction']

    matrix = csr_matrix((data, (row, col)), shape=(len(playlist_ids), len(track_uris)))
    return matrix, playlist_index, track_index

def compute_track_similarity(matrix):
    track_track_sim = cosine_similarity(matrix.T)
    return track_track_sim

def train_als_model(matrix, factors=64, regularization=0.1, iterations=15):
    model = implicit.als.AlternatingLeastSquares(factors=factors, regularization=regularization, iterations=iterations)
    model.fit(matrix.T)
    return model

def recommend_from_seed(seed_uris, track_index, sim_matrix, top_k=10):
    seed_indices = [track_index[uri] for uri in seed_uris if uri in track_index]
    if not seed_indices:
        return []
    
    sim_scores = sim_matrix[seed_indices].mean(axis=0)
    top_indices = np.argsort(sim_scores)[::-1][:top_k]
    inv_index = {i: uri for uri, i in track_index.items()}
    return [inv_index[i] for i in top_indices if i not in seed_indices]

def recommend_from_als(model, user_index, track_index, playlist_id, top_k=10):
    if playlist_id not in user_index:
        return []
    user_id = user_index[playlist_id]
    ids, _ = model.recommend(user_id, model.user_factors[user_id], N=top_k)
    inv_index = {i: uri for uri, i in track_index.items()}
    return [inv_index[i] for i in ids]

def build_interaction_matrix(playlists):
    track_set = set()
    for p in playlists:
        for t in p.get("tracks", []):
            uri = t.get("track_uri")
            if uri:
                track_set.add(uri)

    if not track_set or not playlists:
        raise ValueError("No valid tracks or playlists found.")
    
    track_ids = sorted(track_set)
    track_index = {uri: i for i, uri in enumerate(track_ids)}
    matrix = lil_matrix((len(playlists), len(track_ids)), dtype=np.float32)

    for i, p in enumerate(playlists):
        for t in p.get("tracks", []):
            uri = t.get("track_uri")
            if uri in track_index:
                matrix[i, track_index[uri]] = 1.0
    return matrix, track_ids

def train_cf_model(matrix, factors=50):
    model = implicit.als.AlternatingLeastSquares(factors=factors)
    model.fit(matrix.T)
    return model

def simulate_user_vector(seed_tracks, track_ids):
    indices = [track_ids[t] for t in seed_tracks if t in track_ids]
    data = [1] * len(indices)
    return csr_matrix((data, ([0]*len(indices), indices)), shape=(1, len(track_ids)))

def recommend_cf_for_playlist(seed_tracks, model, track_ids, reverse_track_ids, top_k=100):
    user_vec = simulate_user_vector(seed_tracks, track_ids)
    recs = model.recommend(0, user_vec, N=top_k)
    return [reverse_track_ids[i] for i, _ in recs]
