import numpy as np
import pandas as pd

from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

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

def train_als_model(matrix, factors=64, regularization=0.1, iterations=15):
    model = AlternatingLeastSquares(factors=factors, regularization=regularization, iterations=iterations)
    model.fit(matrix.T)
    return model

def compute_track_similarity(matrix):
    return cosine_similarity(matrix.T)

def normalize_scores(scores):
    scaler = MinMaxScaler()
    return scaler.fit_transform(scores.reshape(-1, 1)).flatten()

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

def hybrid_score(item_scores, als_scores, popularity_scores, weights=(0.5, 0.3, 0.2)):
    w_item, w_als, w_pop = weights
    return (
        w_item * normalize_scores(item_scores) +
        w_als * normalize_scores(als_scores) +
        w_pop * normalize_scores(popularity_scores)
    )

def rank_tracks(track_uris, hybrid_scores, seed_uris, top_k=10):
    scored = [(uri, score) for uri, score in zip(track_uris, hybrid_scores) if uri not in seed_uris]
    return sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]

def get_recommendations(seed_uris, method, item_sim_matrix, als_model, popularity_dict, track_index, playlist_index=None, seed_playlist_id=None, top_k=10):
    track_uris = list(track_index.keys())

    if method == 'item':
        seed_indices = [track_index[uri] for uri in seed_uris if uri in track_index]
        item_scores = item_sim_matrix[seed_indices].mean(axis=0)
        return rank_tracks(track_uris, item_scores, seed_uris, top_k)
    
    elif method == 'als':
        if seed_playlist_id not in playlist_index:
            return []
        
        user_id = playlist_index[seed_playlist_id]
        ids, _ = als_model.recommend(user_id, als_model.user_factors[user_id], N=top_k)
        inv_index = {i: uri for uri, i in track_index.items()}
        return [(inv_index[i], None) for i in ids if inv_index[i] not in seed_uris]
    
    elif method == 'hybrid':
        seed_indices = [track_index[uri] for uri in seed_uris if uri in track_index]
        item_scores = item_sim_matrix[seed_indices].mean(axis=0) if seed_indices else np.zeros(len(track_index))
        als_scores = np.zeros(len(track_index))

        for uri in seed_uris:
            if uri in track_index:
                als_scores += np.array([score for _, score in als_model.similar_items(track_index[uri], N=len(track_index))])
        
        popularity_scores = np.array([popularity_dict.get(uri, 0) for uri in track_uris])
        hybrid_scores = hybrid_score(item_scores, als_scores, popularity_scores)
        return rank_tracks(track_uris, hybrid_scores, seed_uris, top_k)
    
    else:
        raise ValueError(f"Unknown method: {method}.")