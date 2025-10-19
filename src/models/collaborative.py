import implicit

from scipy.sparse import coo_matrix, csr_matrix

def build_interaction_matrix(playlists):
    track_ids = {}
    rows, cols, data = [], [], []

    for pid, playlist in enumerate(playlists):
        for track in playlist['tracks']:
            tid = track['track_uri']
            if tid not in track_ids:
                track_ids[tid] = len(track_ids)
            rows.append(pid)
            cols.append(track_ids[tid])
            data.append(1)

    matrix = coo_matrix((data, (rows, cols)))
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
