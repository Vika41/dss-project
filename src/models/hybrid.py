from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

def simulate_user_vector(seed_tracks, track_ids):
    indices = [track_ids[t] for t in seed_tracks if t in track_ids]
    data = [1] * len(indices)
    return csr_matrix((data, ([0]*len(indices), indices)), shape=(1, len(track_ids)))

def get_cf_scores(model, user_vec, reverse_track_ids):
    recs = model.recommend(0, user_vec, N=len(reverse_track_ids))
    return {reverse_track_ids[i]: score for i, score in recs}

def get_cbf_scores(seed_tracks, track_features):
    seed_vecs = track_features.loc[track_features.index.intersection(seed_tracks)]
    if seed_vecs.empty:
        return {}
    mean_vec = seed_vecs.mean().values.reshape(1, -1)
    similarities = cosine_similarity(track_features.values, mean_vec).flatten()
    return dict(zip(track_features.index, similarities))

def blend_scores(cf_scores, cbf_scores, alpha=0.7, top_k=100):
    all_tracks = set(cf_scores) | set(cbf_scores)
    scores = {}
    for t in all_tracks:
        cf = cf_scores.get(t, 0)
        cbf = cbf_scores.get(t, 0)
        scores[t] = alpha * cf + (1 - alpha) * cbf
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
