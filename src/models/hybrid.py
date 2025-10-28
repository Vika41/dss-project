import numpy as np

from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

def normalize_dict(d):
    items = list(d.items())
    if not items:
        return {}
    uris, vals = zip(*items)
    scaled = MinMaxScaler().fit_transform(np.array(vals).reshape(-1, 1)).flatten()
    return dict(zip(uris, scaled))

def simulate_user_vector(seed_tracks, track_ids):
    indices = [track_ids[t] for t in seed_tracks if t in track_ids]
    data = [1] * len(indices)
    return csr_matrix((data, ([0]*len(indices), indices)), shape=(1, len(track_ids)))

def get_cf_scores(model, user_vec, reverse_track_ids):
    recs = model.recommend(0, user_vec, N=len(reverse_track_ids))
    return {reverse_track_ids[i]: score for i, score in recs}

def get_cb_scores(seed_tracks, track_features):
    flat_seed_tracks = [track for sublist in seed_tracks for track in sublist]
    seed_vecs = track_features.loc[track_features.index.intersection(flat_seed_tracks)]
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

def blend_multi_scores(score_dicts, weights, top_k=100):
    all_tracks = set().union(*[set(d.keys()) for d in score_dicts])
    scores = {}
    for t in all_tracks:
        scores[t] = sum(
            weights[i] * score_dicts[i].get(t, 0)
            for i in range(len(score_dicts))
        )
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
