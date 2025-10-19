import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity

def recommend_content_based(seed_tracks, track_features, top_k=100):
    seed_vecs = track_features.loc[track_features.index.intersection(seed_tracks)]
    if seed_vecs.empty:
        return []

    mean_vec = seed_vecs.mean().values.reshape(1, -1)
    similarities = cosine_similarity(track_features.values, mean_vec).flatten()
    scores = pd.Series(similarities, index=track_features.index)
    return scores.sort_values(ascending=False).head(top_k).index.tolist()
