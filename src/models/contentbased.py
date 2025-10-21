import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

imputer = SimpleImputer(strategy="mean")
scaler = StandardScaler()

def recommend_content_based(seed_tracks, track_features, top_k=100):
    flat_seed_tracks = [track for sublist in seed_tracks for track in sublist]
    seed_vecs = track_features.loc[track_features.index.intersection(flat_seed_tracks)]
    if seed_vecs.empty:
        return []

    mean_vec = seed_vecs.select_dtypes(include='number').mean().values.reshape(1, -1)
    numeric_features = track_features.select_dtypes(include='number')
    imputed_features = imputer.fit_transform(numeric_features)
    scaled_features = scaler.fit_transform(imputed_features)
    mean_vec_imputed = imputer.transform(mean_vec)
    scaled_mean = scaler.transform(mean_vec_imputed)
    similarities = cosine_similarity(scaled_features, scaled_mean).flatten()
    scores = pd.Series(similarities, index=track_features.index)
    return scores.sort_values(ascending=False).head(top_k).index.tolist()
