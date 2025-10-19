import json
import os

from collections import defaultdict, Counter

def load_playlists(json_dir):
    playlists = []
    for fname in os.listdir(json_dir):
        if fname.endswith('.json'):
            with open(os.path.join(json_dir, fname)) as f:
                data = json.load(f)
                playlists.extend(data['playlists'])
    return playlists

def build_co_matrix(playlists):
    co_matrix = defaultdict(Counter)
    for playlist in playlists:
        tracks = [f"{t['track_uri']}" for t in playlist['tracks']]
        for i, t1 in enumerate(tracks):
            for t2 in tracks[i+1:]:
                co_matrix[t1][t2] += 1
                co_matrix[t2][t1] += 1
    return co_matrix

def get_popularity(playlists):
    pop = Counter()
    for playlist in playlists:
        for track in playlist['tracks']:
            pop[track['track_uri']] += 1
    return pop

def predict(seed_tracks, co_matrix, popularity, top_k=100):
    scores = Counter()
    for track in seed_tracks:
        scores.update(co_matrix.get(track, {}))
    for track in popularity:
        scores[track] += 0.1 * popularity[track]
    return [t for t, _ in scores.most_common(top_k)]

def predict_next_tracks(seed_tracks, co_matrix, popularity, top_k=100):
    scores = defaultdict(float)
    for track in seed_tracks:
        for co_track, weight in co_matrix[track].items():
            scores[co_track] += weight
    for track in popularity:
        scores[track] += 0.1 * popularity[track]  # small boost
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [track for track, _ in ranked[:top_k]]
