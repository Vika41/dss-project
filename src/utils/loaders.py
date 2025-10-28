import glob
import json
import pandas as pd
import os

from src.models.collaborative import build_interaction_df, df_to_sparse_matrix, train_als_model, compute_track_similarity

def load_playlists_cf(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    playlists = data.get("playlists", [])
    return [p for p in playlists if any("track_uri" in t for t in p.get("tracks", []))]

def load_slice_cf(path):
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

def load_all_cf(folder):
    file = glob.glob(f"{folder}/challenge_set.json")
    all_records = []
    for f in file:
        all_records.append(load_slice_cf(f))
    return pd.concat(all_records, ignore_index=True)

def load_playlists(json_dir):
    playlists = []
    for fname in os.listdir(json_dir):
        if fname.endswith('.json'):
            with open(os.path.join(json_dir, fname)) as f:
                data = json.load(f)
                playlists.extend(data['playlists'])
    return playlists

def load_track_meta(playlists):
    meta = {}
    for p in playlists:
        for t in p.get("tracks", []):
            uri = t.get("track_uri")
            if uri and uri not in meta:
                meta[uri] = {
                    "track_name": t.get("track_name"),
                    "artist_name": t.get("artist_name"),
                    "album_name": t.get("album_name"),
                    "duration_ms": t.get("duration_ms")
                }
    return meta

def build_models(playlists):
    df = build_interaction_df(playlists)
    matrix, playlist_index, track_index = df_to_sparse_matrix(df)
    als_model = train_als_model(matrix)
    item_sim_matrix = compute_track_similarity(matrix)
    popularity_dict = df["track_uri"].value_counts().to_dict()
    return item_sim_matrix, als_model, popularity_dict, track_index, playlist_index
