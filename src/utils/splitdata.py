import glob
import json
import os
import pandas as pd
import random

def load_data(json_dir):
    with open(json_dir, "r") as f:
        data = json.load(f)
    playlists = data.get("playlists", [])
    playlists = [p for p in playlists if any("track_uri" in t for t in p.get("tracks", []))]
    return playlists

def load_all_playlists(json_dir):
    playlists = []
    for fname in os.listdir(json_dir):
        if fname.endswith(".json"):
            with open(os.path.join(json_dir, fname)) as f:
                data = json.load(f)
                playlists.extend(data["playlists"])
    return playlists

def load_slice(path):
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

def load_all(folder):
    file = glob.glob(f"{folder}/challenge_set.json")
    all_records = []
    for f in file:
        all_records.append(load_slice(f))
    return pd.concat(all_records, ignore_index=True)

def split_playlists(playlists, challenge_size=10000, seed=42):#(path, challenge_size=10000, seed=42):
    #with open(path, "r") as f:
    #    data = json.load(f)
    #playlists = data.get("playlists", [])

    random.seed(seed)
    indices = list(range(len(playlists)))
    random.shuffle(indices)

    challenge_indices = set(indices[:challenge_size])
    train_playlists = [p for i, p in enumerate(playlists) if i not in challenge_indices]
    challenge_playlists = [p for i, p in enumerate(playlists) if i in challenge_indices]

    #train = [p for p in train_playlists if any("track_uri" in t and t["track_uri"] for t in p.get("tracks", []))]
    #challenge = [p for p in challenge_playlists if any("track_uri" in t and t["track_uri"] for t in p.get("tracks", []))]

    return train_playlists, challenge_playlists

