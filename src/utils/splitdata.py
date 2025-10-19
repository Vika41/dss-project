import json
import os
import random

def load_all_playlists(json_dir):
    playlists = []
    for fname in os.listdir(json_dir):
        if fname.endswith(".json"):
            with open(os.path.join(json_dir, fname)) as f:
                data = json.load(f)
                playlists.extend(data["playlists"])
    return playlists

def split_playlists(playlists, challenge_size=10000, seed=42):
    random.seed(seed)
    indices = list(range(len(playlists)))
    random.shuffle(indices)

    challenge_indices = set(indices[:challenge_size])
    train_playlists = [p for i, p in enumerate(playlists) if i not in challenge_indices]
    challenge_playlists = [p for i, p in enumerate(playlists) if i in challenge_indices]

    return train_playlists, challenge_playlists
