import os
import json
import pandas as pd
from collections import defaultdict

def build_track_features(json_dir):
    track_meta = defaultdict(lambda: {
        "track_name": None,
        "artist_name": None,
        "album_name": None,
        "duration_ms": [],
        "playlist_count": 0,
        "positions": []
    })

    for fname in os.listdir(json_dir):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(json_dir, fname)) as f:
            data = json.load(f)
            for playlist in data["playlists"]:
                for track in playlist["tracks"]:
                    uri = track["track_uri"]
                    meta = track_meta[uri]
                    meta["track_name"] = track["track_name"]
                    meta["artist_name"] = track["artist_name"]
                    meta["album_name"] = track["album_name"]
                    meta["duration_ms"].append(track["duration_ms"])
                    meta["positions"].append(track["track_position"])
                    meta["playlist_count"] += 1

    rows = []
    for uri, meta in track_meta.items():
        rows.append({
            "track_uri": uri,
            "track_name": meta["track_name"],
            "artist_name": meta["artist_name"],
            "album_name": meta["album_name"],
            "avg_duration_ms": sum(meta["duration_ms"]) / len(meta["duration_ms"]),
            "avg_position": sum(meta["positions"]) / len(meta["positions"]),
            "playlist_count": meta["playlist_count"]
        })

    df = pd.DataFrame(rows).set_index("track_uri")
    return df
