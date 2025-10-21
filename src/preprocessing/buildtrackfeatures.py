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
        "positions": [],
        "playlist_count": 0
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
                    meta["positions"].append(track.get("track_position", None))
                    meta["playlist_count"] += 1

    rows = []
    for uri, meta in list(track_meta.items()):
        def safe_number(x):
            try:
                return float(x)
            except (TypeError, ValueError):
                return None
        
        valid_durations = [safe_number(d) for d in meta.get('duration_ms', [])]
        valid_durations = [d for d in valid_durations if d is not None]
        
        valid_positions = [
            int(p) for p in meta.get('positions', [])
            if isinstance(p, (int, float, str)) and str(p).isdigit()
        ]

        rows.append({
            "track_uri": uri,
            "track_name": meta.get('track_name'),
            "artist_name": meta.get('artist_name'),
            "album_name": meta.get('album_name'),
            "avg_duration_ms": sum(valid_durations) / len(valid_durations) if valid_durations else None,
            "avg_position": sum(valid_positions) / len(valid_positions) if valid_positions else None,
            "playlist_count": meta.get('playlist_count', 0)
        })

    df = pd.DataFrame(rows).set_index("track_uri")
    return df
