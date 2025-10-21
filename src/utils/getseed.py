import random

def get_seed(playlist, category):
    tracks = [t['track_uri'] for t in playlist['tracks']]
    title = playlist.get('name', '').strip()

    if category == 1:
        return [], title
    elif category == 2:
        return tracks[:1], title
    elif category == 3:
        return tracks[:5], title
    elif category == 4:
        return tracks[:5], ''
    elif category == 5:
        return tracks[:10], title
    elif category == 6:
        return tracks[:10], ''
    elif category == 7:
        return tracks[:25], title
    elif category == 8:
        return random.sample(tracks, min(25, len(tracks))), title
    elif category == 9:
        return tracks[:100], title
    elif category == 10:
        return random.sample(tracks, min(100, len(tracks))), title
    else:
        raise ValueError(f"Unknown category: {category}")