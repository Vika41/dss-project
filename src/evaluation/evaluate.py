import math
import numpy as np

def get_ground_truth_tracks(playlist):
    return [t["track_uri"] for t in playlist.get("tracks", [])[:5]]

def precision_at_k(preds, ground_truth, k):
    pred_k = preds[:k]
    hits = sum(1 for track in pred_k if track in ground_truth)
    return hits / k

def recall_at_k(preds, ground_truth, k):
    hits = sum(1 for track in preds[:k] if track in ground_truth)
    return hits / len(ground_truth) if ground_truth else 0

def ndcg_at_k(preds, ground_truth, k):
    dcg = 0.0
    for i, track in enumerate(preds[:k]):
        if track in ground_truth:
            dcg += 1 / np.log2(i + 2)
    ideal_dcg = sum(1 / np.log2(i + 2) for i in range(min(len(ground_truth), k)))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0

def r_precision(preds, ground_truth):
    return len(set(preds) & set(ground_truth)) / len(ground_truth)

def ndcg(preds, ground_truth):
    dcg = 0
    for i, p in enumerate(preds):
        if p in ground_truth:
            dcg += 1 / math.log2(i + 2)
    idcg = sum(1 / math.log2(i + 2) for i in range(len(ground_truth)))
    return dcg / idcg if idcg > 0 else 0

def clicks(preds, ground_truth):
    for i, p in enumerate(preds):
        if p in ground_truth:
            return i
    return 500

def evaluate(preds, ground_truth):
    return {
        "R-Precision": round(r_precision(preds, ground_truth), 3),
        "NDCG": round(ndcg(preds, ground_truth), 3),
        "Clicks": clicks(preds, ground_truth)
    }

def evaluate_k(preds, ground_truth, k):
    return {
        "Precision @ k": round(precision_at_k(preds, ground_truth, k), 3),
        "Recall @ k": round(recall_at_k(preds, ground_truth, k), 3),
        "NDCG @ k": round(ndcg_at_k(preds, ground_truth, k), 3),
        "R-Precision": round(r_precision(preds, ground_truth), 3),
        "NDCG": round(ndcg(preds, ground_truth), 3),
        "Clicks": clicks(preds, ground_truth)
    }

def evaluate_model(playlists, model, k=10):
    metrics = []
    for p in playlists:
        seed = [t["track_uri"] for t in p.get("tracks", [])[:5]]
        truth = [t["track_uri"] for t in p.get("tracks", [])[:5]]
        preds = model(seed)
        metrics.append((
            precision_at_k(preds, truth, k),
            recall_at_k(preds, truth, k),
            ndcg_at_k(preds, truth, k)
        ))
    return np.mean(metrics, axis=0)