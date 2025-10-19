import math

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
