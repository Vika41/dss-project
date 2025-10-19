import numpy as np

def r_precision(predicted, ground_truth):
    gt_set = set(ground_truth)
    pred_set = set(predicted[:len(ground_truth)])
    return len(gt_set & pred_set) / len(gt_set)

def ndcg(predicted, ground_truth):
    gt_set = set(ground_truth)
    dcg = sum([1 / np.log2(i + 2) if track in gt_set else 0
               for i, track in enumerate(predicted)])
    idcg = sum([1 / np.log2(i + 2) for i in range(min(len(gt_set), len(predicted)))])
    return dcg / idcg if idcg > 0 else 0.0
