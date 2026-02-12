"""
Inference script for Exercise 5.2.4.
Given an input image, generate proposals, classify with trained SVM,
and visualize detected balloons.
"""

import os
import argparse
import numpy as np
import skimage.io
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib

from selective_search import selective_search
from feature_utils import FeatureExtractor, box_iou


def nms(rects, scores, iou_thresh):
    order = np.argsort(scores)[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        filtered = []
        for j in rest:
            if box_iou(rects[i], rects[j]) <= iou_thresh:
                filtered.append(j)
        order = np.array(filtered, dtype=np.int64)
    return keep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--model", default="../results/balloon_svm.joblib")
    parser.add_argument("--out", default="../results/inference.png")
    parser.add_argument("--scale", type=float, default=300)
    parser.add_argument("--sigma", type=float, default=0.8)
    parser.add_argument("--min_size", type=int, default=50)
    parser.add_argument("--max_merges", type=int, default=None)
    parser.add_argument("--out_size", type=int, default=128)
    parser.add_argument("--score_thresh", type=float, default=0.0)
    parser.add_argument("--nms_thresh", type=float, default=0.5)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--feature", choices=["auto", "hog", "cnn"], default="auto")
    args = parser.parse_args()

    image = skimage.io.imread(args.image)
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    if image.shape[2] > 3:
        image = image[:, :, :3]

    clf = joblib.load(args.model)
    model_dim = int(getattr(clf, "n_features_in_", -1))
    feature_type = args.feature
    if feature_type == "auto":
        if model_dim == 512:
            feature_type = "cnn"
        else:
            feature_type = "hog"

    extractor = FeatureExtractor(feature=feature_type, out_size=args.out_size)
    print("Using feature extractor:", feature_type)

    _, regions = selective_search(
        image, scale=args.scale, sigma=args.sigma, min_size=args.min_size, max_merges=args.max_merges
    )
    rects = [r["rect"] for r in regions]

    feats = []
    rects_kept = []
    for r in rects:
        feat = extractor.extract(image, r, augment=False)
        if feat is None:
            continue
        feats.append(feat)
        rects_kept.append(r)

    if not feats:
        print("No valid proposals.")
        return

    X = np.array(feats, dtype=np.float32)
    if model_dim > 0 and X.shape[1] != model_dim:
        raise RuntimeError(
            f"Feature dimension mismatch: extracted {X.shape[1]} but model expects {model_dim}. "
            f"Use --feature {'cnn' if model_dim == 512 else 'hog'} or retrain the model."
        )
    scores = clf.decision_function(X)
    rects_np = np.array(rects_kept, dtype=np.float32)
    print("Score stats (min/mean/max):", float(scores.min()), float(scores.mean()), float(scores.max()))

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 6))
    ax.imshow(image)
    count = 0
    # filter by score and apply NMS
    keep_mask = scores >= args.score_thresh
    rects_f = rects_np[keep_mask]
    scores_f = scores[keep_mask]
    if rects_f.size == 0:
        print("No detections above threshold.")
        return
    keep_idx = nms(rects_f, scores_f, args.nms_thresh)
    rects_f = rects_f[keep_idx]
    scores_f = scores_f[keep_idx]
    if rects_f.shape[0] > args.top_k:
        order = np.argsort(scores_f)[::-1][: args.top_k]
        rects_f = rects_f[order]
        scores_f = scores_f[order]

    for rect, score in zip(rects_f, scores_f):
        x, y, w, h = rect
        rect_patch = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor="red", linewidth=1
        )
        ax.add_patch(rect_patch)
        count += 1
    plt.axis("off")

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print("Detections:", count, "Saved to:", out_path)


if __name__ == "__main__":
    main()
