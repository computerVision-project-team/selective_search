"""
Inference script for Exercise 5.2.4.
Given an input image, generate proposals, classify with trained SVM,
and visualize detected balloons.
"""

import os
import argparse
import numpy as np
import skimage.io
import skimage.transform
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
from skimage.feature import hog
from PIL import Image

from selective_search import selective_search

try:
    import torch
    from torchvision import models, transforms
    from torchvision.models import ResNet18_Weights
except Exception:  # pragma: no cover
    torch = None
    models = None
    transforms = None
    ResNet18_Weights = None


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def extract_hog(image, rect, out_size):
    x, y, w, h = rect
    x, y, w, h = int(x), int(y), int(w), int(h)
    h_img, w_img = image.shape[:2]
    x2, y2 = min(x + w, w_img), min(y + h, h_img)
    x, y = max(0, x), max(0, y)
    if x2 <= x or y2 <= y:
        return None
    crop = image[y:y2, x:x2]
    if crop.size == 0:
        return None
    crop = skimage.transform.resize(crop, (out_size, out_size), anti_aliasing=True)
    feat = hog(
        crop,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        channel_axis=-1,
        feature_vector=True,
    )
    return feat


def extract_cnn(image, rect, preprocess, model):
    x, y, w, h = rect
    x, y, w, h = int(x), int(y), int(w), int(h)
    h_img, w_img = image.shape[:2]
    x2, y2 = min(x + w, w_img), min(y + h, h_img)
    x, y = max(0, x), max(0, y)
    if x2 <= x or y2 <= y:
        return None
    crop = image[y:y2, x:x2]
    if crop.size == 0:
        return None
    if crop.dtype != np.uint8:
        crop = (np.clip(crop, 0, 1) * 255).astype(np.uint8)
    crop_pil = Image.fromarray(crop)
    with torch.no_grad():
        inp = preprocess(crop_pil).unsqueeze(0)
        feat = model(inp).squeeze(0).cpu().numpy()
    norm = np.linalg.norm(feat)
    if norm > 1e-12:
        feat = feat / norm
    return feat


def box_iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union


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
    parser.add_argument("--scale", type=float, default=500)
    parser.add_argument("--sigma", type=float, default=0.8)
    parser.add_argument("--min_size", type=int, default=20)
    parser.add_argument("--max_merges", type=int, default=None)
    parser.add_argument("--out_size", type=int, default=128)
    parser.add_argument("--score_thresh", type=float, default=0.5)
    parser.add_argument("--nms_thresh", type=float, default=0.4)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--feature", choices=["hog", "cnn"], default="hog")
    args = parser.parse_args()

    image = skimage.io.imread(args.image)
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    if image.shape[2] > 3:
        image = image[:, :, :3]

    clf = joblib.load(args.model)

    preprocess = None
    feature_model = None
    if args.feature == "cnn":
        if torch is None:
            raise RuntimeError("PyTorch/torchvision not installed. Install torch and torchvision to use CNN features.")
        weights = ResNet18_Weights.DEFAULT
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        feature_model = models.resnet18(weights=weights)
        feature_model.fc = torch.nn.Identity()
        feature_model.eval()

    _, regions = selective_search(
        image, scale=args.scale, sigma=args.sigma, min_size=args.min_size, max_merges=args.max_merges
    )
    rects = [r["rect"] for r in regions]

    feats = []
    rects_kept = []
    for r in rects:
        if args.feature == "hog":
            feat = extract_hog(image, r, args.out_size)
        else:
            feat = extract_cnn(image, r, preprocess, feature_model)
        if feat is None:
            continue
        feats.append(feat)
        rects_kept.append(r)

    if not feats:
        print("No valid proposals.")
        return

    X = np.array(feats, dtype=np.float32)
    scores = clf.decision_function(X)
    rects_np = np.array(rects_kept, dtype=np.float32)

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 6))
    ax.imshow(image)
    count = 0
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
