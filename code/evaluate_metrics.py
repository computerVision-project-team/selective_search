"""
Evaluation for Exercise 5.2.5.
Computes COCO-style mAP (IoU 0.50:0.95) and MABO on the test split.
Single-class (balloon) evaluation.
"""

import os
import json
import argparse
import numpy as np
import skimage.io
import skimage.transform
from skimage.feature import hog
import joblib

from selective_search import selective_search


def load_coco_boxes(ann_path):
    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    img_id_to_file = {img["id"]: img["file_name"] for img in data["images"]}
    file_to_boxes = {fn: [] for fn in img_id_to_file.values()}
    for ann in data["annotations"]:
        img_id = ann["image_id"]
        fn = img_id_to_file[img_id]
        file_to_boxes[fn].append(ann["bbox"])
    return file_to_boxes


def iou(box, gt):
    x, y, w, h = box
    gx, gy, gw, gh = gt
    x2, y2 = x + w, y + h
    gx2, gy2 = gx + gw, gy + gh
    ix1, iy1 = max(x, gx), max(y, gy)
    ix2, iy2 = min(x2, gx2), min(y2, gy2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    union = w * h + gw * gh - inter
    return inter / union


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


def precision_recall_ap(tp, fp, num_gt):
    if num_gt == 0:
        return 0.0
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    recall = tp / (num_gt + 1e-12)
    precision = tp / (tp + fp + 1e-12)

    # Precision envelope
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    # Integrate over recall
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return ap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="../data/balloon_dataset")
    parser.add_argument("--proposals_root", default=None,
                        help="Optional proposals root (use if already computed).")
    parser.add_argument("--model", default="../results/balloon_svm.joblib")
    parser.add_argument("--scale", type=float, default=500)
    parser.add_argument("--sigma", type=float, default=0.8)
    parser.add_argument("--min_size", type=int, default=20)
    parser.add_argument("--max_merges", type=int, default=None)
    parser.add_argument("--out_size", type=int, default=128)
    parser.add_argument("--score_thresh", type=float, default=-1e9)
    args = parser.parse_args()

    data_root = os.path.abspath(args.data_root)
    test_dir = os.path.join(data_root, "test")
    ann_path = os.path.join(test_dir, "_annotations.coco.json")
    file_to_boxes = load_coco_boxes(ann_path)

    clf = joblib.load(args.model)

    detections = []
    total_gts = 0
    mabo_list = []

    for fn, gts in file_to_boxes.items():
        img_path = os.path.join(test_dir, fn)
        if not os.path.isfile(img_path):
            continue
        image = skimage.io.imread(img_path)
        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)
        if image.shape[2] > 3:
            image = image[:, :, :3]

        # proposals: from saved if available, else compute
        rects = None
        if args.proposals_root:
            prop_path = os.path.join(os.path.abspath(args.proposals_root), "test", fn + ".npz")
            if os.path.isfile(prop_path):
                rects = np.load(prop_path)["rects"]
        if rects is None:
            _, regions = selective_search(
                image, scale=args.scale, sigma=args.sigma,
                min_size=args.min_size, max_merges=args.max_merges
            )
            rects = np.array([r["rect"] for r in regions], dtype=np.int32)

        # MABO (best proposal overlap per GT)
        if len(gts) > 0 and len(rects) > 0:
            for gt in gts:
                best = 0.0
                for r in rects:
                    best = max(best, iou(r, gt))
                mabo_list.append(best)
        total_gts += len(gts)

        feats = []
        rects_kept = []
        for r in rects:
            feat = extract_hog(image, r, args.out_size)
            if feat is None:
                continue
            feats.append(feat)
            rects_kept.append(r)

        if not feats:
            continue
        X = np.array(feats, dtype=np.float32)
        scores = clf.decision_function(X)

        for r, s in zip(rects_kept, scores):
            if s < args.score_thresh:
                continue
            detections.append((fn, float(s), r))

    # COCO-style mAP (IoU 0.50:0.95)
    iou_thresholds = np.arange(0.5, 0.96, 0.05)
    aps = []
    for thr in iou_thresholds:
        detections_sorted = sorted(detections, key=lambda x: x[1], reverse=True)
        gt_used = {fn: np.zeros(len(gts), dtype=bool) for fn, gts in file_to_boxes.items()}
        tp = []
        fp = []
        for fn, score, r in detections_sorted:
            gts = file_to_boxes.get(fn, [])
            best_iou = 0.0
            best_idx = -1
            for i, gt in enumerate(gts):
                iou_val = iou(r, gt)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_idx = i
            if best_iou >= thr and best_idx >= 0 and not gt_used[fn][best_idx]:
                tp.append(1)
                fp.append(0)
                gt_used[fn][best_idx] = True
            else:
                tp.append(0)
                fp.append(1)

        ap = precision_recall_ap(np.array(tp), np.array(fp), total_gts)
        aps.append(ap)

    mAP = float(np.mean(aps)) if aps else 0.0
    ap50 = float(aps[0]) if aps else 0.0
    mabo = float(np.mean(mabo_list)) if mabo_list else 0.0

    print("mAP (0.50:0.95):", round(mAP, 4))
    print("AP@0.50:", round(ap50, 4))
    print("MABO:", round(mabo, 4))


if __name__ == "__main__":
    main()
