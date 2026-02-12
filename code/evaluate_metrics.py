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
import joblib

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except Exception:  # pragma: no cover
    COCO = None
    COCOeval = None

from selective_search import selective_search
from feature_utils import FeatureExtractor, box_iou


def load_coco_boxes(ann_path):
    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    img_id_to_file = {img["id"]: img["file_name"] for img in data["images"]}
    file_to_boxes = {fn: [] for fn in img_id_to_file.values()}
    for ann in data["annotations"]:
        img_id = ann["image_id"]
        fn = img_id_to_file[img_id]
        file_to_boxes[fn].append(ann["bbox"])
    return file_to_boxes, img_id_to_file, data


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
    parser.add_argument("--data_root", default="../data/balloon_dataset")
    parser.add_argument("--proposals_root", default="../data/balloon_dataset/proposals",
                        help="Use cached proposals first (fallback: on-the-fly selective search).")
    parser.add_argument("--model", default="../results/balloon_svm.joblib")
    parser.add_argument("--scale", type=float, default=300)
    parser.add_argument("--sigma", type=float, default=0.8)
    parser.add_argument("--min_size", type=int, default=50)
    parser.add_argument("--max_merges", type=int, default=None)
    parser.add_argument("--out_size", type=int, default=128)
    parser.add_argument("--score_thresh", type=float, default=0.0)
    parser.add_argument("--nms_thresh", type=float, default=0.5)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--feature", choices=["auto", "hog", "cnn"], default="auto")
    args = parser.parse_args()

    data_root = os.path.abspath(args.data_root)
    test_dir = os.path.join(data_root, "test")
    ann_path = os.path.join(test_dir, "_annotations.coco.json")
    file_to_boxes, img_id_to_file, ann_data = load_coco_boxes(ann_path)

    clf = joblib.load(args.model)
    model_dim = int(getattr(clf, "n_features_in_", -1))
    feature_type = args.feature
    if feature_type == "auto":
        if model_dim == 512:
            feature_type = "cnn"
        else:
            feature_type = "hog"

    extractor = FeatureExtractor(feature=feature_type, out_size=args.out_size)

    detections = []  # for COCO eval: list of dicts
    total_gts = 0
    mabo_list = []
    cache_hits = 0
    online_generated = 0

    # choose category id from annotations (robust to duplicated/unused category entries)
    if len(ann_data.get("categories", [])) == 0:
        raise RuntimeError("No categories found in COCO annotations.")
    ann_cat_ids = [a.get("category_id") for a in ann_data.get("annotations", []) if "category_id" in a]
    if ann_cat_ids:
        cat_id = int(max(set(ann_cat_ids), key=ann_cat_ids.count))
    else:
        cat_id = int(ann_data["categories"][0]["id"])

    # build filename -> image_id map
    file_to_img_id = {v: k for k, v in img_id_to_file.items()}

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
                cache_hits += 1
        if rects is None:
            _, regions = selective_search(
                image, scale=args.scale, sigma=args.sigma,
                min_size=args.min_size, max_merges=args.max_merges
            )
            rects = np.array([r["rect"] for r in regions], dtype=np.int32)
            online_generated += 1

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
            feat = extractor.extract(image, r, augment=False)
            if feat is None:
                continue
            feats.append(feat)
            rects_kept.append(r)

        if not feats:
            continue
        X = np.array(feats, dtype=np.float32)
        if model_dim > 0 and X.shape[1] != model_dim:
            raise RuntimeError(
                f"Feature dimension mismatch: extracted {X.shape[1]} but model expects {model_dim}. "
                f"Use --feature {'cnn' if model_dim == 512 else 'hog'} or retrain the model."
            )
        scores = clf.decision_function(X)
        rects_np = np.array(rects_kept, dtype=np.float32)

        keep_mask = scores >= args.score_thresh
        rects_f = rects_np[keep_mask]
        scores_f = scores[keep_mask]
        if rects_f.size == 0:
            continue
        keep_idx = nms(rects_f, scores_f, args.nms_thresh)
        rects_f = rects_f[keep_idx]
        scores_f = scores_f[keep_idx]
        if rects_f.shape[0] > args.top_k:
            order = np.argsort(scores_f)[::-1][: args.top_k]
            rects_f = rects_f[order]
            scores_f = scores_f[order]

        img_id = file_to_img_id.get(fn)
        for r, s in zip(rects_f, scores_f):
            detections.append({
                "image_id": int(img_id),
                "category_id": int(cat_id),
                "bbox": [float(r[0]), float(r[1]), float(r[2]), float(r[3])],
                "score": float(s),
            })

    # COCO official mAP (requires pycocotools)
    if COCO is None or COCOeval is None:
        raise RuntimeError("pycocotools not installed. Please install it to compute official COCO mAP.")

    coco_gt = COCO(ann_path)
    coco_dt = coco_gt.loadRes(detections) if detections else coco_gt.loadRes([])
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.useCats = 1
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    mAP = float(coco_eval.stats[0])  # AP IoU=0.50:0.95
    ap50 = float(coco_eval.stats[1])  # AP IoU=0.50
    mabo = float(np.mean(mabo_list)) if mabo_list else 0.0

    print("mAP (0.50:0.95):", round(mAP, 4))
    print("AP@0.50:", round(ap50, 4))
    print("MABO:", round(mabo, 4))
    print("Proposal source (cached / generated):", cache_hits, "/", online_generated)


if __name__ == "__main__":
    main()
