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
from PIL import Image

try:
    import torch
    from torchvision import models, transforms
    from torchvision.models import ResNet18_Weights
except Exception:  # pragma: no cover
    torch = None
    models = None
    transforms = None
    ResNet18_Weights = None

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except Exception:  # pragma: no cover
    COCO = None
    COCOeval = None

from selective_search import selective_search


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


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


def precision_recall_ap(tp, fp, num_gt):
    if num_gt == 0:
        return 0.0
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    recall = tp / (num_gt + 1e-12)
    precision = tp / (tp + fp + 1e-12)

    precision = np.maximum.accumulate(precision[::-1])[::-1]
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
    parser.add_argument("--score_thresh", type=float, default=-1.0)
    parser.add_argument("--nms_thresh", type=float, default=0.4)
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

    preprocess = None
    feature_model = None
    if feature_type == "cnn":
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

    detections = []
    total_gts = 0
    mabo_list = []

    if len(ann_data.get("categories", [])) == 0:
        raise RuntimeError("No categories found in COCO annotations.")
    ann_cat_ids = [a.get("category_id") for a in ann_data.get("annotations", []) if "category_id" in a]
    if ann_cat_ids:
        cat_id = int(max(set(ann_cat_ids), key=ann_cat_ids.count))
    else:
        cat_id = int(ann_data["categories"][0]["id"])

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
            if feature_type == "hog":
                feat = extract_hog(image, r, args.out_size)
            else:
                feat = extract_cnn(image, r, preprocess, feature_model)
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
        if not np.any(keep_mask):
            continue
        rects_f = rects_np[keep_mask]
        scores_f = scores[keep_mask]
        keep_idx = nms(rects_f, scores_f, args.nms_thresh)
        rects_f = rects_f[keep_idx]
        scores_f = scores_f[keep_idx]
        if args.top_k > 0 and rects_f.shape[0] > args.top_k:
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

    if COCO is None or COCOeval is None:
        raise RuntimeError("pycocotools not installed. Please install it to compute official COCO mAP.")

    coco_gt = COCO(ann_path)
    coco_dt = coco_gt.loadRes(detections) if detections else coco_gt.loadRes([])
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.useCats = 1
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    mAP = float(coco_eval.stats[0])
    ap50 = float(coco_eval.stats[1])
    mabo = float(np.mean(mabo_list)) if mabo_list else 0.0

    print("mAP (0.50:0.95):", round(mAP, 4))
    print("AP@0.50:", round(ap50, 4))
    print("MABO:", round(mabo, 4))


if __name__ == "__main__":
    main()
