"""
Simple detection pipeline for Exercise 5.2 (balloon dataset).
Steps:
1) Load selective search proposals (npz with rects)
2) Build positive/negative samples by IoU thresholds
3) Extract HOG/CNN features
4) Train linear SVM and evaluate on valid split
"""

import os
import json
import argparse
import numpy as np
import skimage.io
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import joblib
from feature_utils import FeatureExtractor, box_iou


def load_coco_boxes(ann_path):
    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    img_id_to_file = {img["id"]: img["file_name"] for img in data["images"]}
    file_to_boxes = {fn: [] for fn in img_id_to_file.values()}
    for ann in data["annotations"]:
        img_id = ann["image_id"]
        fn = img_id_to_file[img_id]
        # COCO bbox: [x, y, w, h]
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


def build_samples(split_dir, proposals_dir, ann_path, tp, tn, out_size,
                  max_pos_per_img, max_neg_per_img, feature_type,
                  extractor=None, augment=False, aug_pos=1, neg_dedup_iou=0.9,
                  neg_pos_ratio=3.0, min_neg_per_img=10):
    file_to_boxes = load_coco_boxes(ann_path)
    X, y = [], []

    for fn, gts in file_to_boxes.items():
        img_path = os.path.join(split_dir, fn)
        prop_path = os.path.join(proposals_dir, fn + ".npz")
        if not os.path.isfile(img_path) or not os.path.isfile(prop_path):
            continue

        image = skimage.io.imread(img_path)
        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)
        if image.shape[2] > 3:
            image = image[:, :, :3]

        rects = np.load(prop_path)["rects"].tolist()
        if not rects:
            continue

        # Score each proposal by max IoU first, then pick best/hard samples instead of raw generation order.
        scored = []
        for r in rects:
            max_iou = 0.0
            for gt in gts:
                max_iou = max(max_iou, iou(r, gt))
            scored.append((r, max_iou))

        pos_candidates = [it for it in scored if it[1] >= tp]
        pos_candidates.sort(key=lambda t: t[1], reverse=True)
        pos_selected = pos_candidates[:max_pos_per_img]

        neg_limit = max_neg_per_img
        if neg_pos_ratio is not None and neg_pos_ratio > 0:
            if len(pos_selected) > 0:
                neg_limit = min(
                    max_neg_per_img,
                    max(min_neg_per_img, int(np.ceil(len(pos_selected) * neg_pos_ratio))),
                )
            else:
                neg_limit = min(max_neg_per_img, min_neg_per_img)

        neg_candidates = [it for it in scored if it[1] <= tn]
        # Hard negatives first: IoU closer to tn are more confusing.
        neg_candidates.sort(key=lambda t: t[1], reverse=True)
        neg_selected = []
        for r, max_iou in neg_candidates:
            duplicated = False
            for r_keep, _ in neg_selected:
                if box_iou(r, r_keep) > neg_dedup_iou:
                    duplicated = True
                    break
            if duplicated:
                continue
            neg_selected.append((r, max_iou))
            if len(neg_selected) >= neg_limit:
                break

        for r, _ in pos_selected:
            feat = extractor.extract(image, r, augment=False)
            if feat is None:
                continue
            X.append(feat)
            y.append(1)
            if augment:
                for _ in range(aug_pos):
                    feat_aug = extractor.extract(image, r, augment=True)
                    if feat_aug is not None:
                        X.append(feat_aug)
                        y.append(1)

        for r, _ in neg_selected:
            feat = extractor.extract(image, r, augment=False)
            if feat is None:
                continue
            X.append(feat)
            y.append(0)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="../data/balloon_dataset")
    parser.add_argument("--proposals_root", default="../data/balloon_dataset/proposals")
    parser.add_argument("--tp", type=float, default=0.5)
    parser.add_argument("--tn", type=float, default=0.3)
    parser.add_argument("--out_size", type=int, default=128)
    parser.add_argument("--max_pos_per_img", type=int, default=50)
    parser.add_argument("--max_neg_per_img", type=int, default=20)
    parser.add_argument("--model_out", default="../results/balloon_svm.joblib")
    parser.add_argument("--feature", choices=["hog", "cnn"], default="hog")
    parser.add_argument("--hard_neg", action="store_true", help="Enable hard negative mining")
    parser.add_argument("--hn_per_img", type=int, default=20)
    parser.add_argument("--augment", action="store_true", help="Enable training-time augmentation on positive samples")
    parser.add_argument("--aug_pos", type=int, default=1, help="Number of augmented copies per positive sample")
    parser.add_argument("--neg_dedup_iou", type=float, default=0.9, help="IoU threshold to deduplicate near-identical negatives")
    parser.add_argument("--neg_pos_ratio", type=float, default=3.0, help="Max negative:positive ratio per image")
    parser.add_argument("--min_neg_per_img", type=int, default=10, help="Minimum negatives per image when proposals are available")
    args = parser.parse_args()

    data_root = os.path.abspath(args.data_root)
    proposals_root = os.path.abspath(args.proposals_root)

    train_dir = os.path.join(data_root, "train")
    valid_dir = os.path.join(data_root, "valid")
    train_props = os.path.join(proposals_root, "train")
    valid_props = os.path.join(proposals_root, "valid")
    train_ann = os.path.join(train_dir, "_annotations.coco.json")
    valid_ann = os.path.join(valid_dir, "_annotations.coco.json")

    extractor = FeatureExtractor(feature=args.feature, out_size=args.out_size)

    X_train, y_train = build_samples(
        train_dir, train_props, train_ann, args.tp, args.tn, args.out_size,
        args.max_pos_per_img, args.max_neg_per_img, args.feature,
        extractor=extractor, augment=args.augment, aug_pos=args.aug_pos,
        neg_dedup_iou=args.neg_dedup_iou, neg_pos_ratio=args.neg_pos_ratio,
        min_neg_per_img=args.min_neg_per_img,
    )
    X_valid, y_valid = build_samples(
        valid_dir, valid_props, valid_ann, args.tp, args.tn, args.out_size,
        args.max_pos_per_img, args.max_neg_per_img, args.feature,
        extractor=extractor, augment=False, aug_pos=0,
        neg_dedup_iou=args.neg_dedup_iou, neg_pos_ratio=args.neg_pos_ratio,
        min_neg_per_img=args.min_neg_per_img,
    )

    print("Train samples:", X_train.shape, "Pos:", int(y_train.sum()), "Neg:", int((y_train == 0).sum()))
    print("Valid samples:", X_valid.shape, "Pos:", int(y_valid.sum()), "Neg:", int((y_valid == 0).sum()))
    if int(y_train.sum()) < 100:
        print("WARNING: Too few positive train samples. Try lowering --tp or regenerating proposals.")

    clf = LinearSVC(class_weight="balanced", max_iter=5000)
    clf.fit(X_train, y_train)

    # Hard negative mining on train set (optional)
    if args.hard_neg:
        extra_X = []
        extra_y = []
        file_to_boxes = load_coco_boxes(train_ann)
        for fn, gts in file_to_boxes.items():
            img_path = os.path.join(train_dir, fn)
            prop_path = os.path.join(train_props, fn + ".npz")
            if not os.path.isfile(img_path) or not os.path.isfile(prop_path):
                continue
            image = skimage.io.imread(img_path)
            if image.ndim == 2:
                image = np.stack([image, image, image], axis=-1)
            if image.shape[2] > 3:
                image = image[:, :, :3]
            rects = np.load(prop_path)["rects"]
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
            scores = clf.decision_function(X)
            # pick top false positives
            order = np.argsort(scores)[::-1]
            count = 0
            for idx in order:
                r = rects_kept[idx]
                max_iou = 0.0
                for gt in gts:
                    max_iou = max(max_iou, iou(r, gt))
                if max_iou <= args.tn:
                    extra_X.append(X[idx])
                    extra_y.append(0)
                    count += 1
                    if count >= args.hn_per_img:
                        break
        if extra_X:
            X_train2 = np.vstack([X_train, np.array(extra_X, dtype=np.float32)])
            y_train2 = np.concatenate([y_train, np.array(extra_y, dtype=np.int32)])
            clf = LinearSVC(class_weight="balanced", max_iter=5000)
            clf.fit(X_train2, y_train2)
    y_pred = clf.predict(X_valid)
    print(classification_report(y_valid, y_pred, digits=3))

    os.makedirs(os.path.dirname(os.path.abspath(args.model_out)), exist_ok=True)
    joblib.dump(clf, args.model_out)
    print("Saved model to:", os.path.abspath(args.model_out))


if __name__ == "__main__":
    main()
