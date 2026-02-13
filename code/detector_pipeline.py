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
import skimage.transform
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
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


def get_crop(image, rect):
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
    return crop


def maybe_augment_crop(crop, enable_aug):
    if not enable_aug:
        return crop
    out = crop.copy()
    if np.random.rand() < 0.5:
        out = np.fliplr(out)
    # brightness/contrast jitter
    alpha = np.random.uniform(0.85, 1.15)  # contrast
    beta = np.random.uniform(-20, 20)      # brightness
    out = np.clip(alpha * out + beta, 0, 255).astype(np.uint8)
    return out


def extract_hog(image, rect, out_size, enable_aug=False):
    crop = get_crop(image, rect)
    if crop is None:
        return None
    if crop.dtype != np.uint8:
        crop = (np.clip(crop, 0, 1) * 255).astype(np.uint8)
    crop = maybe_augment_crop(crop, enable_aug)
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


def extract_cnn(image, rect, preprocess, model, enable_aug=False):
    crop = get_crop(image, rect)
    if crop is None:
        return None
    if crop.dtype != np.uint8:
        crop = (np.clip(crop, 0, 1) * 255).astype(np.uint8)
    crop = maybe_augment_crop(crop, enable_aug)
    crop_pil = Image.fromarray(crop)
    # torchvision expects PIL or torch tensor; we use numpy -> torch
    with torch.no_grad():
        inp = preprocess(crop_pil).unsqueeze(0)
        feat = model(inp).squeeze(0).cpu().numpy()
    norm = np.linalg.norm(feat)
    if norm > 1e-12:
        feat = feat / norm
    return feat


def build_samples(split_dir, proposals_dir, ann_path, tp, tn, out_size,
                  max_pos_per_img, max_neg_per_img, feature_type,
                  preprocess=None, model=None, augment=False, aug_pos=1):
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

        rects = np.load(prop_path)["rects"]
        pos_count = 0
        neg_count = 0

        for r in rects:
            max_iou = 0.0
            for gt in gts:
                max_iou = max(max_iou, iou(r, gt))

            if max_iou >= tp and pos_count < max_pos_per_img:
                if feature_type == "hog":
                    feat = extract_hog(image, r, out_size, enable_aug=False)
                else:
                    feat = extract_cnn(image, r, preprocess, model, enable_aug=False)
                if feat is not None:
                    X.append(feat)
                    y.append(1)
                    pos_count += 1
                    if augment:
                        for _ in range(aug_pos):
                            if feature_type == "hog":
                                feat_aug = extract_hog(image, r, out_size, enable_aug=True)
                            else:
                                feat_aug = extract_cnn(image, r, preprocess, model, enable_aug=True)
                            if feat_aug is not None:
                                X.append(feat_aug)
                                y.append(1)
            elif max_iou <= tn and neg_count < max_neg_per_img:
                if feature_type == "hog":
                    feat = extract_hog(image, r, out_size, enable_aug=False)
                else:
                    feat = extract_cnn(image, r, preprocess, model, enable_aug=False)
                if feat is not None:
                    X.append(feat)
                    y.append(0)
                    neg_count += 1

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
    parser.add_argument("--augment", dest="augment", action="store_true", help="Enable training-time augmentation on positive samples")
    parser.add_argument("--no_augment", dest="augment", action="store_false", help="Disable training-time augmentation on positive samples")
    parser.add_argument("--aug_pos", type=int, default=10, help="Number of augmented copies per positive sample")
    parser.set_defaults(augment=True)
    args = parser.parse_args()

    data_root = os.path.abspath(args.data_root)
    proposals_root = os.path.abspath(args.proposals_root)

    train_dir = os.path.join(data_root, "train")
    valid_dir = os.path.join(data_root, "valid")
    train_props = os.path.join(proposals_root, "train")
    valid_props = os.path.join(proposals_root, "valid")
    train_ann = os.path.join(train_dir, "_annotations.coco.json")
    valid_ann = os.path.join(valid_dir, "_annotations.coco.json")

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

    X_train, y_train = build_samples(
        train_dir, train_props, train_ann, args.tp, args.tn, args.out_size,
        args.max_pos_per_img, args.max_neg_per_img, args.feature,
        preprocess=preprocess, model=feature_model, augment=args.augment, aug_pos=args.aug_pos
    )
    X_valid, y_valid = build_samples(
        valid_dir, valid_props, valid_ann, args.tp, args.tn, args.out_size,
        args.max_pos_per_img, args.max_neg_per_img, args.feature,
        preprocess=preprocess, model=feature_model, augment=False, aug_pos=0
    )

    print("Train samples:", X_train.shape, "Pos:", int(y_train.sum()), "Neg:", int((y_train == 0).sum()))
    print("Valid samples:", X_valid.shape, "Pos:", int(y_valid.sum()), "Neg:", int((y_valid == 0).sum()))

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
                if args.feature == "hog":
                    feat = extract_hog(image, r, args.out_size, enable_aug=False)
                else:
                    feat = extract_cnn(image, r, preprocess, feature_model, enable_aug=False)
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
