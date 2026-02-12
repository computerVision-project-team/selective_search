"""
Generate selective search region proposals for the balloon dataset.
Task 5.2.1: save proposals to disk so later steps can reuse them.
"""

import os
import argparse
import numpy as np
import skimage.io

from selective_search import selective_search


def iter_images(root, exts):
    for name in sorted(os.listdir(root)):
        if os.path.splitext(name.lower())[1] in exts:
            yield os.path.join(root, name)


def save_proposals(image_path, out_dir, scale, sigma, min_size, max_merges):
    image = skimage.io.imread(image_path)
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    if image.shape[2] > 3:
        image = image[:, :, :3]

    _, regions = selective_search(
        image, scale=scale, sigma=sigma, min_size=min_size, max_merges=max_merges
    )

    rects = np.array([r["rect"] for r in regions], dtype=np.int32)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, os.path.basename(image_path) + ".npz")
    np.savez_compressed(out_path, rects=rects)
    return rects.shape[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="../data/balloon_dataset")
    parser.add_argument("--out_root", default="../data/balloon_dataset/proposals")
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    parser.add_argument("--scale", type=float, default=300)
    parser.add_argument("--sigma", type=float, default=0.8)
    parser.add_argument("--min_size", type=int, default=50)
    parser.add_argument("--max_merges", type=int, default=None)
    args = parser.parse_args()

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    data_root = os.path.abspath(args.data_root)
    out_root = os.path.abspath(args.out_root)

    for split in args.splits:
        in_dir = os.path.join(data_root, split)
        out_dir = os.path.join(out_root, split)
        if not os.path.isdir(in_dir):
            print("Missing split dir:", in_dir)
            continue
        total = 0
        for img_path in iter_images(in_dir, exts):
            n = save_proposals(
                img_path, out_dir, args.scale, args.sigma, args.min_size, args.max_merges
            )
            total += 1
            print("Saved", n, "proposals for", os.path.basename(img_path))
        print("Done split", split, "images:", total)


if __name__ == "__main__":
    main()
