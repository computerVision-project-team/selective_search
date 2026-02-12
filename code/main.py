'''
@author: Prathmesh R Madhu.
For educational purposes only
'''

# -*- coding: utf-8 -*-
from __future__ import (
    division,
    print_function,
)

import os
import time
import skimage.data
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import skimage.transform
import skimage.util
import numpy as np
from tqdm import tqdm

from selective_search import selective_search

# Toggle for speed vs. full-quality runs
FAST_MODE = False
SHOW_PLOTS = True


def _ensure_uint8(image):
    if image.dtype != "uint8":
        if np.issubdtype(image.dtype, np.floating):
            maxv = float(image.max()) if image.size else 1.0
            if maxv > 1.0:
                image = image / 255.0
        image = skimage.util.img_as_ubyte(image)
    return image


def process_image(image_path, results_dir):
    image = skimage.io.imread(image_path)
    print("Image:", image_path, "shape:", image.shape)

    # Optional: downscale large images for faster selective search
    max_side = 450 if FAST_MODE else None
    h, w = image.shape[:2]
    if max_side is not None and max(h, w) > max_side:
        scale = max_side / float(max(h, w))
        try:
            image = skimage.transform.rescale(
                image, scale, channel_axis=2, anti_aliasing=True, preserve_range=True
            )
        except TypeError:
            image = skimage.transform.rescale(
                image, scale, multichannel=True, anti_aliasing=True, preserve_range=True
            )
        image = _ensure_uint8(image)
        print("Rescaled to:", image.shape)
    else:
        image = _ensure_uint8(image)

    # perform selective search
    if FAST_MODE:
        scale = 150
        min_size = 80
        max_merges = 3500
        min_rect_size = 500
        max_aspect = 2.5
    else:
        scale = 200
        min_size = 40
        max_merges = 4000
        min_rect_size = 800
        max_aspect = 1.5

    t0 = time.time()
    image_label, regions = selective_search(
        image,
        scale=scale,
        min_size=min_size,
        max_merges=max_merges,
    )
    print("Selective search time: %.2fs" % (time.time() - t0))

    candidates = set()
    for r in regions:
        # excluding same rectangle (with different segments)
        if r["rect"] in candidates:
            continue

        if r["size"] < min_rect_size:
            continue

        # exclude very distorted rects
        x, y, w, h = r["rect"]
        if w == 0 or h == 0:
            continue
        if w / h > max_aspect or h / w > max_aspect:
            continue

        candidates.add(r["rect"])

    # Draw rectangles on the original image
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(image)
    for x, y, w, h in candidates:
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor="red", linewidth=1
        )
        ax.add_patch(rect)
    plt.axis("off")

    # saving the image
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    out_path = os.path.join(results_dir, os.path.basename(image_path))
    fig.savefig(out_path)

    if SHOW_PLOTS:
        try:
            plt.show()
        except NotImplementedError:
            print("Display backend error; image saved to results/.")
    plt.close(fig)


def main():
    base = os.path.dirname(__file__)  # .../code
    data_root = os.path.join(base, "..", "data")
    results_root = os.path.join(base, "..", "results")

    domains = ["chrisarch", "arthist", "classarch"]
    exts = {".jpg", ".jpeg", ".png", ".bmp"}

    # build image list for simple progress reporting
    image_list = []
    for d in domains:
        in_dir = os.path.join(data_root, d)
        if not os.path.isdir(in_dir):
            continue
        for name in sorted(os.listdir(in_dir)):
            if os.path.splitext(name.lower())[1] in exts:
                image_list.append((d, os.path.join(in_dir, name)))

    total = len(image_list)
    idx = 0

    for d, img_path in tqdm(image_list, total=total, desc="selective search"):
        idx += 1
        out_dir = os.path.join(results_root, d)
        process_image(img_path, out_dir)


if __name__ == "__main__":
    main()
