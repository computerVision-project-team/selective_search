'''
@author: Prathmesh R Madhu.
For educational purposes only
'''
# -*- coding: utf-8 -*-
from __future__ import division

import skimage.io
import skimage.color
import skimage.transform
import skimage.util
import skimage.segmentation
import skimage.filters
from skimage.segmentation import felzenszwalb

import numpy as np

def generate_segments(im_orig, scale, sigma, min_size):
    # im_orig: H x W x 3 (RGB), usually float in [0,1] or uint8 in [0,255]
    labels = felzenszwalb(im_orig, scale=scale, sigma=sigma, min_size=min_size)
    labels = labels.astype(np.int32)
    im_4c = np.dstack((im_orig, labels))
    return im_4c

def hist_intersection(h1, h2):
    """
    Histogram intersection:
    sum_i min(h1[i], h2[i])
    Assumes h1, h2 are L1-normalized.
    """
    return np.sum(np.minimum(h1, h2))

def sim_colour(r1, r2):
    h1 = r1.get("hist_c", r1.get("hist_color", r1.get("colour_hist")))
    h2 = r2.get("hist_c", r2.get("hist_color", r2.get("colour_hist")))
    if h1 is None or h2 is None:
        raise KeyError("Colour histogram not found in region dict (expected key like 'hist_c').")
    return hist_intersection(h1, h2)


def sim_texture(r1, r2):
    h1 = r1.get("hist_t", r1.get("hist_texture", r1.get("texture_hist")))
    h2 = r2.get("hist_t", r2.get("hist_texture", r2.get("texture_hist")))
    if h1 is None or h2 is None:
        raise KeyError("Texture histogram not found in region dict (expected key like 'hist_t').")
    return hist_intersection(h1, h2)


def sim_size(r1, r2, imsize):
    s1 = r1.get("size", r1.get("area"))
    s2 = r2.get("size", r2.get("area"))
    if s1 is None or s2 is None:
        raise KeyError("Region size not found (expected key 'size' or 'area').")
    return 1.0 - (float(s1) + float(s2)) / float(imsize)


def sim_fill(r1, r2, imsize):
    s1 = r1.get("size", r1.get("area"))
    s2 = r2.get("size", r2.get("area"))
    if s1 is None or s2 is None:
        raise KeyError("Region size not found (expected key 'size' or 'area').")

    # bbox：常见存法 1）min_x/min_y/max_x/max_y  2）rect=(x,y,w,h)
    if "min_x" in r1 and "min_y" in r1 and "max_x" in r1 and "max_y" in r1:
        min_x = min(r1["min_x"], r2["min_x"])
        min_y = min(r1["min_y"], r2["min_y"])
        max_x = max(r1["max_x"], r2["max_x"])
        max_y = max(r1["max_y"], r2["max_y"])
        bbox_area = float((max_x - min_x) * (max_y - min_y))
    elif "rect" in r1 and "rect" in r2:
        # rect = (x, y, w, h)
        x1, y1, w1, h1 = r1["rect"]
        x2, y2, w2, h2 = r2["rect"]
        min_x = min(x1, x2)
        min_y = min(y1, y2)
        max_x = max(x1 + w1, x2 + w2)
        max_y = max(y1 + h1, y2 + h2)
        bbox_area = float((max_x - min_x) * (max_y - min_y))
    else:
        raise KeyError("BBox not found. Expected keys min_x/min_y/max_x/max_y or rect=(x,y,w,h).")

    empty = bbox_area - float(s1) - float(s2)
    return 1.0 - empty / float(imsize)

SIM_WEIGHTS = (1.0, 1.0, 1.0, 1.0)

def calc_sim(r1, r2, imsize):
    a1, a2, a3, a4 = SIM_WEIGHTS
    return (a1 * sim_colour(r1, r2)
            + a2 * sim_texture(r1, r2)
            + a3 * sim_size(r1, r2, imsize)
            + a4 * sim_fill(r1, r2, imsize))

def calc_colour_hist(img):
    BINS = 25
    hist = np.array([], dtype=np.float32)

    # img: (N,3) HSV values in [0,1]
    for c in range(3):
        hc, _ = np.histogram(img[:, c], bins=BINS, range=(0.0, 1.0))
        hist = np.concatenate([hist, hc.astype(np.float32)])

    # L1 normalize
    s = hist.sum()
    if s > 0:
        hist /= s

    return hist

def calc_texture_gradient(img):
    """
    Texture gradient using 8 directional derivatives per channel.
    Returns (H, W, 24): 8 directions * 3 channels.
    """
    h, w, ch = img.shape
    out = np.zeros((h, w, ch * 8), dtype=np.float32)

    for c in range(ch):
        gx = skimage.filters.sobel_h(img[:, :, c])
        gy = skimage.filters.sobel_v(img[:, :, c])
        mag = np.sqrt(gx * gx + gy * gy)

        mmax = mag.max()
        if mmax > 0:
            mag = mag / mmax

        theta = np.arctan2(gy, gx)  # [-pi, pi]
        theta = (theta + 2.0 * np.pi) % (2.0 * np.pi)  # [0, 2pi)
        bin_idx = np.floor(theta / (2.0 * np.pi) * 8.0).astype(np.int32)
        bin_idx = np.clip(bin_idx, 0, 7)

        for b in range(8):
            out[:, :, c * 8 + b] = mag * (bin_idx == b)

    return out

def calc_texture_hist(img):
    BINS = 10
    hist = np.array([], dtype=np.float32)

    # img: (N, 24) directional gradient magnitudes in [0,1]
    for c in range(img.shape[1]):
        hc, _ = np.histogram(img[:, c], bins=BINS, range=(0.0, 1.0))
        hist = np.concatenate([hist, hc.astype(np.float32)])

    # L1 normalize
    s = hist.sum()
    if s > 0:
        hist /= s

    return hist

def extract_regions(img):
    '''
    Task 2.5: Generate regions denoted as datastructure R
    - Convert image to hsv color map
    - Count pixel positions
    - Calculate the texture gradient
    - calculate color and texture histograms
    - Store all the necessary values in R.
    '''
    R = {}
    # split
    rgb = img[:, :, :3]
    labels = img[:, :, 3].astype(np.int32)

    # hsv in [0,1]
    hsv = skimage.color.rgb2hsv(rgb)

    # texture gradient (8 directions per channel)
    tex = calc_texture_gradient(hsv)

    # build regions
    for lab in np.unique(labels):
        ys, xs = np.where(labels == lab)
        if len(xs) == 0:
            continue

        min_x, max_x = xs.min(), xs.max() + 1  # +1 -> consistent bbox width
        min_y, max_y = ys.min(), ys.max() + 1

        size = int(len(xs))

        hsv_pixels = hsv[ys, xs, :]  # (N,3)
        tex_pixels = tex[ys, xs, :]  # (N,24)

        R[lab] = {
            "min_x": int(min_x),
            "min_y": int(min_y),
            "max_x": int(max_x),
            "max_y": int(max_y),
            "size": size,
            "labels": [int(lab)],
            "hist_c": calc_colour_hist(hsv_pixels),
            "hist_t": calc_texture_hist(tex_pixels),
        }

    return R

def extract_neighbours(regions):

    def intersect(a, b):
        if (a["min_x"] < b["min_x"] < a["max_x"]
                and a["min_y"] < b["min_y"] < a["max_y"]) or (
            a["min_x"] < b["max_x"] < a["max_x"]
                and a["min_y"] < b["max_y"] < a["max_y"]) or (
            a["min_x"] < b["min_x"] < a["max_x"]
                and a["min_y"] < b["max_y"] < a["max_y"]) or (
            a["min_x"] < b["max_x"] < a["max_x"]
                and a["min_y"] < b["min_y"] < a["max_y"]):
            return True
        return False

    # Hint 1: List of neighbouring regions
    # Hint 2: The function intersect has been written for you and is required to check neighbours
    neighbours = []
    ### YOUR CODE HERE ###
    items = list(regions.items())

    for idx1 in range(len(items)):
        a_id, a = items[idx1]
        for idx2 in range(idx1 + 1, len(items)):
            b_id, b = items[idx2]

            if intersect(a, b) or intersect(b, a):
                neighbours.append(((a_id, a), (b_id, b)))

    return neighbours

def merge_regions(r1, r2):
    new_size = r1["size"] + r2["size"]
    rt = {}

    # bbox
    rt["min_x"] = min(r1["min_x"], r2["min_x"])
    rt["min_y"] = min(r1["min_y"], r2["min_y"])
    rt["max_x"] = max(r1["max_x"], r2["max_x"])
    rt["max_y"] = max(r1["max_y"], r2["max_y"])

    # size
    rt["size"] = new_size

    # labels（把两个 region 的 label 列表合并）
    rt["labels"] = r1["labels"] + r2["labels"]

    # colour/texture histogram：按面积加权平均（保持 L1 normalize 的性质）
    rt["hist_c"] = (r1["hist_c"] * r1["size"] + r2["hist_c"] * r2["size"]) / new_size
    rt["hist_t"] = (r1["hist_t"] * r1["size"] + r2["hist_t"] * r2["size"]) / new_size

    return rt


def selective_search(image_orig, scale=1.0, sigma=0.8, min_size=50, max_merges=None):
    '''
    Selective Search for Object Recognition" by J.R.R. Uijlings et al.
    :arg:
        image_orig: np.ndarray, Input image
        scale: int, determines the cluster size in felzenszwalb segmentation
        sigma: float, width of Gaussian kernel for felzenszwalb segmentation
        min_size: int, minimum component size for felzenszwalb segmentation

    :return:
        image: np.ndarray,
            image with region label
            region label is stored in the 4th value of each pixel [r,g,b,(region)]
            [
                {
                    'rect': (left, top, width, height),
                    'labels': [...],
                    'size': component_size
                },
                ...
            ]
    '''

    # Checking the 3 channel of input image
    assert image_orig.shape[2] == 3, "Please use image with three channels."
    imsize = image_orig.shape[0] * image_orig.shape[1]

    # Task 1: Load image and get smallest regions. Refer to `generate_segments` function.
    image = generate_segments(image_orig, scale, sigma, min_size)

    if image is None:
        return None, {}

    # Task 2: Extracting regions from image
    # Task 2.1-2.4: Refer to functions "sim_colour", "sim_texture", "sim_size", "sim_fill"
    # Task 2.5: Refer to function "extract_regions". You would also need to fill "calc_colour_hist",
    # "calc_texture_hist" and "calc_texture_gradient" in order to finish task 2.5.
    R = extract_regions(image)

    # Task 3: Extracting neighbouring information
    # Refer to function "extract_neighbours"
    neighbours = extract_neighbours(R)

    # Calculating initial similarities
    S = {}
    for (ai, ar), (bi, br) in neighbours:
        S[(ai, bi)] = calc_sim(ar, br, imsize)

    # Hierarchical search for merging similar regions
    merges = 0
    while S != {}:

        # Get highest similarity
        i, j = sorted(S.items(), key=lambda i: i[1])[-1][0]

        # Task 4: Merge corresponding regions. Refer to function "merge_regions"
        t = max(R.keys()) + 1
        R[t] = merge_regions(R[i], R[j])
        merges += 1
        if max_merges is not None and merges >= max_merges:
            break

        # Task 5: Mark similarities for regions to be removed
        keys_to_delete = []
        for (a, b) in list(S.keys()):
            if a in (i, j) or b in (i, j):
                keys_to_delete.append((a, b))

        # Task 6: Remove old similarities of related regions
        for k in keys_to_delete:
            del S[k]

        # Task 7: Calculate similarities with the new region
        def intersect_bbox(a, b):
            if (a["min_x"] < b["min_x"] < a["max_x"]
                and a["min_y"] < b["min_y"] < a["max_y"]) or (
                    a["min_x"] < b["max_x"] < a["max_x"]
                    and a["min_y"] < b["max_y"] < a["max_y"]) or (
                    a["min_x"] < b["min_x"] < a["max_x"]
                    and a["min_y"] < b["max_y"] < a["max_y"]) or (
                    a["min_x"] < b["max_x"] < a["max_x"]
                    and a["min_y"] < b["min_y"] < a["max_y"]):
                return True
            return False

        for k in list(R.keys()):
            if k == t:
                continue
            if intersect_bbox(R[t], R[k]) or intersect_bbox(R[k], R[t]):
                a, b = (t, k) if t < k else (k, t)
                S[(a, b)] = calc_sim(R[a], R[b], imsize)


    # Task 8: Generating the final regions from R
    regions = []
    ### YOUR CODE HERE ###
    seen = set()
    for k, r in R.items():
        x, y = r["min_x"], r["min_y"]
        w, h = r["max_x"] - r["min_x"], r["max_y"] - r["min_y"]

        rect = (x, y, w, h)

        # 可选：去重（很多 region 会给出同一个 bbox）
        if rect in seen:
            continue
        seen.add(rect)

        regions.append({
            "rect": rect,
            "size": r["size"],
            "labels": r["labels"]
        })
    print("Number of region proposals:", len(regions))
    return image, regions
