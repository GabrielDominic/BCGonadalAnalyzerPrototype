import os
import re
import random
import cv2
import numpy as np
from skimage.feature.texture import graycomatrix, graycoprops
from skimage.feature import local_binary_pattern
from sklearn.decomposition import PCA

folder = "Histological Slides - Female/"

# ============================
# Image Preprocessing
# ============================

def preprocess_image_segmented(img):
    img = cv2.resize(img, (512, 512))

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    L = cv2.GaussianBlur(L, (99, 99), 0)
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    img = cv2.bilateralFilter(img, 7, 40, 40)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    L = clahe.apply(L)
    lab = cv2.merge((L, A, B))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    #this is the part that detects the low-level stuff
    # -----------------
    # Hysteresis Inspired AHE
    # -----------------
    himg = img.copy()
    for tiles in [64, 32, 16, 8]:
        lab = cv2.cvtColor(himg, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(tiles, tiles))
        L = clahe.apply(L)
        lab = cv2.merge((L, A, B))
        himg = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    gb = cv2.GaussianBlur(himg, (5, 5), 0)
    hsv = cv2.cvtColor(gb, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    S = cv2.GaussianBlur(S, (5, 5), 0)
    _, S_binary = cv2.threshold(S, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    S_inv = cv2.bitwise_not(S_binary)
    S_further = cv2.GaussianBlur(S_inv, (5, 5), 0)
    _, S_binary2 = cv2.threshold(S_further, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    S_and = cv2.bitwise_and(S_binary, S_binary2)
    S_and_eroded = cv2.erode(S_and, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)

    contours, _ = cv2.findContours(S_and_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cc_mask = np.zeros_like(S_and_eroded)

    for c in contours:
        area = cv2.contourArea(c)
        if area > 1000:
            cv2.drawContours(cc_mask, [c], -1, 255, -1)

    # Morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cc_mask = cv2.morphologyEx(cc_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # If too little mask, fallback to resized original
    if np.sum(cc_mask) < 5000:
        return cv2.cvtColor(cv2.resize(img, (256, 256)), cv2.COLOR_BGR2RGB)

    coords = cv2.findNonZero(cc_mask)
    x, y, w, h = cv2.boundingRect(coords)

    img_crop = img[y:y+h, x:x+w]
    img_crop = cv2.resize(img_crop, (256, 256))
    return cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)


def preprocess_image_clean(img):
    img = cv2.resize(img, (256, 256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# ============================
# Feature Extraction
# ============================

def extract_edge_features(img_gray):
    # Sobel
    sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)

    # Edge density via Canny
    edges = cv2.Canny(img_gray, 50, 150)
    edge_density = edges.mean()  # proportion of edge pixels

    # Simple statistics
    sobel_mean = sobel_mag.mean()
    sobel_std = sobel_mag.std()

    return np.array([sobel_mean, sobel_std, edge_density], dtype=np.float32)

def extract_glcm(img_gray):
    glcm = graycomatrix(
        img_gray,
        distances=[1, 2, 3],
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=256,
        symmetric=True,
        normed=True
    )

    props = {}
    for prop in ['contrast', 'homogeneity', 'energy', 'correlation']:
        vals = graycoprops(glcm, prop).ravel()
        props[prop] = np.mean(vals), np.std(vals)

    features = []
    for prop in ['contrast', 'homogeneity', 'energy', 'correlation']:
        mean_val, std_val = props[prop]
        features.append(mean_val)
        features.append(std_val)

    return np.array(features, dtype=np.float32)


def extract_lbp(img_gray, P=24, R=3):
    lbp = local_binary_pattern(img_gray, P, R, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist


def extract_color_moments(img_rgb):
    channels = cv2.split(img_rgb)
    moments = []
    for c in channels:
        mean = np.mean(c)
        std = np.std(c)
        skewness = np.mean((c - mean)**3) / (np.std(c)**3 + 1e-7)
        moments.extend([mean, std, skewness])
    return np.array(moments, dtype=np.float32)


def extract_morph_features(img_gray):
    _, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask_foreground = (thresh == 0).astype(np.uint8)
    area_foreground = np.sum(mask_foreground)

    contours, _ = cv2.findContours(mask_foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return np.array([0, 0, 0], dtype=np.float32)

    main_contour = max(contours, key=cv2.contourArea)
    area_contour = cv2.contourArea(main_contour)
    perimeter = cv2.arcLength(main_contour, True)

    if area_contour <= 0 or perimeter <= 0:
        circularity = 0.0
    else:
        circularity = 4 * np.pi * area_contour / (perimeter**2)

    return np.array([area_foreground, area_contour, circularity], dtype=np.float32)

# ============================
# Utility Functions
# ============================

def sanitize_path_for_filename(path):
    base = os.path.basename(path)
    sanitized = re.sub(r'[^a-zA-Z0-9_.-]', '_', base)
    return sanitized[:200]

# ============================
# Main Loading & Feature Pipeline
# ============================

def load_and_extract_features(root_folder=folder):
    images_original = []
    images_segmented = []
    filenames = []
    orig_paths = []

    for dirpath, dirnames, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(("20x.png", "20x.jpg", "20x.jpeg")):
                abs_path = os.path.join(dirpath, file)
                img = cv2.imread(abs_path)
                if img is None:
                    print(f"Warning: Could not read image: {abs_path}")
                    continue

                seg = preprocess_image_segmented(img)
                cln = preprocess_image_clean(img)

                images_original.append(cln)
                images_segmented.append(seg)
                orig_paths.append(abs_path)
                filenames.append(sanitize_path_for_filename(abs_path))

    # ----------------------------
    # Generalized resampling across named gonadal stages
    # ----------------------------
    STAGE_KEYWORDS = ["spawning", "mature", "spent"]

    # Determine labels from the original directory path (folder names)
    labels = []
    for p in orig_paths:
        comps = [c.lower() for c in os.path.normpath(p).split(os.sep) if c]
        found = None
        for sk in STAGE_KEYWORDS:
            if any(sk in c for c in comps):
                found = sk
                break
        labels.append(found if found is not None else 'other')

    # Build index lists per stage
    stage_indices = {}
    for i, l in enumerate(labels):
        stage_indices.setdefault(l, []).append(i)

    # Report counts before balancing
    counts_report = ", ".join(f"{k}: {len(v)}" for k, v in stage_indices.items())
    print(f"Found counts by stage (before balancing): {counts_report}")

    # Consider only named stages (exclude 'other') that have at least one sample
    named_stages = [s for s in STAGE_KEYWORDS if s in stage_indices and len(stage_indices[s]) > 0]

    if len(named_stages) > 1:
        # Hybrid upsample–downsample using the average count across named stages as target
        total_named = sum(len(stage_indices[s]) for s in named_stages)
        target_per_stage = max(1, int(round(total_named / len(named_stages))))
        print(f"Balancing named stages to approximately {target_per_stage} samples each using hybrid upsample–downsample.")

        random.seed(42)
        indices_to_keep = []

        for s in named_stages:
            idxs = stage_indices[s]
            n = len(idxs)
            if n == 0:
                continue

            if n > target_per_stage:
                # Downsample majority stages
                keep = random.sample(idxs, target_per_stage)
            elif n < target_per_stage:
                # Upsample minority stages by sampling with replacement
                keep = idxs[:]
                needed = target_per_stage - n
                extra = [random.choice(idxs) for _ in range(needed)]
                keep.extend(extra)
            else:
                # Already at target size
                keep = idxs[:]

            indices_to_keep.extend(keep)

        # Keep all 'other' samples unchanged (if they exist)
        if 'other' in stage_indices:
            indices_to_keep.extend(stage_indices['other'])

        # Sort indices and apply to all lists
        indices_to_keep = sorted(indices_to_keep)

        images_original = [images_original[i] for i in indices_to_keep]
        images_segmented = [images_segmented[i] for i in indices_to_keep]
        filenames = [filenames[i] for i in indices_to_keep]
        orig_paths = [orig_paths[i] for i in indices_to_keep]

        # Report counts after balancing
        counts_report_after = ", ".join(
            f"{k}: {len([i for i in indices_to_keep if labels[i] == k])}"
            for k in stage_indices.keys()
        )
        print(
            f"Balanced named stages to ~{target_per_stage} each. "
            f"New dataset size: {len(filenames)}. "
            f"Counts after balancing: {counts_report_after}"
        )
    else:
        print("No balancing applied (not enough named stages present to balance)")

    feature_vectors = []
    for img in images_segmented:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        glcm_feat = extract_glcm(gray)
        lbp_feat = extract_lbp(gray)
        cm_feat = extract_color_moments(img)
        morph = extract_morph_features(gray)
        edge = extract_edge_features(gray)

        full = np.hstack([edge])
        feature_vectors.append(full)

    feature_vectors = np.array(feature_vectors)

    # Choose whether to apply PCA. Require more than 50 samples to run PCA.
    if feature_vectors.size == 0:
        raise ValueError("No feature vectors found for PCA. Check that images were loaded correctly.")
    n_samples, n_features = feature_vectors.shape

    if n_samples > 50:
        requested_components = 20
        max_components = min(requested_components, n_samples, n_features)
        if max_components < requested_components:
            print(f"Note: reducing PCA components from {requested_components} to {max_components} (n_samples={n_samples}, n_features={n_features})")

        pca = PCA(n_components=max_components)
        pca_features = pca.fit_transform(feature_vectors)
    else:
        print(f"Skipping PCA because sample size is {n_samples} (<=50). Returning raw feature vectors.")
        pca_features = feature_vectors

    # Return full original paths rather than sanitized basenames so
    # downstream clustering scripts can detect folder tokens like
    # 'Male'/'Female' and stage names when they build save names.
    return images_original, orig_paths, pca_features
