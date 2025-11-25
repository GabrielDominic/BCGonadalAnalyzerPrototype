import os
import re
import cv2
import numpy as np
from skimage.feature.texture import graycomatrix, graycoprops
from skimage.feature import local_binary_pattern
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


folder = "Histological Slides - Male/"


images_original = []      
images_segmented = []    
filenames = []



def preprocess_image_segmented(img):
    img = cv2.resize(img, (512, 512))

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    L = cv2.GaussianBlur(L, (99, 99), 0)
    img_corrected = cv2.divide(lab[:, :, 0], L, scale=255)
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


    img = cv2.bilateralFilter(img, 7, 40, 40)


    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    L = clahe.apply(L)
    img = cv2.cvtColor(cv2.merge([L, A, B]), cv2.COLOR_LAB2BGR)

    lab2 = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    _, A, _ = cv2.split(lab2)

    A_blur = cv2.GaussianBlur(A, (9, 9), 0)

    _, mask = cv2.threshold(
        A_blur, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    if np.sum(mask) < 5000:
        return cv2.cvtColor(cv2.resize(img, (256, 256)), cv2.COLOR_BGR2RGB)

    coords = cv2.findNonZero(mask)
    x, y, w, h = cv2.boundingRect(coords)

    img_crop = img[y:y+h, x:x+w]

    img_crop = cv2.resize(img_crop, (256, 256))
    img_crop = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)

    return img_crop



def preprocess_image_clean(img):
    img = cv2.resize(img, (256, 256))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)



def sanitize_path_for_filename(path):
    path = re.sub(r'^[A-Za-z]:', '', path)
    path = path.replace('/', '_').replace('\\', '_')
    path = re.sub(r'[^A-Za-z0-9._-]', '_', path)
    path = re.sub(r'__+', '_', path)
    path = path.strip('_')
    return path



for root, dirs, files in os.walk(folder):
    for file in files:
        if file.lower().endswith(("4x.jpg", "4x.jpeg", "4x.png")):

            path = os.path.join(root, file)

            img = cv2.imread(path)

            clean_img = preprocess_image_clean(img)
            images_original.append(clean_img)

            seg_img = preprocess_image_segmented(img)
            images_segmented.append(seg_img)

            abs_path = os.path.abspath(path)
            safe_name = sanitize_path_for_filename(abs_path)

            filenames.append(safe_name)

# for root, dirs, files in os.walk(folder):
#     for file in files:
#         if file.lower().endswith(("4x.jpg", "4x.jpeg", "x.png")):

#             path = os.path.join(root, file)
#             abs_path = os.path.abspath(path)
#             base_safe = sanitize_path_for_filename(abs_path)

#             # Load original image
#             img = cv2.imread(path)

#             # ====== SPLIT INTO 4 GRIDS (2x2) ======
#             h, w = img.shape[:2]
#             mid_h, mid_w = h // 2, w // 2

#             grids = [
#                 img[0:mid_h,      0:mid_w],   # top-left
#                 img[0:mid_h,      mid_w:w],   # top-right
#                 img[mid_h:h,      0:mid_w],   # bottom-left
#                 img[mid_h:h,      mid_w:w],   # bottom-right
#             ]

#             # Process each grid separately
#             for i, grid in enumerate(grids):

#                 # Clean version (NO segmentation, used for saving)
#                 clean_img = preprocess_image_clean(grid)
#                 images_original.append(clean_img)

#                 # Segmented version (used for feature extraction)
#                 seg_img = preprocess_image_segmented(grid)
#                 images_segmented.append(seg_img)

#                 # Give unique grid-based filename
#                 # Example: home_user_project_slide1_img10x_grid0.jpg
#                 ext = os.path.splitext(file)[1]   # .jpg / .png
#                 grid_name = f"{base_safe}_grid{i}{ext}"

#                 filenames.append(grid_name)



print("Loaded:", len(images_original), "images")



def extract_glcm(img_gray):
    glcm = graycomatrix(
        img_gray,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True
    )
    return np.array([
        graycoprops(glcm, 'contrast')[0, 0],
        graycoprops(glcm, 'homogeneity')[0, 0],
        graycoprops(glcm, 'energy')[0, 0],
        graycoprops(glcm, 'correlation')[0, 0]
    ])


def extract_lbp(img_gray):
    lbp = local_binary_pattern(img_gray, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 59), range=(0, 58))
    hist = hist.astype("float")
    hist /= hist.sum()
    return hist


def extract_hsv_hist(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist(
        [hsv], [0, 1, 2], None,
        [8, 8, 8], [0, 180, 0, 256, 0, 256]
    )
    hist = cv2.normalize(hist, hist).flatten()
    return hist


def extract_morph_features(img_gray):
    _, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
    kernel = np.ones((3, 3), np.uint8)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return np.zeros(5)

    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)

    x, y, w, h = cv2.boundingRect(c)
    rect_area = w * h
    extent = area / rect_area if rect_area != 0 else 0

    perimeter = cv2.arcLength(c, True)
    circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter != 0 else 0

    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area != 0 else 0

    return np.array([area, extent, perimeter, circularity, solidity])



feature_vectors = []

for img in images_segmented:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    glcm_feat = extract_glcm(gray)
    lbp_feat = extract_lbp(gray)
    hsv_feat = extract_hsv_hist(img)
    morph_feat = extract_morph_features(gray)

    full_feature = np.hstack([glcm_feat, lbp_feat, hsv_feat, morph_feat])
    feature_vectors.append(full_feature)

feature_vectors = np.array(feature_vectors)




pca = PCA(n_components=20)
pca_features = pca.fit_transform(feature_vectors)

k = 2
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(pca_features)



print("\nCluster Assignments:")

cluster_groups = {}
for fname, cid in zip(filenames, clusters):
    cluster_groups.setdefault(cid, []).append(fname)

for cid in sorted(cluster_groups.keys()):
    print(f"\n===== CLUSTER {cid} ({len(cluster_groups[cid])} images) =====")
    for fname in cluster_groups[cid]:
        print(fname)

print("\n")




def save_cluster_images(cluster_id, output_root="clustered_images"):
    cluster_dir = os.path.join(output_root, f"cluster_{cluster_id}")
    os.makedirs(cluster_dir, exist_ok=True)

    saved = 0
    for img, cid, fname in zip(images_original, clusters, filenames):
        if cid != cluster_id:
            continue

        base = os.path.basename(fname)
        save_name = f"{saved:03d}_{base}"
        save_path = os.path.join(cluster_dir, save_name)

        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, bgr)

        saved += 1

    print(f"Saved {saved} images to {cluster_dir}")


for cid in range(k):
    save_cluster_images(cid)
