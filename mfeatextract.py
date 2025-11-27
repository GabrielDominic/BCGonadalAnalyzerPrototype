import os
import re
import cv2
import numpy as np
from skimage.feature.texture import graycomatrix, graycoprops
from skimage.feature import local_binary_pattern
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pickle

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

## #Data Preparation
dir = 'C:\\GitProjects\\BCGonadalAnalyzerPrototype\\imagedataset'
categories = ['spawning', 'spent']

data = []
feature_vectors = []
labels = []
images_original = []      
filenames = []

for category in categories:
    images_segmented = []    
    path = os.path.join(dir, category)
    label = categories.index(category)

    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        try:    
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                print(f'Image not found or unreadable: {img_path}')
        
            #Preprocessing
            clean_img = preprocess_image_clean(img)
            images_original.append(clean_img)
            seg_img = preprocess_image_segmented(img)
            images_segmented.append((seg_img, img_path))

            #Feature Extraction
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            glcm_feat = extract_glcm(gray)
            lbp_feat = extract_lbp(gray)
            hsv_feat = extract_hsv_hist(img)
            morph_feat = extract_morph_features(gray)

            full_feature = np.hstack([glcm_feat, lbp_feat, hsv_feat, morph_feat])
            labels.append(label)
            filenames.append(img_path)
            data.append([full_feature, label, img_path])
        except Exception as e:
            print(f'Failed to process {img_path}: {e}')

print(f'Features Extracted: {len(data)}')
print("Loaded:", len(images_original), "images")

# Saving Data
pick_in = open('mfeaturefile.pickle', 'wb')
pickle.dump(data, pick_in)
pick_in.close()

