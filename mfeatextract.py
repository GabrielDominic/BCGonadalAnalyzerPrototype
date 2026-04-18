import os
import re
import cv2
import numpy as np
from skimage.feature.texture import graycomatrix, graycoprops
from skimage.feature import local_binary_pattern
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pickle
import time

def preprocess_image_segmented(img):
    # img = cv2.resize(img, (512, 512))

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

    # if np.sum(mask) < 5000:
    #     return cv2.cvtColor(cv2.resize(img, (256, 256)), cv2.COLOR_BGR2RGB)

    coords = cv2.findNonZero(mask)
    x, y, w, h = cv2.boundingRect(coords)

    img_crop = img[y:y+h, x:x+w]

    img_crop = cv2.resize(img_crop, (256, 256))
    img_crop = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)

    return img_crop



def preprocess_image_clean(img):
    img = cv2.resize(img, (256, 256))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def extract_edge_features(img_gray):
    # Sobel
    sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)

    # Edge density via Canny
    edges = cv2.Canny(img_gray, 50, 150)
    edge_density = edges.mean()  # proportion of edge pixels
    # cv2.imshow('edges', edges)

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

def extract_gamete_area(img_gray, image, folder_name):
    ##TISSUE MASK##
    #Otsu's Thresholding to create a binary mask of the tissue
    ret, tissue_mask = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # cv2.imshow('Tissue Mask', tissue_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    total_tissue_pixels = cv2.countNonZero(tissue_mask)

    #Calculate Gamete Area
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    #Variation in histological slides 
    print(f'Folder name: {folder_name}')
    if folder_name == 'M':
        lower_red = np.array([110, 40, 20])
        upper_red = np.array([170, 255, 150])
    elif folder_name == 'F':
        lower_red = np.array([110, 40, 20])
        upper_red = np.array([170, 255, 200])

    gamete_mask = cv2.inRange(hsv, lower_red, upper_red)
    gamete_mask = cv2.morphologyEx(gamete_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    gamete_mask = cv2.bitwise_and(gamete_mask, gamete_mask, mask=tissue_mask)
    gamete_pixels = cv2.countNonZero(gamete_mask)

    #CALCULATE GAMETE VOLUME AREA FRACTION
    if total_tissue_pixels > 0:
        area_fraction = gamete_pixels / total_tissue_pixels
    else:
        area_fraction = 0
    print("Total Tissue Pixels:", total_tissue_pixels)
    print("Gamete pixels:", gamete_pixels)
    print("Gamete Area Fraction:", area_fraction)

    return np.array([total_tissue_pixels, gamete_pixels, area_fraction], dtype=np.float32)

## #Data Preparation
# dir = 'C:\\GitProjects\\BCGonadalAnalyzerPrototype\\output_images\\M'
# dir = 'D:\\SP\\BCDataset15-4-2026\\M'
dir = 'C:\\GitProjects\\BCGonadalAnalyzerPrototype\\normalized_updated_dataset\\M'
categories = ['developing','maturing','spawning','spent']

folder_name = os.path.basename(dir)

#Time the feature extraction process
glcm_times = []
lbp_times = []
cm_times = []
morph_times = []
edge_times = []
gamete_times = []

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
            # clean_img = preprocess_image_clean(img)
            # images_original.append(clean_img)
            # seg_img = preprocess_image_segmented(img)
            # images_segmented.append((seg_img, img_path))

            # #Feature Extraction
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            total_start = time.time()

            start = time.time()
            glcm_feat = extract_glcm(gray)
            end = time.time()
            glcm_times.append(end - start)
            print(f'GLCM features for {img_path} took {end - start:.2f} seconds')
            
            start = time.time()
            lbp_feat = extract_lbp(gray)
            end = time.time()
            lbp_times.append(end - start)
            print(f'LBP features for {img_path} took {end - start:.2f} seconds')

            start = time.time()
            cm_feat = extract_color_moments(img)
            end = time.time()
            cm_times.append(end - start)
            print(f'Color moments for {img_path} took {end - start:.2f} seconds')
            
            start = time.time()
            morph_feat = extract_morph_features(gray)
            end = time.time()
            morph_times.append(end - start)
            print(f'Morphological features for {img_path} took {end - start:.2f} seconds')
            
            start = time.time()
            edge = extract_edge_features(gray)
            end = time.time()
            edge_times.append(end - start)
            print(f'Edge features for {img_path} took {end - start:.2f} seconds')
            
            start = time.time()
            gamete_area = extract_gamete_area(gray, img, folder_name)
            end = time.time()
            gamete_times.append(end - start)
            print(f'Gamete area features for {img_path} took {end - start:.2f} seconds')

            full_feature = np.hstack([glcm_feat, lbp_feat, cm_feat, morph_feat, edge, gamete_area])
            # full_feature = np.hstack([lbp_feat, cm_feat, morph_feat, edge])
            labels.append(label)
            filenames.append(img_path)
            data.append([full_feature, label, img_path])
        except Exception as e:
            print(f'Failed to process {img_path}: {e}')

print(f'Features Extracted: {len(data)}')
print("Loaded:", len(images_original), "images")

print(f'Average GLCM extraction time: {np.mean(glcm_times):.2f} seconds')
print(f'Average LBP extraction time: {np.mean(lbp_times):.2f} seconds')
print(f'Average Color Moments extraction time: {np.mean(cm_times):.2f} seconds')
print(f'Average Morphological extraction time: {np.mean(morph_times):.2f} seconds')
print(f'Average Edge extraction time: {np.mean(edge_times):.2f} seconds')
print(f'Average Gamete Area extraction time: {np.mean(gamete_times):.2f} seconds')

# Saving Data
pick_in = open('CTGAFmaleupdatedFeatures.pickle', 'wb')
pickle.dump(data, pick_in)
pick_in.close()

cv2.waitKey(0)
cv2.destroyAllWindows()