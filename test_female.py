import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage.feature import local_binary_pattern

def preprocess_image(img):
    img = cv2.resize(img, (256, 256))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def resize_for_display(img, size=(512, 512)):
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

def extract_glcm(img_gray):
    glcm = graycomatrix(
        img_gray,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True
    )

    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]

    return np.array([contrast, homogeneity, energy, correlation], dtype=np.float32)

def extract_lbp(img_gray, P=24, R=3):
    lbp = local_binary_pattern(img_gray, P, R, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

def extract_color_moments(img_rgb):
    moments = []
    for c in cv2.split(img_rgb):
        mean = np.mean(c)
        std = np.std(c)
        skew = np.mean((c - mean)**3) / (std**3 + 1e-7)
        moments.extend([mean, std, skew])
    return np.array(moments, dtype=np.float32)

# -----------------------------
# MORPHOLOGY
# -----------------------------
def extract_morph_features(img_gray):
    _, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return np.array([0, 0, 0], dtype=np.float32)

    cnt = max(contours, key=cv2.contourArea)

    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    circularity = 0 if perimeter == 0 else 4 * np.pi * area / (perimeter**2)

    return np.array([area, perimeter, circularity], dtype=np.float32)

# -----------------------------
# EDGE FEATURES
# -----------------------------
def extract_edge_features(img_gray):
    sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)

    edges = cv2.Canny(img_gray, 50, 150)
    edge_density = cv2.countNonZero(edges) / (img_gray.shape[0] * img_gray.shape[1])

    return np.array([
        sobel_mag.mean(),
        sobel_mag.std(),
        edge_density
    ], dtype=np.float32)

# GAMETE COUNT (VISUAL)
def extract_gamete_count(img_gray, img_color):

    # Step 1: Threshold
    _, thresh = cv2.threshold(
        img_gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Step 2: Noise removal
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Step 3: Sure background
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Step 4: Distance transform
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, 0)

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Step 5: Marker labeling
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Step 6: Watershed
    img_copy = img_color.copy()
    markers = cv2.watershed(img_copy, markers)

    # Step 7: Count objects
    count = len(np.unique(markers)) - 2  # remove background & boundary

    # Visualization
    img_copy[markers == -1] = [0, 0, 255]

    cv2.putText(
        img_copy,
        f"Gamete Count: {count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Watershed Segmentation", resize_for_display(img_copy))

    return np.array([count], dtype=np.float32)

# -----------------------------
# GAMETE AREA
# -----------------------------
def extract_gamete_area(img_gray, img_color):

    h, w = img_color.shape[:2]

    tissue_mask = cv2.threshold(
        img_gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_OPEN, kernel, iterations=2)

    hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
    hsv = cv2.GaussianBlur(hsv, (5,5), 0)

    lower = np.array([0, 50, 50])
    upper = np.array([10, 255, 255])

    mask1 = cv2.inRange(hsv, lower, upper)

    lower2 = np.array([170, 50, 50])
    upper2 = np.array([180, 255, 255])

    mask2 = cv2.inRange(hsv, lower2, upper2)
    gamete_mask = cv2.bitwise_or(mask1, mask2)
    gamete_mask = cv2.bitwise_and(gamete_mask, gamete_mask, mask=tissue_mask)

    # cv2.imshow("Tissue Mask", tissue_mask)
    # cv2.imshow("Gamete Mask (Threshold Result)", gamete_mask)

    overlay = img_color.copy()
    overlay[gamete_mask > 0] = [0, 0, 255]  # highlight gametes in red
    # cv2.imshow("Gamete Overlay", overlay)

    cv2.imshow("Tissue Mask", resize_for_display(tissue_mask, (512, 512)))
    cv2.imshow("Gamete Mask", resize_for_display(gamete_mask, (512, 512)))
    cv2.imshow("Gamete Overlay", resize_for_display(overlay, (512, 512)))

    gamete_pixels = cv2.countNonZero(gamete_mask)
    tissue_pixels = cv2.countNonZero(tissue_mask)

    fraction = gamete_pixels / tissue_pixels if tissue_pixels > 0 else 0

    return np.array([tissue_pixels, gamete_pixels, fraction], dtype=np.float32)

if __name__ == "__main__":

    img_path = r"C:\4th yr 2nd Sem\CMSC 198.2\BCGonadalAnalyzerPrototype\imagedataset\FemaleNormalized\spawning\7-F-Spawning-20x.jpg"

    img = cv2.imread(img_path)
    if img is None:
        print("Image not found!")
        exit()

    print("Loaded:", img_path)

    display_size = (512, 512)

    clean = preprocess_image(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.imshow("Original", resize_for_display(img, display_size))
    cv2.imshow("Clean", resize_for_display(cv2.cvtColor(clean, cv2.COLOR_RGB2BGR), display_size))

    glcm = extract_glcm(gray)
    lbp = extract_lbp(gray)
    color = extract_color_moments(img)
    morph = extract_morph_features(gray)
    edge = extract_edge_features(gray)

    gamete_count = extract_gamete_count(gray, img)
    print(f"Gamete Count: {gamete_count[0]}")
    gamete_area = extract_gamete_area(gray, img)
    print("Gamete Area:", gamete_area)

    full_feature = np.hstack([
        glcm,
        lbp,
        color,
        morph,
        edge,
        gamete_area,
        gamete_count,
    ])

    print("\nFINAL FEATURE VECTOR LENGTH:", len(full_feature))

    cv2.waitKey(0)
    cv2.destroyAllWindows()