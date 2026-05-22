import pickle
import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn
import numpy as np
import cv2
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
from skimage.feature.texture import graycomatrix, graycoprops
from skimage.feature import local_binary_pattern
from utils.normalization import reinhard_normalization
import joblib
import os

# ── App setup 
app = FastAPI(title="BC Gonadal Stage Analyzer API", version="1.0.0")

origins = [
    "http://localhost:3000",
    "https://your-app-name.vercel.app", 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],   # adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_path(relative_path: str):
    return os.path.join(BASE_DIR, relative_path)

device = torch.device("cpu")

# ── Model loading 
try:
    #ML Models
    # MODEL_F = pickle.load(open("best_xgb_model_F.pickle", "rb"))
    MODEL_F = pickle.load(open("best_gb_model_F(Balanced).pickle", "rb"))
    # MODEL_F = pickle.load(open("best_svc_model_F.pickle", "rb"))
    MODEL_M = pickle.load(open("best_xgb_model_M.pickle", "rb"))
    
    #Bouncer Models
    BOUNCER_M = joblib.load("histology_bouncer_male.joblib")
    BOUNCER_F = joblib.load("histology_bouncer_female[Balanced].joblib")
    print("ML Models and Bouncers loaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load ML models: {e}")
    MODEL_F = MODEL_M = BOUNCER_F = BOUNCER_M = None

try:
    #DL Models
    #MALE MODEL: EfficientNet-B0
    DL_MODEL_M = models.efficientnet_b0(weights=None)
    in_features = DL_MODEL_M.classifier[1].in_features
    DL_MODEL_M.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, 4)
    )
    m_path = get_path("dlmodels/male_best_model_efficientnet_b0.pth")
    DL_MODEL_M.load_state_dict(torch.load(m_path, map_location='cpu'))
    DL_MODEL_M.eval()

    #FEMALE MODEL: ResNet-50
    DL_MODEL_F = models.resnet50(weights=None)
    num_features = DL_MODEL_F.fc.in_features
    DL_MODEL_F.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, 4)
    )
    f_path = get_path("dlmodels/female_best_model_resnet50.pth")
    #loading Weights
    DL_MODEL_F.load_state_dict(torch.load(f_path, map_location='cpu'))
    DL_MODEL_F.eval()
    print("DL Models loaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load DL models: {e}")
    DL_MODEL_M = DL_MODEL_F = None

#Transform Pipelines for DL Models
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

# try:
#     with open(MODEL_PATH, "rb") as f:
#         MODEL = pickle.load(f)
#     print(f"[OK] Model loaded from {MODEL_PATH}")
# except FileNotFoundError:
#     print(f"[WARN] Model file '{MODEL_PATH}' not found. /predict will fail until model is present.")
#     MODEL = None

MALE_CATEGORIES   = ["developing", "maturing", "spawning", "spent"]
FEMALE_CATEGORIES = ["developing", "maturing", "spawning", "spent"]

GLCM_NAMES  = ["contrast_mean","contrast_std","homogeneity_mean","homogeneity_std",
                "energy_mean","energy_std","correlation_mean","correlation_std"]
P = 24
LBP_NAMES   = [f"lbp_bin_{i}" for i in range(P + 2)]   # 26 bins
CM_NAMES    = ["R_mean","R_std","R_skew","G_mean","G_std","G_skew","B_mean","B_std","B_skew"]
MORPH_NAMES = ["area_foreground","area_contour","circularity"]
EDGE_NAMES  = ["sobel_mean","sobel_std","edge_density"]
GAMETE_NAMES= ["total_tissue_pixels","gamete_pixels","area_fraction"]
ALL_FEATURE_NAMES = GLCM_NAMES + LBP_NAMES + CM_NAMES + MORPH_NAMES + EDGE_NAMES + GAMETE_NAMES

def predict_dl(img_bgr, sex):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_tensor = transform(img_rgb).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        if sex == "M":
            outputs = DL_MODEL_M(img_tensor)
        if sex == "F":
            outputs = DL_MODEL_F(img_tensor)
        
        probabilities = torch.softmax(outputs, dim=1).cpu()
        predicted_idx = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_idx].item()
    
    return predicted_idx, confidence, probabilities[0].cpu().numpy()

def extract_glcm(img_gray: np.ndarray) -> np.ndarray:
    glcm = graycomatrix(
        img_gray,
        distances=[1, 2, 3],
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=256, symmetric=True, normed=True
    )
    features = []
    for prop in ["contrast", "homogeneity", "energy", "correlation"]:
        vals = graycoprops(glcm, prop).ravel()
        features += [float(np.mean(vals)), float(np.std(vals))]
    return np.array(features, dtype=np.float32)


def extract_lbp(img_gray: np.ndarray, P: int = 24, R: int = 3) -> np.ndarray:
    lbp = local_binary_pattern(img_gray, P, R, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist.astype(np.float32)


def extract_color_moments(img_rgb: np.ndarray) -> np.ndarray:
    moments = []
    for c in cv2.split(img_rgb):
        mean = float(np.mean(c))
        std  = float(np.std(c))
        skew = float(np.mean((c - mean)**3) / (np.std(c)**3 + 1e-7))
        moments += [mean, std, skew]
    return np.array(moments, dtype=np.float32)


def extract_morph_features(img_gray: np.ndarray) -> np.ndarray:
    _, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (thresh == 0).astype(np.uint8)
    area_fg = float(np.sum(mask))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    c = max(contours, key=cv2.contourArea)
    area_c = float(cv2.contourArea(c))
    perim  = float(cv2.arcLength(c, True))
    circ   = 4 * np.pi * area_c / (perim**2) if area_c > 0 and perim > 0 else 0.0
    return np.array([area_fg, area_c, circ], dtype=np.float32)


def extract_edge_features(img_gray: np.ndarray) -> np.ndarray:
    sx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sx**2 + sy**2)
    edges = cv2.Canny(img_gray, 50, 150)
    return np.array([mag.mean(), mag.std(), edges.mean()], dtype=np.float32)


def extract_gamete_area(img_gray: np.ndarray, img_bgr: np.ndarray, sex: str) -> np.ndarray:
    """Sex-specific HSV ranges matching your male feature extraction script."""
    _, tissue_mask = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    total_tissue = int(cv2.countNonZero(tissue_mask))

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    if sex == "M":
        lower = np.array([110, 40, 20])
        upper = np.array([170, 255, 150])
    else:  # F
        lower = np.array([110, 40, 20])
        upper = np.array([170, 255, 200])

    gamete_mask = cv2.inRange(hsv, lower, upper)
    gamete_mask = cv2.morphologyEx(gamete_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    gamete_mask = cv2.bitwise_and(gamete_mask, gamete_mask, mask=tissue_mask)
    gamete_px   = int(cv2.countNonZero(gamete_mask))
    fraction    = gamete_px / total_tissue if total_tissue > 0 else 0.0

    return np.array([float(total_tissue), float(gamete_px), fraction], dtype=np.float32)


def build_feature_vector(img_bgr: np.ndarray, sex: str) -> np.ndarray:
    img_proc = img_bgr
    gray = cv2.cvtColor(img_proc, cv2.COLOR_BGR2GRAY)

    glcm   = extract_glcm(gray)
    lbp    = extract_lbp(gray)
    cm     = extract_color_moments(img_proc)
    morph  = extract_morph_features(gray)
    edge   = extract_edge_features(gray)
    gamete = extract_gamete_area(gray, img_proc, sex)

    return np.hstack([glcm, lbp, cm, morph, edge, gamete])


class FeatureGroup(BaseModel):
    name: str
    features: Dict[str, float]

class PredictionResponse(BaseModel):
    predicted_stage: str
    confidence: float
    probabilities: Dict[str, float]
    feature_groups: List[FeatureGroup]

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_F_loaded": MODEL_F is not None,
        "model_M_loaded": MODEL_M is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    sex: str = Form(...),   # "M" or "F"
    model_choice: str = Form("ML")  # "ML" or "DL"
    ):
    sex = sex.upper()
    if sex in ["MALE", "M"]:
        sex = "M"
    elif sex in ["FEMALE", "F"]:
        sex = "F"
    else:
        raise HTTPException(...)
    
    # Read image
    contents = await file.read()
    nparr    = np.frombuffer(contents, np.uint8)
    img_bgr  = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode image. Send a valid JPG/PNG.")

    # Select reference image
    if sex == "M":
        ref_path = "reference_images/M_reference.jpg"
    else:
        ref_path = "reference_images/F_reference.jpg"

    # Apply Reinhard normalization
    img_norm = reinhard_normalization(img_bgr, ref_path)
    categories = MALE_CATEGORIES if sex == "M" else FEMALE_CATEGORIES

    #DL Path
    if model_choice.upper() == "DL":
        if DL_MODEL_M is None or DL_MODEL_F is None:
            raise HTTPException(status_code=503, detail="DL Models not loaded.")

        pred_idx, confidence, proba = predict_dl(img_norm, sex)
        predicted_stage = categories[pred_idx]
        probabilities = {cat: float(p) for cat, p in zip(categories, proba)} 
        groups = []  
    
    else:

        #ML Path
        MODEL = MODEL_M if sex == "M" else MODEL_F

        if MODEL is None:
            raise HTTPException(status_code=503, detail="Model not loaded.")

        #Bouncer Check for Anomaly Detection
        if sex.upper() in ["M", "MALE"]:
            CURRENT_BOUNCER = BOUNCER_M
            CURRENT_MODEL = MODEL_M
        else:
            CURRENT_BOUNCER = BOUNCER_F
            CURRENT_MODEL = MODEL_F

        # Feature extraction
        try:
            fv = build_feature_vector(img_norm, sex)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Feature extraction failed: {e}")

        #ML Prediction
        fv_2d = fv.reshape(1, -1)
        # Check Anomalies with Bouncer
        anomaly_score = CURRENT_BOUNCER.decision_function(fv_2d)[0]
        print(f"Anomaly Score: {anomaly_score:.4f}")
        
        # if CURRENT_BOUNCER.predict(fv_2d)[0] == -1:
        #     raise HTTPException(status_code=400, detail="Invalid histology image for the selected sex. Please check quality and try again.")
        # if CURRENT_BOUNCER == BOUNCER_M and anomaly_score < 0.065:
            # raise HTTPException(status_code=400, detail="Invalid histology image for the selected sex. Please check quality and try again.")
        if CURRENT_BOUNCER == BOUNCER_F and anomaly_score < 0.05:
            pass   
        # if anomaly_score < 0.065:
        
        # Prediction
        pred_idx   = int(MODEL.predict(fv_2d)[0])
        proba      = MODEL.predict_proba(fv_2d)[0]

        categories = MALE_CATEGORIES if sex == "M" else FEMALE_CATEGORIES
        predicted_stage = categories[pred_idx]
        confidence      = float(proba[pred_idx])
        probabilities   = {cat: float(p) for cat, p in zip(categories, proba)}

        # Build feature groups for the frontend
        fv_list = fv.tolist()
        idx = 0
        groups: List[FeatureGroup] = []

        def take(names):
            nonlocal idx
            d = {n: round(fv_list[idx + i], 6) for i, n in enumerate(names)}
            idx += len(names)
            return d

        groups.append(FeatureGroup(name="GLCM (Texture)",      features=take(GLCM_NAMES)))
        groups.append(FeatureGroup(name="LBP (Local Pattern)", features=take(LBP_NAMES)))
        groups.append(FeatureGroup(name="Color Moments",       features=take(CM_NAMES)))
        groups.append(FeatureGroup(name="Morphology",          features=take(MORPH_NAMES)))
        groups.append(FeatureGroup(name="Edge (Sobel/Canny)",  features=take(EDGE_NAMES)))
        groups.append(FeatureGroup(name="Gamete Area",         features=take(GAMETE_NAMES)))

    return PredictionResponse(
        predicted_stage=predicted_stage,
        confidence=confidence,
        probabilities=probabilities,
        feature_groups=groups,
    )