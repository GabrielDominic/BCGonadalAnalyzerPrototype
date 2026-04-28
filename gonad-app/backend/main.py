from fastapi import FastAPI, File, UploadFile, Form
import numpy as np
import cv2
import io
from PIL import Image

from utils.normalization import reinhard_normalization

app = FastAPI()

@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    sex: str = Form(...)
):
    # Read uploaded image
    contents = await image.read()
    pil_img = Image.open(io.BytesIO(contents)).convert("RGB")

    # Convert to OpenCV format
    img_np = np.array(pil_img)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Select reference image
    if sex == "male":
        ref_path = "reference_images/M_reference.png"
    else:
        ref_path = "reference_images/F_reference.png"

    # Apply Reinhard normalization
    normalized_img = reinhard_normalization(img_bgr, ref_path)

    cv2.imwrite("debug_normalized.jpg", normalized_img)

    prediction = "Mature"
    confidence = 0.93

    return {
        "stage": prediction,
        "confidence": confidence,
        "sex": sex
    }