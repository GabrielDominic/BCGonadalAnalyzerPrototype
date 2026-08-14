import torch
import torch.nn as nn
from torchvision import models
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_path(relative_path: str):
    return os.path.join(BASE_DIR, relative_path)

# MALE DL model Architecture and loading weights
DL_MODEL_M = models.efficientnet_b0(weights=None)
in_features = DL_MODEL_M.classifier[1].in_features
DL_MODEL_M.classifier = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(in_features, 256),
    nn.ReLU(),
    nn.Dropout(p=0.2),
    nn.Linear(256, 4)
)
EN_m_path = get_path("dlmodels/male_best_model_efficientnet_b0.pth")
DL_MODEL_M.load_state_dict(torch.load(EN_m_path, map_location='cpu'))
DL_MODEL_M.eval()

dummy_input_m = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    DL_MODEL_M,
    dummy_input_m,
    "male_model.onnx",
    export_params=True,
    opset_version=18,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
)

#Female DL model Architecture and loading weights
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

dummy_input_f = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    DL_MODEL_F,
    dummy_input_f,
    "female_model.onnx",
    export_params=True,
    opset_version=18,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
)
print("ONNX models have been successfully created and saved as 'male_model.onnx' and 'female_model.onnx'.")