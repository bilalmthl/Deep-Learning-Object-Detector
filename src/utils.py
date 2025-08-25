import os
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms as T

# --- Class reader ---
def read_classes(path):
    if path and os.path.exists(path):
        with open(path, "r") as f:
            return [ln.strip() for ln in f if ln.strip()]
    return ["apples", "bananas", "oranges", "none"]

# --- Transform ---
def make_transform():
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

# --- Model builder ---
def build_model(num_classes):
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

def load_model(model_path, num_classes, device):
    m = build_model(num_classes)
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        m.load_state_dict(state["state_dict"])
    else:
        m.load_state_dict(state)
    m.to(device).eval()
    return m
