from pathlib import Path
from typing import List
import torch
from PIL import Image
from torchvision import transforms
from .models import build_model, predict_topk

class Predictor:
    def __init__(self, checkpoint_path: str):
        payload = torch.load(checkpoint_path, map_location="cpu")
        self.classes: List[str] = payload["classes"]
        self.model = build_model(payload["model_name"], len(self.classes))
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        self.tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

    def predict(self, image_path: str, k: int = 5):
        img = Image.open(image_path).convert("RGB")
        x = self.tf(img).unsqueeze(0)
        conf, idx = predict_topk(self.model, x, k=k)
        return [(self.classes[i], float(c)) for c, i in zip(conf[0], idx[0])]
