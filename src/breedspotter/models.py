import torch
import torch.nn as nn
import timm

def build_model(model_name: str, num_classes: int) -> nn.Module:
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    return model

@torch.inference_mode()
def predict_topk(model, tensor, k=5):
    model.eval()
    logits = model(tensor)
    probs = logits.softmax(dim=1)
    conf, idx = probs.topk(k, dim=1)
    return conf.cpu().numpy(), idx.cpu().numpy()
