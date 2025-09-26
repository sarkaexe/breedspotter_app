from pathlib import Path
import torch
from torch import nn, optim
from tqdm import tqdm
from .config import TrainConfig
from .data import build_loaders
from .models import build_model

def train():
    cfg = TrainConfig()
    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader, classes = build_loaders(cfg.data_dir, cfg.image_size, cfg.batch_size)
    model = build_model(cfg.model_name, len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr)

    best_val = 0.0
    for epoch in range(cfg.epochs):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)
        train_acc = correct / total
        train_loss = loss_sum / total

        # quick val
        model.eval()
        v_total, v_correct = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                v_correct += (out.argmax(1) == y).sum().item()
                v_total += x.size(0)
        val_acc = v_correct / v_total if v_total else train_acc

        print(f"[{epoch+1}] loss={train_loss:.4f} acc={train_acc:.3f} val_acc={val_acc:.3f}")

        if val_acc > best_val:
            best_val = val_acc
            ckpt = Path(cfg.checkpoint_dir) / "best.pt"
            torch.save({"model_name": cfg.model_name,
                        "classes": classes,
                        "state_dict": model.state_dict()}, ckpt)
            print(f"Saved: {ckpt}")

if __name__ == "__main__":
    train()
