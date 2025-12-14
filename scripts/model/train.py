from collections import Counter
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from src.training.transformations import train_transform, base_transform
from src.voice_cnn import VoiceCNN


DATA_ROOT = Path("data/processed") 
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3
IMG_SIZE = 128

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data = datasets.ImageFolder(DATA_ROOT / "train", transform=train_transform)
val_data   = datasets.ImageFolder(DATA_ROOT / "val",   transform=base_transform)
test_data  = datasets.ImageFolder(DATA_ROOT / "test",  transform=base_transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_data, batch_size=32)
test_loader  = DataLoader(test_data, batch_size=32)

def count_classes(dataset):
    # dataset.targets contains class indices (0 = accept, 1 = reject)
    counts = Counter(dataset.targets)
    class_names = dataset.classes  # ["accept", "reject"]
    return {class_names[i]: counts[i] for i in counts}

print("Train:", count_classes(train_data))
print("Val:",   count_classes(val_data))
print("Test:",  count_classes(test_data))

model = VoiceCNN().to(DEVICE)
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

def run_epoch(model, loader, train: bool):
    model.train() if train else model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            if train:
                optimizer.zero_grad()

            out = model(x)
            loss = criterion(out, y)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * x.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    return total_loss / total, correct / total

# =========================
# TRAINING
# =========================

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = run_epoch(model, train_loader, train=True)
    val_loss, val_acc     = run_epoch(model, val_loader,   train=False)

    print(
        f"Epoch {epoch:02d} | "
        f"Train loss {train_loss:.4f}, acc {train_acc:.3f} | "
        f"Val loss {val_loss:.4f}, acc {val_acc:.3f}"
    )

# =========================
# TEST
# =========================

test_loss, test_acc = run_epoch(model, test_loader, train=False)
print(f"TEST accuracy: {test_acc * 100:.2f}%")

# =========================
# SAVE MODEL
# =========================

torch.save(model.state_dict(), "voice_cnn.pth")
print("Model saved to voice_cnn.pth")
