import argparse
import yaml
import sys
from collections import Counter
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from src.training.choose_optimizer import choose_optimizer
from src.training.initialize_weights import initialize_weights
from src.training.transformations import train_transform, base_transform
from src.training.VoiceCNN import VoiceCNN
import csv

EXPERIMENTS_FILE = Path("outputs/experiments/results.csv")
EXPERIMENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
EPOCHS_CSV = Path("outputs/experiments/epochs.csv")
EPOCHS_CSV.parent.mkdir(parents=True, exist_ok=True)


def log_experiment(row: dict):
    write_header = not EXPERIMENTS_FILE.exists()

    with EXPERIMENTS_FILE.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)


DATA_ROOT = Path("data/processed_aug") 
BATCH_SIZE = 32
EPOCHS = 10
IMG_SIZE = 128

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path, help="Config filename (relative to src/config/ directory)")
    return parser.parse_args()

def count_classes(dataset):
    # dataset.targets contains class indices (0 = accept, 1 = reject)
    counts = Counter(dataset.targets)
    class_names = dataset.classes  # ["accept", "reject"]
    return {class_names[i]: counts[i] for i in counts}

from pathlib import Path

def create_model_name(config_path: Path, cfg: dict, init_weights_strategy: str) -> str:
    lr = cfg["training"]["learning_rate"]
    dropout = cfg["model"]["dropout"]
    batch_norm_mode = cfg["model"]["batch_norm_mode"]
    activation = cfg["model"]["activation"]

    lr_str = f"{lr:.0e}" if lr < 1e-2 else f"{lr}".replace(".", "x")
    dropout_str = f"{dropout}".replace(".", "x")
    batch_norm_str = f"{batch_norm_mode}{"activation" if batch_norm_mode != "none" else ""}"

    return f"{config_path.stem}_lr_{lr_str}_dropout_{dropout_str}_batchnorm_{batch_norm_str}_{activation}_{init_weights_strategy}_weights.pth"

def main():
    write_header = not EPOCHS_CSV.exists()
    epoch_file = EPOCHS_CSV.open("a", newline="")

    epoch_writer = csv.DictWriter(
        epoch_file,
        fieldnames=[
            "model_name",
            "epoch",
            "train_loss",
            "train_accuracy",
            "val_loss",
            "val_accuracy",
        ],
    )

    if write_header:
        epoch_writer.writeheader()

    args = parse_args()
    config_path = Path("src/config") / args.config

    with config_path.open() as f:
        cfg = yaml.safe_load(f)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data = datasets.ImageFolder(DATA_ROOT / "train", transform=train_transform)
    val_data   = datasets.ImageFolder(DATA_ROOT / "val",   transform=base_transform)
    test_data  = datasets.ImageFolder(DATA_ROOT / "test",  transform=base_transform)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_data, batch_size=32)
    test_loader  = DataLoader(test_data, batch_size=32)

    print("Train:", count_classes(train_data))
    print("Val:",   count_classes(val_data))
    print("Test:",  count_classes(test_data))

    # =========================
    # APPLYING MODIFICATIONS TO THE BASE MODEL
    # =========================

    model = VoiceCNN(
        dropout_rate=cfg["model"]["dropout"], 
        batch_norm_mode=cfg["model"]["batch_norm_mode"],
        activation=cfg["model"]["activation"]
    ).to(DEVICE)

    init_weights_strategy = initialize_weights(
        model, 
        cfg["model"]["activation"],
        init_type=cfg["model"]["initialize_weights"]
    )
    print(model)
    print(f"Weights were initializeed using {init_weights_strategy} strategy...")

    criterion = nn.CrossEntropyLoss()
    optimizer = choose_optimizer(model, cfg)
    model_name = create_model_name(config_path, cfg, init_weights_strategy)

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

    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_epoch = -1

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, train=True)
        val_loss, val_acc     = run_epoch(model, val_loader,   train=False)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch

        print(
            f"Epoch {epoch:02d} | "
            f"Train loss {train_loss:.4f}, acc {train_acc:.3f} | "
            f"Val loss {val_loss:.4f}, acc {val_acc:.3f}"
        )
        epoch_writer.writerow({
            "model_name": model_name,
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_accuracy": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_accuracy": round(val_acc, 4),
        })


    # =========================
    # TEST
    # =========================

    test_loss, test_acc = run_epoch(model, test_loader, train=False)
    print(f"TEST accuracy: {test_acc * 100:.2f}%")

    # =========================
    # SAVE MODEL
    # =========================

    OUTPUT_DIR = Path("outputs/models")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = OUTPUT_DIR / model_name
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    log_experiment({
        "config": config_path.stem,
        "optimizer": cfg["optimizer"]["type"],
        "learning_rate": cfg["training"]["learning_rate"],
        "activation": cfg["model"]["activation"],
        "batch_norm": cfg["model"]["batch_norm_mode"],
        "dropout": cfg["model"]["dropout"],
        "weight_init": init_weights_strategy,
        "epochs": EPOCHS,
        "best_val_accuracy": round(best_val_acc, 4),
        "best_val_loss": round(best_val_loss, 4),
        "test_accuracy": round(test_acc, 4),
        "best_epoch": best_epoch,
    })

    epoch_file.close()


if __name__ == "__main__":
    main()