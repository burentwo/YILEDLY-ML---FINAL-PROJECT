import os
import copy
import time
import json
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import timm

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# CONFIGURATION

# AFTER (Colab paths)
DATASET_DIRS = [
    "/content/DaTalongSet/DaTalongSet/Talong_DataSets/Original Dataset",
    "/content/DaTalongSet/DaTalongSet/Talong_DataSets/Augmented Dataset",
]
OUTPUT_PATH = "/content/drive/MyDrive/yieldy_model.pth"
CONFUSION_MATRIX_PATH = "/content/drive/MyDrive/confusion_matrix.png"
NUM_CLASSES         = 7

IMG_SIZE            = 260
BATCH_SIZE          = 16
NUM_EPOCHS          = 25
LR                  = 1e-4    
LR_BACKBONE         = 1e-5   
VAL_SPLIT           = 0.15
SEED                = 42
NUM_WORKERS         = 2      

CLASS_NAMES = [
    "Eggplant Healthy Fruit",
    "Eggplant Phomopsis Blight",
    "Eggplant Shoot and Fruit Borer",
    "Eggplant Healthy Leaf",
    "Eggplant Insect Pest Disease",
    "Eggplant Leaf Spot Disease",
    "Eggplant Wilt Disease",
]

CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# CUSTOM DATASET

class EggplantDataset(Dataset):
    """
    Loads images from one or more subfolder-per-class directories.
    Accepts a single path string or a list of paths.
    Flat-file fallback (_NNN suffix stripping) is used when no matching
    subfolders are found inside a given directory.
    """

    def __init__(self, directories, class_to_idx: dict, transform=None):
        self.transform    = transform
        self.class_to_idx = class_to_idx
        self.samples      = []   
        self.skipped      = []

        if isinstance(directories, (str, Path)):
            directories = [directories]

        for directory in directories:
            root        = Path(directory)
            dir_samples = []

            # Subfolder Layout
            for subfolder in sorted(root.iterdir()):
                if not subfolder.is_dir():
                    continue
                label = subfolder.name.strip()
                if label not in class_to_idx:
                    continue
                class_idx = class_to_idx[label]
                for fpath in sorted(subfolder.iterdir()):
                    if fpath.suffix.lower() not in SUPPORTED_EXTENSIONS:
                        continue
                    dir_samples.append((str(fpath), class_idx))

            # Flat-file fallback
            if not dir_samples:
                for fpath in sorted(root.iterdir()):
                    if fpath.suffix.lower() not in SUPPORTED_EXTENSIONS:
                        continue
                    label = self._parse_label(fpath.stem)
                    if label is None or label not in class_to_idx:
                        self.skipped.append(fpath.name)
                        continue
                    dir_samples.append((str(fpath), class_to_idx[label]))

            self.samples.extend(dir_samples)
            print(f"   📁 {root.name:<25} → {len(dir_samples):>5} images loaded")

    @staticmethod
    def _parse_label(stem: str):
        idx = stem.rfind("_")
        return stem[:idx].strip() if idx != -1 else None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        fpath, label = self.samples[index]
        image = Image.open(fpath).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# TRANSFORMS

train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# DATA LOADING

def make_loaders(dataset_dirs):
    print("\n📂 Scanning source folders:")
    train_dataset = EggplantDataset(dataset_dirs, CLASS_TO_IDX, transform=train_transforms)
    print()
    val_dataset   = EggplantDataset(dataset_dirs, CLASS_TO_IDX, transform=val_transforms)

    if train_dataset.skipped:
        print(f"\n⚠️  Skipped {len(train_dataset.skipped)} unrecognised file(s):")
        for name in train_dataset.skipped[:10]:
            print(f"     {name}")
        if len(train_dataset.skipped) > 10:
            print(f"     … and {len(train_dataset.skipped) - 10} more.")

    n_total = len(train_dataset)
    if n_total == 0:
        raise RuntimeError(
            "No images were loaded.\n"
            "  1. Confirm all paths in DATASET_DIRS point to folders with class subfolders.\n"
            "  2. Confirm subfolder names match CLASS_NAMES exactly.\n"
            "  3. Confirm image files are .jpg / .jpeg / .png / .bmp / .webp"
        )

    counts = Counter(label for _, label in train_dataset.samples)
    print("\n📂 Images found per class:")
    for name, idx in CLASS_TO_IDX.items():
        print(f"   [{idx}] {name:<42} {counts.get(idx, 0):>5} images")

    n_val   = max(1, int(n_total * VAL_SPLIT))
    n_train = n_total - n_val
    generator = torch.Generator().manual_seed(SEED)
    shuffled  = torch.randperm(n_total, generator=generator).tolist()
    train_idx, val_idx = shuffled[:n_train], shuffled[n_train:]

    # Weighted sampler (oversamples minority classes in training)
    train_labels   = [train_dataset.samples[i][1] for i in train_idx]
    train_counts   = Counter(train_labels)
    sample_weights = [
        1.0 / train_counts[train_dataset.samples[i][1]]
        for i in train_idx
    ]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_idx),
        replacement=True,
    )
    print(f"\n  Weighted sampler active — minority classes will be upsampled.")

    # Class weights for weighted loss
    class_weights = torch.zeros(NUM_CLASSES)
    for class_idx, cnt in train_counts.items():
        class_weights[class_idx] = n_train / (NUM_CLASSES * cnt)
    print(f"  Loss weights per class:")
    for name, idx in CLASS_TO_IDX.items():
        print(f"     [{idx}] {name:<42} weight={class_weights[idx]:.4f}")

    train_loader = DataLoader(
        Subset(train_dataset, train_idx),
        batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    val_loader = DataLoader(
        Subset(val_dataset, val_idx),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    print(f"\n   Train: {n_train}  |  Val: {n_val}  |  Total: {n_total}\n")
    return train_loader, val_loader, class_weights


# MODEL

def build_model(device: torch.device):
    model = timm.create_model("efficientnet_b2", pretrained=True, num_classes=NUM_CLASSES)
    return model.to(device)


# TRAINING LOOP

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = total_correct = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        out  = model(images)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss    += loss.item() * images.size(0)
        total_correct += (out.argmax(1) == labels).sum().item()
    n = len(loader.dataset)
    return total_loss / n, total_correct / n


# EVALUATION LOOP  (Priority 4 — Advanced Metrics)

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """
    Evaluate model on the validation set.

    Returns
    -------
    val_loss    : float
    val_acc     : float
    all_preds   : list[int]   — predicted class indices for each sample
    all_labels  : list[int]   — ground-truth class indices for each sample

    The caller uses all_preds and all_labels to compute per-class
    Precision, Recall, and F1-score via scikit-learn.
    """
    model.eval()
    total_loss    = 0.0
    total_correct = 0
    all_preds     = []
    all_labels    = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        out = model(images)
        total_loss    += criterion(out, labels).item() * images.size(0)
        preds          = out.argmax(dim=1)
        total_correct += (preds == labels).sum().item()

        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    n = len(loader.dataset)
    return total_loss / n, total_correct / n, all_preds, all_labels


def log_per_class_metrics(all_preds, all_labels, epoch: int):
    """
    Compute and print per-class Precision, Recall, and F1 to stdout.
    Also returns the weighted-average F1 for use in best-model tracking.
    """
    # Weighted averages (handle zero-division for classes not seen in val)
    precision_w = precision_score(all_labels, all_preds, average="weighted",
                                   zero_division=0)
    recall_w    = recall_score(all_labels, all_preds, average="weighted",
                                zero_division=0)
    f1_w        = f1_score(all_labels, all_preds, average="weighted",
                            zero_division=0)

    # Per-class scores
    precision_pc = precision_score(all_labels, all_preds, average=None,
                                    labels=list(range(NUM_CLASSES)), zero_division=0)
    recall_pc    = recall_score(all_labels, all_preds, average=None,
                                 labels=list(range(NUM_CLASSES)), zero_division=0)
    f1_pc        = f1_score(all_labels, all_preds, average=None,
                             labels=list(range(NUM_CLASSES)), zero_division=0)

    # Support (count of true samples per class in val set)
    label_counts = Counter(all_labels)

    col_w = 44
    print(f"\n  ── Per-class metrics (Epoch {epoch}) ───────────────────────────────────")
    header = f"  {'Class':<{col_w}} {'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'Supp':>5}"
    print(header)
    print("  " + "─" * (col_w + 30))

    for idx, name in enumerate(CLASS_NAMES):
        print(
            f"  {name:<{col_w}} "
            f"{precision_pc[idx]:>6.3f}  "
            f"{recall_pc[idx]:>6.3f}  "
            f"{f1_pc[idx]:>6.3f}  "
            f"{label_counts.get(idx, 0):>5}"
        )

    print("  " + "─" * (col_w + 30))
    print(
        f"  {'Weighted avg':<{col_w}} "
        f"{precision_w:>6.3f}  {recall_w:>6.3f}  {f1_w:>6.3f}"
    )
    print()

    return f1_w


# CONFUSION MATRIX  (Priority 4)

def save_confusion_matrix(all_preds, all_labels, output_path: str):
    """
    Generate a confusion-matrix heatmap and save it as a PNG.

    Uses seaborn when available; falls back to plain matplotlib otherwise.
    Short class aliases are used on the axes to keep the figure readable.
    """
    import matplotlib.pyplot as plt

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_CLASSES)))

    # Short aliases for axis labels (keep figure readable)
    short_labels = [
        "H. Fruit", "Phomopsis", "S&F Borer",
        "H. Leaf", "Insect Pest", "Leaf Spot", "Wilt",
    ]

    fig, ax = plt.subplots(figsize=(9, 7))

    try:
        import seaborn as sns
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=short_labels,
            yticklabels=short_labels,
            ax=ax,
            linewidths=0.4,
            linecolor="#cccccc",
        )
    except ImportError:
        # Matplotlib fallback
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        fig.colorbar(im, ax=ax)
        ax.set(
            xticks=range(NUM_CLASSES),
            yticks=range(NUM_CLASSES),
            xticklabels=short_labels,
            yticklabels=short_labels,
        )
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                ax.text(j, i, str(cm[i, j]),
                        ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black",
                        fontsize=9)

    ax.set_xlabel("Predicted Label", fontsize=11, labelpad=10)
    ax.set_ylabel("True Label", fontsize=11, labelpad=10)
    ax.set_title("Confusion Matrix — EfficientNet-B2 (Validation Set)", fontsize=13, pad=14)
    plt.xticks(rotation=35, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"   Confusion matrix saved → {output_path}")


# MAIN

def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Device : {device}")
    if device.type == "cpu":
        print("   (No GPU detected — training on CPU will be slower)\n")

    for path in DATASET_DIRS:
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"Dataset directory not found:\n  {path}\n"
                "Check that the path is correct and the folder exists."
            )

    train_loader, val_loader, class_weights = make_loaders(DATASET_DIRS)
    model     = build_model(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=0.1,
    )

    backbone_params   = [p for n, p in model.named_parameters() if "classifier" not in n]
    classifier_params = [p for n, p in model.named_parameters() if "classifier"     in n]
    optimizer = torch.optim.AdamW([
        {"params": backbone_params,   "lr": LR_BACKBONE},
        {"params": classifier_params, "lr": LR},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    best_val_acc   = 0.0
    best_val_f1    = 0.0
    best_state     = None
    final_preds    = []   
    final_labels   = []

    print("=" * 72)
    print(f"  EfficientNet-B2  ·  {NUM_CLASSES} classes  ·  {NUM_EPOCHS} epochs")
    print("=" * 72)

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, all_preds, all_labels = evaluate(
            model, val_loader, criterion, device
        )
        scheduler.step()

        log_metrics_this_epoch = (epoch % 5 == 0) or (epoch == NUM_EPOCHS)
        if log_metrics_this_epoch:
            f1_w = log_per_class_metrics(all_preds, all_labels, epoch)
        else:
            f1_w = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

        flag = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_f1  = f1_w
            best_state   = copy.deepcopy(model.state_dict())
            final_preds  = all_preds
            final_labels = all_labels
            flag = "  ← best ✅"

        print(
            f"Epoch {epoch:>2}/{NUM_EPOCHS} | "
            f"train loss {train_loss:.4f}  acc {train_acc*100:.1f}% | "
            f"val loss {val_loss:.4f}  acc {val_acc*100:.1f}%  "
            f"wF1 {f1_w:.3f} | "
            f"{time.time() - t0:.0f}s{flag}"
        )

    torch.save(best_state, OUTPUT_PATH)
    print(f"\n✅  Best model saved  → {OUTPUT_PATH}")
    print(f"    Best val accuracy  : {best_val_acc * 100:.2f}%")
    print(f"    Best weighted F1   : {best_val_f1:.4f}")

    print("\n── Full classification report (best epoch) ──────────────────────────────")
    print(classification_report(
        final_labels, final_preds,
        target_names=CLASS_NAMES,
        zero_division=0,
        digits=4,
    ))

    save_confusion_matrix(final_preds, final_labels, CONFUSION_MATRIX_PATH)

    map_path = OUTPUT_PATH.replace(".pth", "_class_map.json")
    with open(map_path, "w") as f:
        json.dump({
            "class_to_idx": CLASS_TO_IDX,
            "idx_to_class": {str(v): k for k, v in CLASS_TO_IDX.items()},
        }, f, indent=2)
    print(f"   Class map saved    → {map_path}")
    print("\n▶  Place yieldy_model.pth next to app.py, then run:  streamlit run app.py\n")


if __name__ == "__main__":
    main()