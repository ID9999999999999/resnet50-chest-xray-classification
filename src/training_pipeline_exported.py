# %%
# Check GPU
# SKIPPED_COLAB_COMMAND: !nvidia-smi


# %%
from google.colab import drive
drive.mount('/content/drive')


# %%
# SKIPPED_COLAB_COMMAND: !pip install -q kagglehub scikit-learn matplotlib


# %%
from pathlib import Path

OUTPUT_DIR = Path('/content/drive/MyDrive/pneumonia_resnet50_outputs')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print('Output directory:', OUTPUT_DIR)
print('Existing files:')
for p in sorted(OUTPUT_DIR.glob('*')):
    print(' -', p.name)


# %%
script_code = r'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Resumable pretrained CNN training pipeline for Chest X-Ray Pneumonia classification.

This version saves a checkpoint after every epoch.
If Kaggle / Colab / the browser disconnects, you do NOT restart from zero.

Typical command:

python -u train_resnet_pneumonia_resumable.py \
  --dataset-root /kaggle/input/chest-xray-pneumonia/chest_xray \
  --model resnet50 \
  --epochs-head 5 \
  --epochs-finetune 15 \
  --batch-size 32 \
  --output-dir outputs_resnet50_pneumonia \
  --auto-resume \
  2>&1 | tee -a outputs_resnet50_pneumonia/training_log.txt

If it stops, run the exact same command again.
The script will load outputs_resnet50_pneumonia/checkpoint_last.pth and continue.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, models, transforms

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def find_chest_xray_root(start_path: Path) -> Path:
    start_path = Path(start_path)
    if (start_path / "train").is_dir() and (start_path / "test").is_dir():
        return start_path
    for root, dirs, _files in os.walk(start_path):
        root_path = Path(root)
        if "train" in dirs and "test" in dirs:
            train_path = root_path / "train"
            if (train_path / "NORMAL").is_dir() and (train_path / "PNEUMONIA").is_dir():
                return root_path
    raise FileNotFoundError(
        f"Could not find chest_xray root under: {start_path}. "
        "Expected train/NORMAL, train/PNEUMONIA, test/NORMAL, test/PNEUMONIA."
    )


def get_dataset_root(dataset_root: Optional[str]) -> Path:
    if dataset_root:
        return find_chest_xray_root(Path(dataset_root))
    try:
        import kagglehub
        downloaded = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
        return find_chest_xray_root(Path(downloaded))
    except Exception as exc:
        raise RuntimeError(
            "dataset_root was not provided and KaggleHub automatic download failed. "
            "Please pass --dataset-root manually."
        ) from exc


@dataclass
class SplitInfo:
    train_size: int
    val_size: int
    test_size: int
    class_to_idx: Dict[str, int]
    train_counts: Dict[str, int]
    val_counts: Dict[str, int]
    test_counts: Dict[str, int]


def build_transforms(img_size: int = 224):
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomApply([transforms.RandomRotation(degrees=7)], p=0.70),
        transforms.RandomApply([
            transforms.RandomAffine(degrees=0, translate=(0.03, 0.03), scale=(0.95, 1.05))
        ], p=0.50),
        transforms.RandomHorizontalFlip(p=0.50),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])
    return train_transform, eval_transform


def count_by_class(targets: List[int], class_names: List[str]) -> Dict[str, int]:
    counts = {name: 0 for name in class_names}
    for y in targets:
        counts[class_names[int(y)]] += 1
    return counts


def make_loaders(root: Path, img_size: int, batch_size: int, num_workers: int, val_fraction: float, seed: int, balance_mode: str):
    train_transform, eval_transform = build_transforms(img_size)
    full_train_aug = datasets.ImageFolder(root / "train", transform=train_transform)
    full_train_eval = datasets.ImageFolder(root / "train", transform=eval_transform)
    test_dataset = datasets.ImageFolder(root / "test", transform=eval_transform)
    class_names = full_train_aug.classes
    targets = np.array(full_train_aug.targets)
    train_idx, val_idx = train_test_split(
        np.arange(len(targets)), test_size=val_fraction, random_state=seed, stratify=targets
    )
    train_subset = Subset(full_train_aug, train_idx.tolist())
    val_subset = Subset(full_train_eval, val_idx.tolist())
    train_targets = targets[train_idx]
    class_counts = np.bincount(train_targets, minlength=len(class_names))
    total = class_counts.sum()
    class_weights = total / (len(class_names) * np.maximum(class_counts, 1))
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    if balance_mode == "sampler":
        sample_weights = np.array([1.0 / class_counts[y] for y in train_targets], dtype=np.float64)
        sampler = WeightedRandomSampler(weights=torch.from_numpy(sample_weights), num_samples=len(sample_weights), replacement=True)
        shuffle = False
    else:
        sampler = None
        shuffle = True
    pin_memory = torch.cuda.is_available()
    persistent_workers = num_workers > 0
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=shuffle, sampler=sampler, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)
    split_info = SplitInfo(
        train_size=len(train_subset),
        val_size=len(val_subset),
        test_size=len(test_dataset),
        class_to_idx=full_train_aug.class_to_idx,
        train_counts=count_by_class(train_targets.tolist(), class_names),
        val_counts=count_by_class(targets[val_idx].tolist(), class_names),
        test_counts=count_by_class(test_dataset.targets, class_names),
    )
    return train_loader, val_loader, test_loader, class_names, split_info, class_weights_tensor


def build_model(model_name: str, num_classes: int, dropout: float) -> nn.Module:
    model_name = model_name.lower().strip()
    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
        return model
    if model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
        return model
    if model_name == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
        return model
    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
        return model
    raise ValueError(f"Unknown model: {model_name}")


def freeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_classifier(model: nn.Module, model_name: str) -> None:
    if model_name.startswith("resnet"):
        for p in model.fc.parameters():
            p.requires_grad = True
    elif model_name == "densenet121":
        for p in model.classifier.parameters():
            p.requires_grad = True
    elif model_name == "efficientnet_b0":
        for p in model.classifier.parameters():
            p.requires_grad = True
    else:
        raise ValueError(model_name)


def unfreeze_last_block_and_classifier(model: nn.Module, model_name: str) -> None:
    freeze_all(model)
    if model_name.startswith("resnet"):
        for p in model.layer4.parameters():
            p.requires_grad = True
        for p in model.fc.parameters():
            p.requires_grad = True
    elif model_name == "densenet121":
        for p in model.features.denseblock4.parameters():
            p.requires_grad = True
        for p in model.features.norm5.parameters():
            p.requires_grad = True
        for p in model.classifier.parameters():
            p.requires_grad = True
    elif model_name == "efficientnet_b0":
        for block in model.features[-2:]:
            for p in block.parameters():
                p.requires_grad = True
        for p in model.classifier.parameters():
            p.requires_grad = True
    else:
        raise ValueError(model_name)


def set_trainable_for_stage(model: nn.Module, model_name: str, stage_name: str) -> None:
    if stage_name == "head":
        freeze_all(model)
        unfreeze_classifier(model, model_name)
    elif stage_name == "finetune":
        unfreeze_last_block_and_classifier(model, model_name)
    else:
        raise ValueError(stage_name)


def build_optimizer(model: nn.Module, lr: float, weight_decay: float) -> optim.Optimizer:
    return optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)


def count_trainable_params(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


@dataclass
class MetricResult:
    loss: float
    accuracy: float
    precision_pneumonia: float
    sensitivity_pneumonia: float
    specificity_normal: float
    f1_pneumonia: float
    macro_f1: float
    roc_auc: float
    threshold: float
    tn: int
    fp: int
    fn: int
    tp: int


def calculate_metrics(y_true: np.ndarray, y_prob_pos: np.ndarray, threshold: float, loss: float) -> MetricResult:
    y_pred = (y_prob_pos >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    try:
        auc = roc_auc_score(y_true, y_prob_pos)
    except ValueError:
        auc = float("nan")
    return MetricResult(
        loss=float(loss),
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision_pneumonia=float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        sensitivity_pneumonia=float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        specificity_normal=float(tn / (tn + fp) if (tn + fp) else 0.0),
        f1_pneumonia=float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        roc_auc=float(auc),
        threshold=float(threshold),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
    )


def choose_threshold(y_true: np.ndarray, y_prob_pos: np.ndarray, mode: str = "youden") -> float:
    if mode == "fixed_05":
        return 0.5
    if mode == "youden":
        fpr, tpr, thresholds = roc_curve(y_true, y_prob_pos)
        scores = tpr - fpr
        threshold = float(thresholds[int(np.argmax(scores))])
        if math.isinf(threshold) or math.isnan(threshold):
            return 0.5
        return max(0.01, min(0.99, threshold))
    if mode == "macro_f1":
        best_t, best_score = 0.5, -1.0
        for t in np.linspace(0.05, 0.95, 181):
            score = f1_score(y_true, (y_prob_pos >= t).astype(int), average="macro", zero_division=0)
            if score > best_score:
                best_score, best_t = score, float(t)
        return best_t
    raise ValueError(mode)


def save_json(path: Path, data) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def write_history_csv(history_path: Path, history: List[Dict[str, object]]) -> None:
    if not history:
        return
    keys = list(history[0].keys())
    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)


def atomic_torch_save(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def save_training_checkpoint(output_dir, model, optimizer, scheduler, scaler, args, class_names, stage_name, completed_epochs_by_stage, best_score, best_threshold, history, img_size):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "args": vars(args),
        "model_name": args.model,
        "class_names": class_names,
        "stage_name": stage_name,
        "completed_epochs_by_stage": completed_epochs_by_stage,
        "best_score": best_score,
        "best_threshold": best_threshold,
        "history": history,
        "img_size": img_size,
        "saved_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    atomic_torch_save(checkpoint, output_dir / "checkpoint_last.pth")
    global_epoch = completed_epochs_by_stage.get("head", 0) + completed_epochs_by_stage.get("finetune", 0)
    atomic_torch_save(checkpoint, output_dir / f"checkpoint_epoch_{global_epoch:03d}_{stage_name}.pth")


def try_load_resume_checkpoint(resume_path: Optional[str], output_dir: Path, auto_resume: bool, device: torch.device):
    path = None
    if resume_path:
        path = Path(resume_path)
    elif auto_resume:
        candidate = output_dir / "checkpoint_last.pth"
        if candidate.exists():
            path = candidate
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")
    print(f"\nRESUME: loading checkpoint from {path}")
    return torch.load(path, map_location=device)


def run_one_epoch(model, loader, criterion, device, optimizer=None, scaler=None, use_amp=True):
    training = optimizer is not None
    model.train(training)
    total_loss, total_items = 0.0, 0
    all_labels, all_probs = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        if training and use_amp and device.type == "cuda":
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = criterion(logits, labels)
            assert scaler is not None
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 2.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            with torch.set_grad_enabled(training):
                logits = model(images)
                loss = criterion(logits, labels)
                if training:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 2.0)
                    optimizer.step()
        probs = torch.softmax(logits.detach(), dim=1)[:, 1]
        bs = labels.size(0)
        total_loss += float(loss.item()) * bs
        total_items += bs
        all_labels.extend(labels.detach().cpu().numpy().astype(int).tolist())
        all_probs.extend(probs.detach().cpu().numpy().astype(float).tolist())
    return total_loss / max(total_items, 1), np.array(all_labels, dtype=int), np.array(all_probs, dtype=float)


def plot_history(history: List[Dict[str, object]], output_dir: Path) -> None:
    if not history:
        return
    epochs = list(range(1, len(history) + 1))
    plt.figure()
    plt.plot(epochs, [float(x["train_loss"]) for x in history], label="train_loss")
    plt.plot(epochs, [float(x["val_loss"]) for x in history], label="val_loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training and validation loss"); plt.legend(); plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=200); plt.close()
    plt.figure()
    plt.plot(epochs, [float(x["val_auc"]) for x in history], label="val_roc_auc")
    plt.plot(epochs, [float(x["val_macro_f1"]) for x in history], label="val_macro_f1")
    plt.xlabel("Epoch"); plt.ylabel("Metric"); plt.title("Validation metrics"); plt.legend(); plt.tight_layout()
    plt.savefig(output_dir / "validation_metrics.png", dpi=200); plt.close()


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], output_path: Path) -> None:
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Test confusion matrix")
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=30, ha="right")
    plt.yticks(ticks, class_names)
    threshold = cm.max() / 2.0 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, f"{cm[i, j]:d}", ha="center", va="center", color="white" if cm[i, j] > threshold else "black")
    plt.ylabel("True label"); plt.xlabel("Predicted label"); plt.tight_layout()
    plt.savefig(output_path, dpi=200); plt.close()


def plot_roc(y_true, y_prob, output_path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="random")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.title("ROC curve"); plt.legend(); plt.tight_layout()
    plt.savefig(output_path, dpi=200); plt.close()


def train_stage(args, model, model_name, stage_name, stage_epochs, lr, train_loader, val_loader, criterion, device, class_names, output_dir, completed_epochs_by_stage, best_score, best_threshold, history, checkpoint_for_resume, img_size, use_amp):
    set_trainable_for_stage(model, model_name, stage_name)
    total_params, trainable_params = count_trainable_params(model)
    print(f"\nSTAGE: {stage_name} | total params={total_params:,} | trainable={trainable_params:,}")
    optimizer = build_optimizer(model, lr=lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))
    already_done = completed_epochs_by_stage.get(stage_name, 0)
    if already_done >= stage_epochs:
        print(f"Stage '{stage_name}' already completed: {already_done}/{stage_epochs}. Skipping.")
        return best_score, best_threshold
    if checkpoint_for_resume is not None and checkpoint_for_resume.get("stage_name") == stage_name:
        try:
            optimizer.load_state_dict(checkpoint_for_resume["optimizer_state_dict"])
            if checkpoint_for_resume.get("scheduler_state_dict") is not None:
                scheduler.load_state_dict(checkpoint_for_resume["scheduler_state_dict"])
            if checkpoint_for_resume.get("scaler_state_dict") is not None:
                scaler.load_state_dict(checkpoint_for_resume["scaler_state_dict"])
            print(f"Loaded optimizer/scheduler state for stage '{stage_name}'.")
        except Exception as exc:
            print(f"Warning: could not load optimizer state, continuing with fresh optimizer. Reason: {exc}")
    for local_epoch in range(already_done + 1, stage_epochs + 1):
        start = time.time()
        train_loss, train_y, train_p = run_one_epoch(model, train_loader, criterion, device, optimizer=optimizer, scaler=scaler, use_amp=use_amp)
        val_loss, val_y, val_p = run_one_epoch(model, val_loader, criterion, device, optimizer=None, scaler=None, use_amp=False)
        threshold = choose_threshold(val_y, val_p, mode=args.threshold_mode)
        train_metrics = calculate_metrics(train_y, train_p, threshold=0.5, loss=train_loss)
        val_metrics = calculate_metrics(val_y, val_p, threshold=threshold, loss=val_loss)
        score = val_metrics.roc_auc
        if math.isnan(score):
            score = val_metrics.macro_f1
        scheduler.step(score)
        completed_epochs_by_stage[stage_name] = local_epoch
        row = {
            "phase": stage_name,
            "local_epoch": local_epoch,
            "global_epoch": completed_epochs_by_stage.get("head", 0) + completed_epochs_by_stage.get("finetune", 0),
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_metrics.loss,
            "train_auc": train_metrics.roc_auc,
            "train_macro_f1": train_metrics.macro_f1,
            "val_loss": val_metrics.loss,
            "val_auc": val_metrics.roc_auc,
            "val_macro_f1": val_metrics.macro_f1,
            "val_sensitivity_pneumonia": val_metrics.sensitivity_pneumonia,
            "val_specificity_normal": val_metrics.specificity_normal,
            "val_threshold": threshold,
            "seconds": time.time() - start,
        }
        history.append(row)
        print(f"[{stage_name}] epoch {local_epoch:02d}/{stage_epochs:02d} | train_loss={train_metrics.loss:.4f} | val_loss={val_metrics.loss:.4f} | val_auc={val_metrics.roc_auc:.4f} | val_macro_f1={val_metrics.macro_f1:.4f} | sens={val_metrics.sensitivity_pneumonia:.4f} | spec={val_metrics.specificity_normal:.4f} | threshold={threshold:.3f} | saved checkpoint")
        if score > best_score:
            best_score = float(score)
            best_threshold = float(threshold)
            atomic_torch_save({
                "model_state_dict": model.state_dict(),
                "model_name": model_name,
                "class_names": class_names,
                "threshold": best_threshold,
                "img_size": img_size,
                "metrics": asdict(val_metrics),
                "stage_name": stage_name,
                "completed_epochs_by_stage": dict(completed_epochs_by_stage),
            }, output_dir / "best_model.pth")
            save_json(output_dir / "best_validation_metrics.json", asdict(val_metrics))
            print(f"  New best model saved with validation score={best_score:.4f}")
        write_history_csv(output_dir / "history.csv", history)
        plot_history(history, output_dir)
        save_training_checkpoint(output_dir, model, optimizer, scheduler, scaler, args, class_names, stage_name, dict(completed_epochs_by_stage), best_score, best_threshold, history, img_size)
    return best_score, best_threshold


def final_test_evaluation(args, test_loader, criterion, device, output_dir: Path):
    best_path = output_dir / "best_model.pth"
    if not best_path.exists():
        print("No best_model.pth found. Skipping final test evaluation.")
        return
    ckpt = torch.load(best_path, map_location=device)
    model = build_model(ckpt["model_name"], num_classes=len(ckpt["class_names"]), dropout=0.0)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device); model.eval()
    class_names = ckpt["class_names"]
    threshold = float(ckpt.get("threshold", 0.5))
    test_loss, y_true, y_prob = run_one_epoch(model, test_loader, criterion, device, optimizer=None, scaler=None, use_amp=False)
    metrics = calculate_metrics(y_true, y_prob, threshold=threshold, loss=test_loss)
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    print("\n" + "=" * 70)
    print("FINAL TEST EVALUATION")
    print("=" * 70)
    print(f"Threshold selected on validation set: {threshold:.4f}")
    print(f"Accuracy:                {metrics.accuracy:.4f}")
    print(f"ROC-AUC:                 {metrics.roc_auc:.4f}")
    print(f"Sensitivity Pneumonia:   {metrics.sensitivity_pneumonia:.4f}")
    print(f"Specificity Normal:      {metrics.specificity_normal:.4f}")
    print(f"Precision Pneumonia:     {metrics.precision_pneumonia:.4f}")
    print(f"F1 Pneumonia:            {metrics.f1_pneumonia:.4f}")
    print(f"Macro F1:                {metrics.macro_f1:.4f}")
    print("\nConfusion matrix [NORMAL, PNEUMONIA]:")
    print(cm)
    print("\nClassification report:")
    print(report)
    save_json(output_dir / "test_metrics.json", asdict(metrics))
    (output_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    with (output_dir / "test_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["true_label", "prob_pneumonia", "pred_label"])
        writer.writeheader()
        for yt, yp, yh in zip(y_true, y_prob, y_pred):
            writer.writerow({"true_label": class_names[int(yt)], "prob_pneumonia": float(yp), "pred_label": class_names[int(yh)]})
    plot_confusion_matrix(cm, class_names, output_dir / "confusion_matrix_test.png")
    try:
        plot_roc(y_true, y_prob, output_dir / "roc_curve_test.png")
    except Exception as exc:
        print(f"Could not plot ROC: {exc}")


def parse_args():
    p = argparse.ArgumentParser(description="Resumable pretrained CNN training for pneumonia classification.")
    p.add_argument("--dataset-root", type=str, default=None)
    p.add_argument("--output-dir", type=str, default="outputs_resnet50_pneumonia")
    p.add_argument("--model", type=str, default="resnet50", choices=["resnet18", "resnet50", "densenet121", "efficientnet_b0"])
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--epochs-head", type=int, default=5)
    p.add_argument("--epochs-finetune", type=int, default=15)
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--lr-finetune", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.35)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--balance-mode", type=str, default="loss", choices=["none", "loss", "sampler"])
    p.add_argument("--threshold-mode", type=str, default="youden", choices=["youden", "macro_f1", "fixed_05"])
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint_last.pth. If not given, use --auto-resume.")
    p.add_argument("--auto-resume", action="store_true", help="Automatically resume from output_dir/checkpoint_last.pth if it exists.")
    p.add_argument("--no-amp", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and torch.cuda.is_available()
    print("=" * 70)
    print("RESUMABLE PNEUMONIA TRAINING PIPELINE")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Output dir: {output_dir.resolve()}")
    print(f"Auto resume: {args.auto_resume}")
    root = get_dataset_root(args.dataset_root)
    print(f"Dataset root: {root}")
    train_loader, val_loader, test_loader, class_names, split_info, class_weights = make_loaders(
        root=root, img_size=args.img_size, batch_size=args.batch_size, num_workers=args.num_workers,
        val_fraction=args.val_fraction, seed=args.seed, balance_mode=args.balance_mode
    )
    save_json(output_dir / "split_info.json", asdict(split_info))
    print("\nDataset split:")
    print(json.dumps(asdict(split_info), indent=2))
    model = build_model(args.model, num_classes=len(class_names), dropout=args.dropout).to(device)
    if args.balance_mode == "loss":
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        print(f"\nUsing class-weighted loss: {class_weights.tolist()}")
    else:
        criterion = nn.CrossEntropyLoss()
        print("\nUsing standard CrossEntropyLoss.")
    resume_ckpt = try_load_resume_checkpoint(args.resume, output_dir, args.auto_resume, device)
    completed_epochs_by_stage = {"head": 0, "finetune": 0}
    best_score = -1.0
    best_threshold = 0.5
    history: List[Dict[str, object]] = []
    if resume_ckpt is not None:
        model.load_state_dict(resume_ckpt["model_state_dict"])
        completed_epochs_by_stage.update(resume_ckpt.get("completed_epochs_by_stage", {}))
        best_score = float(resume_ckpt.get("best_score", -1.0))
        best_threshold = float(resume_ckpt.get("best_threshold", 0.5))
        history = list(resume_ckpt.get("history", []))
        print("Resume state:")
        print(f"  completed_epochs_by_stage = {completed_epochs_by_stage}")
        print(f"  best_score = {best_score:.4f}")
        print(f"  best_threshold = {best_threshold:.4f}")
    else:
        print("\nNo resume checkpoint loaded. Starting from epoch 0.")
    best_score, best_threshold = train_stage(args, model, args.model, "head", args.epochs_head, args.lr_head, train_loader, val_loader, criterion, device, class_names, output_dir, completed_epochs_by_stage, best_score, best_threshold, history, resume_ckpt, args.img_size, use_amp)
    best_score, best_threshold = train_stage(args, model, args.model, "finetune", args.epochs_finetune, args.lr_finetune, train_loader, val_loader, criterion, device, class_names, output_dir, completed_epochs_by_stage, best_score, best_threshold, history, resume_ckpt, args.img_size, use_amp)
    atomic_torch_save({"model_state_dict": model.state_dict(), "model_name": args.model, "class_names": class_names, "img_size": args.img_size, "args": vars(args), "completed_epochs_by_stage": completed_epochs_by_stage}, output_dir / "last_model.pth")
    final_test_evaluation(args, test_loader, criterion, device, output_dir)
    print("\nTraining completed. Important files:")
    for name in ["checkpoint_last.pth", "best_model.pth", "history.csv", "loss_curve.png", "validation_metrics.png", "test_metrics.json", "classification_report.txt", "confusion_matrix_test.png", "roc_curve_test.png"]:
        path = output_dir / name
        print(f" - {name}: {'OK' if path.exists() else 'not found'}")


if __name__ == "__main__":
    main()

'''

Path('/content/train_resnet_pneumonia_resumable.py').write_text(script_code, encoding='utf-8')
print('Script written to /content/train_resnet_pneumonia_resumable.py')
# SKIPPED_COLAB_COMMAND: !python -m py_compile /content/train_resnet_pneumonia_resumable.py
print('Syntax check passed.')


# %%
# Option A does not need DATASET_ROOT.
# The training script will try:
# kagglehub.dataset_download('paultimothymooney/chest-xray-pneumonia')
DATASET_ROOT = ''
print('DATASET_ROOT is empty: automatic KaggleHub download will be attempted.')


# %%
# Uncomment and edit this only if you use a manual dataset path.
# DATASET_ROOT = '/content/drive/MyDrive/chest_xray'

print('Current DATASET_ROOT:', repr(DATASET_ROOT))


# %%
DATASET_ARG = '' if DATASET_ROOT == '' else f'--dataset-root "{DATASET_ROOT}"'

# SKIPPED_COLAB_COMMAND: !python -u /content/train_resnet_pneumonia_resumable.py \
# SKIPPED_COLAB_CONTINUATION:   {DATASET_ARG} \
# SKIPPED_COLAB_CONTINUATION:   --model resnet50 \
# SKIPPED_COLAB_CONTINUATION:   --epochs-head 1 \
# SKIPPED_COLAB_CONTINUATION:   --epochs-finetune 1 \
# SKIPPED_COLAB_CONTINUATION:   --batch-size 16 \
# SKIPPED_COLAB_CONTINUATION:   --img-size 224 \
# SKIPPED_COLAB_CONTINUATION:   --num-workers 0 \
# SKIPPED_COLAB_CONTINUATION:   --balance-mode loss \
# SKIPPED_COLAB_CONTINUATION:   --threshold-mode youden \
# SKIPPED_COLAB_CONTINUATION:   --output-dir "{OUTPUT_DIR}" \
# SKIPPED_COLAB_CONTINUATION:   --auto-resume \
# SKIPPED_COLAB_CONTINUATION:   2>&1 | tee -a "{OUTPUT_DIR}/training_log.txt"


# %%
DATASET_ARG = '' if DATASET_ROOT == '' else f'--dataset-root "{DATASET_ROOT}"'

# SKIPPED_COLAB_COMMAND: !python -u /content/train_resnet_pneumonia_resumable.py \
# SKIPPED_COLAB_CONTINUATION:   {DATASET_ARG} \
# SKIPPED_COLAB_CONTINUATION:   --model resnet50 \
# SKIPPED_COLAB_CONTINUATION:   --epochs-head 5 \
# SKIPPED_COLAB_CONTINUATION:   --epochs-finetune 15 \
# SKIPPED_COLAB_CONTINUATION:   --batch-size 32 \
# SKIPPED_COLAB_CONTINUATION:   --img-size 224 \
# SKIPPED_COLAB_CONTINUATION:   --balance-mode loss \
# SKIPPED_COLAB_CONTINUATION:   --threshold-mode youden \
# SKIPPED_COLAB_CONTINUATION:   --output-dir "{OUTPUT_DIR}" \
# SKIPPED_COLAB_CONTINUATION:   --auto-resume \
# SKIPPED_COLAB_CONTINUATION:   2>&1 | tee -a "{OUTPUT_DIR}/training_log.txt"


# %%
DATASET_ARG = '' if DATASET_ROOT == '' else f'--dataset-root "{DATASET_ROOT}"'

# SKIPPED_COLAB_COMMAND: !python -u /content/train_resnet_pneumonia_resumable.py \
# SKIPPED_COLAB_CONTINUATION:   {DATASET_ARG} \
# SKIPPED_COLAB_CONTINUATION:   --model resnet50 \
# SKIPPED_COLAB_CONTINUATION:   --epochs-head 5 \
# SKIPPED_COLAB_CONTINUATION:   --epochs-finetune 15 \
# SKIPPED_COLAB_CONTINUATION:   --batch-size 16 \
# SKIPPED_COLAB_CONTINUATION:   --img-size 224 \
# SKIPPED_COLAB_CONTINUATION:   --num-workers 0 \
# SKIPPED_COLAB_CONTINUATION:   --balance-mode loss \
# SKIPPED_COLAB_CONTINUATION:   --threshold-mode youden \
# SKIPPED_COLAB_CONTINUATION:   --output-dir "{OUTPUT_DIR}" \
# SKIPPED_COLAB_CONTINUATION:   --auto-resume \
# SKIPPED_COLAB_CONTINUATION:   2>&1 | tee -a "{OUTPUT_DIR}/training_log.txt"


# %%
important_files = [
    'checkpoint_last.pth',
    'best_model.pth',
    'history.csv',
    'loss_curve.png',
    'validation_metrics.png',
    'test_metrics.json',
    'classification_report.txt',
    'confusion_matrix_test.png',
    'roc_curve_test.png',
    'training_log.txt',
]

for name in important_files:
    p = OUTPUT_DIR / name
    print(f'{name:30s}', 'OK' if p.exists() else 'missing')


# %%
import json

metrics_path = OUTPUT_DIR / 'test_metrics.json'
if metrics_path.exists():
    metrics = json.loads(metrics_path.read_text())
    print(json.dumps(metrics, indent=2))
else:
    print('test_metrics.json not found yet. Training may not be finished.')


# %%
from IPython.display import Image, display

for image_name in ['loss_curve.png', 'validation_metrics.png', 'confusion_matrix_test.png', 'roc_curve_test.png']:
    image_path = OUTPUT_DIR / image_name
    print('\n', image_name)
    if image_path.exists():
        display(Image(filename=str(image_path)))
    else:
        print('Missing:', image_path)


# %%
import zipfile
from google.colab import files

zip_path = Path('/content/pneumonia_resnet50_results.zip')
files_to_zip = [
    'test_metrics.json',
    'classification_report.txt',
    'confusion_matrix_test.png',
    'roc_curve_test.png',
    'loss_curve.png',
    'validation_metrics.png',
    'training_log.txt',
    'history.csv',
    'split_info.json',
]

with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
    for name in files_to_zip:
        p = OUTPUT_DIR / name
        if p.exists():
            z.write(p, arcname=name)

print('Created:', zip_path)
files.download(str(zip_path))


