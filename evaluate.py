import torch

from src.config import CIFAR10_CLASSES

from src.model import model_size_mb

from sklearn.metrics import (
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import os

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.amp.autocast(device_type='cuda'):
            logits = model(imgs)
            loss   = criterion(logits, labels)

        total_loss += loss.item() * imgs.size(0)
        preds       = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, all_preds, all_labels


def full_evaluation(model, test_loader, criterion, device,
                    model_name: str, output_dir: str,
                    training_time: float, log: list):
    """Evaluate on test set, save confusion matrix and classification report."""
    print(f'\n{"="*55}')
    print(f'Test Evaluation: {model_name}')
    print(f'{"="*55}')

    te_loss, te_acc, preds, labels = evaluate(
        model, test_loader, criterion, device)
    print(f'Test Loss     : {te_loss:.4f}')
    print(f'Test Accuracy : {te_acc*100:.2f}%')
    print(f'Model Size    : {model_size_mb(model):.1f} MB')
    print(f'Training Time : {training_time/60:.1f} min')

    # Per-class report
    report = classification_report(
        labels, preds, target_names=CIFAR10_CLASSES, digits=4)
    print(f'\nClassification Report:\n{report}')

    # Confusion matrix
    cm  = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay(cm, display_labels=CIFAR10_CLASSES).plot(
        ax=ax, cmap='Blues', colorbar=False, xticks_rotation=45)
    ax.set_title(f'Confusion Matrix — {model_name}\nTest Acc: {te_acc*100:.2f}%',
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_cm.png'), dpi=150)
    plt.show(); plt.close()

    # Training curves
    epochs    = [e['epoch']      for e in log]
    tr_accs   = [e['train_acc']  for e in log]
    va_accs   = [e['val_acc']    for e in log]
    tr_losses = [e['train_loss'] for e in log]
    va_losses = [e['val_loss']   for e in log]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(epochs, tr_losses, label='Train'); axes[0].plot(epochs, va_losses, label='Val')
    axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(True)
    axes[1].plot(epochs, [a*100 for a in tr_accs], label='Train')
    axes[1].plot(epochs, [a*100 for a in va_accs], label='Val')
    axes[1].set_title('Accuracy (%)'); axes[1].legend(); axes[1].grid(True)
    plt.suptitle(f'Training Curves — {model_name}', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_curves.png'), dpi=150)
    plt.show(); plt.close()

    return {
        'model':         model_name,
        'test_acc':      round(te_acc * 100, 2),
        'test_loss':     round(te_loss, 4),
        'model_size_mb': round(model_size_mb(model), 1),
        'train_time_min':round(training_time / 60, 1),
        'params_M':      round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
    }

