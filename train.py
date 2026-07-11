import os
import time
import json

import torch
import torch.nn as nn

import torch.optim as optim

from evaluate import evaluate


from src.model import build_model

class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best_acc  = 0.0

    def __call__(self, val_acc: float) -> bool:
        if val_acc > self.best_acc + self.min_delta:
            self.best_acc = val_acc
            self.counter  = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def train_one_epoch(model, loader, optimizer, criterion,
                    scaler, device, scheduler=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type='cuda'):
            logits = model(imgs)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * imgs.size(0)
        preds       = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += imgs.size(0)

    return total_loss / total, correct / total

def train(cfg: dict, model_name: str, device: torch.device,
          train_loader, val_loader):
    """Full training loop with logging, checkpointing, early stopping."""
    os.makedirs(cfg['output_dir'], exist_ok=True)
    ckpt_path = os.path.join(cfg['output_dir'], f'{model_name}_best.pth')

    model     = build_model(model_name, cfg['num_classes'], device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(),
                            lr=cfg['lr'],
                            weight_decay=cfg['weight_decay'])

    total_steps = cfg['epochs'] * len(train_loader)
    warmup_steps = cfg['warmup_epochs'] * len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg['lr'],
        total_steps=total_steps,
        pct_start=max(0.1, warmup_steps / total_steps),
    )
    scaler       = torch.amp.GradScaler('cuda', enabled=cfg['use_amp'])
    early_stop   = EarlyStopping(patience=5)

    log        = []
    best_acc   = 0.0
    t_start    = time.time()

    print(f'\n{"Epoch":>6} {"TrLoss":>8} {"TrAcc":>7} '
          f'{"VaLoss":>8} {"VaAcc":>7} {"Time":>6}')
    print('─' * 50)

    for epoch in range(1, cfg['epochs'] + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion,
            scaler, device, scheduler)
        va_loss, va_acc, _, _ = evaluate(
            model, val_loader, criterion, device)
        ep_time = time.time() - t0

        print(f'{epoch:>6} {tr_loss:>8.4f} {tr_acc*100:>6.2f}% '
              f'{va_loss:>8.4f} {va_acc*100:>6.2f}% {ep_time:>5.1f}s')

        log.append({
            'epoch': epoch,
            'train_loss': round(tr_loss, 4), 'train_acc': round(tr_acc, 4),
            'val_loss':   round(va_loss, 4), 'val_acc':   round(va_acc, 4),
            'epoch_time': round(ep_time, 2),
        })

        if va_acc > best_acc:
            best_acc = va_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': best_acc,
                'config': cfg,
            }, ckpt_path)
            print(f'  ✓ Best saved (val_acc={best_acc*100:.2f}%)')

        if early_stop(va_acc):
            print(f'  Early stop at epoch {epoch}')
            break

    total_time = time.time() - t_start
    print(f'\nTraining complete in {total_time/60:.1f} min | Best val acc: {best_acc*100:.2f}%')

    with open(os.path.join(cfg['output_dir'],
                           f'{model_name}_log.json'), 'w') as f:
        json.dump(log, f, indent=2)

    return model, log, ckpt_path, total_time
