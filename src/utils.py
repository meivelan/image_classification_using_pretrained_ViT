import os

import torch
import numpy as np

import matplotlib.pyplot as plt

from src.config import CIFAR10_CLASSES


def visualize_attention(model, test_loader, device, model_name: str, output_dir: str):
    """
    Visualize attention maps for a batch of test images.
    Shows what the model 'looks at' when classifying each image.
    """
    model.eval()
    imgs, labels = next(iter(test_loader))
    imgs_gpu = imgs[:8].to(device)

    # Hook to capture attention weights from last transformer block
    attention_maps = []

    def hook_fn(module, input, output):
        # output shape: (B, heads, seq_len, seq_len)
        attention_maps.append(output.detach().cpu())

    # Register hook on last attention block
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.MultiheadAttention):
            hooks.append(module.register_forward_hook(hook_fn))

    with torch.no_grad():
        logits = model(imgs_gpu)

    for h in hooks:
        h.remove()

    preds = logits.argmax(dim=1).cpu().numpy()
    imgs_np = imgs[:8].numpy()

    # Denormalize for display
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2470, 0.2435, 0.2616])

    fig, axes = plt.subplots(2, 8, figsize=(24, 6))
    for i in range(8):
        img = imgs_np[i].transpose(1, 2, 0)
        img = np.clip(img * std + mean, 0, 1)

        true_label = CIFAR10_CLASSES[labels[i].item()]
        pred_label = CIFAR10_CLASSES[preds[i]]
        color = "green" if true_label == pred_label else "red"

        axes[0, i].imshow(img)
        axes[0, i].set_title(
            f"True: {true_label}\nPred: {pred_label}", fontsize=7, color=color
        )
        axes[0, i].axis("off")

        # Attention map — average over heads, take CLS token attention
        if attention_maps:
            # Use first available attention map
            attn = attention_maps[-1]  # (B, heads, seq, seq)
            if attn.shape[0] > i:
                cls_attn = attn[i].mean(0)[0, 1:]  # mean heads, CLS row, skip CLS col
                # Reshape to patch grid
                n_patches = int(cls_attn.shape[0] ** 0.5)
                if n_patches * n_patches == cls_attn.shape[0]:
                    cls_attn = cls_attn.reshape(n_patches, n_patches).numpy()
                    axes[1, i].imshow(cls_attn, cmap="hot", interpolation="nearest")
                    axes[1, i].set_title("Attention", fontsize=7)
                    axes[1, i].axis("off")
                else:
                    axes[1, i].axis("off")
            else:
                axes[1, i].axis("off")
        else:
            axes[1, i].axis("off")

    plt.suptitle(
        f"Attention Maps — {model_name}\n(Top: image | Bottom: attention)", fontsize=12
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"{model_name}_attention.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.show()
    plt.close()
    print(f"Attention visualization saved.")


def print_comparison_table(results: list):
    print(f'\n{"="*70}')
    print(
        f'{"Model":<35} {"Acc(%)":>7} {"Size(MB)":>10} {"Time(min)":>11} {"Params(M)":>11}'
    )
    print(f'{"─"*70}')
    for r in results:
        print(
            f'{r["model"]:<35} {r["test_acc"]:>7.2f} '
            f'{r["model_size_mb"]:>10.1f} '
            f'{r["train_time_min"]:>11.1f} '
            f'{r["params_M"]:>11.1f}'
        )
    print(f'{"="*70}')
