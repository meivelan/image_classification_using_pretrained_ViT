import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import urllib.request
import io

import torch
import torchvision.transforms as T
import timm

from src.config import CIFAR10_CLASSES

INFER_TRANSFORM = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
    ]
)


def load_model(weights_path: str, device: torch.device) -> tuple:
    """
    Load trained ViT from checkpoint.
    Returns (model, model_name, config)
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Checkpoint not found: {weights_path}")

    ckpt = torch.load(weights_path, map_location=device)
    model_name = ckpt.get("model_name", "vit_small_patch16_224")
    config = ckpt.get("config", {})

    model = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=10,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    val_acc = ckpt.get("val_acc", "?")
    print(f"Model loaded : {model_name}")
    print(
        f"Val acc      : {val_acc:.4f}"
        if isinstance(val_acc, float)
        else f"Val acc: {val_acc}"
    )
    print(f"Checkpoint   : {weights_path}")

    return model, model_name, config


def load_image(source) -> Image.Image:
    """
    Load a PIL image from:
      - file path (str or Path)
      - URL (str starting with http)
      - PIL Image directly
      - numpy array
    """
    if isinstance(source, Image.Image):
        return source.convert("RGB")
    if isinstance(source, np.ndarray):
        return Image.fromarray(source).convert("RGB")
    if isinstance(source, (str, Path)):
        src = str(source)
        if src.startswith("http://") or src.startswith("https://"):
            with urllib.request.urlopen(src) as resp:
                return Image.open(io.BytesIO(resp.read())).convert("RGB")
        return Image.open(src).convert("RGB")
    raise ValueError(f"Unsupported image source type: {type(source)}")


@torch.no_grad()
def predict_single(model, image_source, device, top_k: int = 5) -> dict:
    """
    Run inference on a single image.

    Args:
        model:        loaded model (from load_model)
        image_source: file path, URL, PIL image, or numpy array
        device:       torch.device
        top_k:        number of top predictions to return

    Returns:
        dict with keys:
          predicted_class  — class name with highest score
          confidence       — probability (0-1) of predicted class
          top_k            — list of (class, probability) for top k classes
          class_idx        — integer index of predicted class
    """
    img = load_image(image_source)
    tensor = INFER_TRANSFORM(img).unsqueeze(0).to(device)

    logits = model(tensor)  # (1, 10)
    probs = torch.softmax(logits, dim=1)[0]  # (10,)

    top_probs, top_idxs = torch.topk(probs, k=min(top_k, 10))

    return {
        "predicted_class": CIFAR10_CLASSES[probs.argmax().item()],
        "confidence": round(probs.max().item(), 4),
        "class_idx": probs.argmax().item(),
        "top_k": [
            {
                "class": CIFAR10_CLASSES[idx.item()],
                "probability": round(prob.item(), 4),
                "rank": i + 1,
            }
            for i, (prob, idx) in enumerate(zip(top_probs, top_idxs))
        ],
    }


@torch.no_grad()
def predict_batch(model, image_list: list, device, batch_size: int = 32) -> list:
    """
    Run inference on a list of images efficiently using batches.

    Args:
        model:       loaded model
        image_list:  list of file paths / PIL images / URLs
        device:      torch.device
        batch_size:  number of images per forward pass

    Returns:
        list of dicts (same format as predict_single)
    """
    results = []

    for start in range(0, len(image_list), batch_size):
        batch_sources = image_list[start : start + batch_size]
        tensors = []
        for src in batch_sources:
            img = load_image(src)
            tensors.append(INFER_TRANSFORM(img))

        batch = torch.stack(tensors).to(device)  # (B, 3, 224, 224)
        logits = model(batch)  # (B, 10)
        probs = torch.softmax(logits, dim=1)  # (B, 10)

        for i in range(len(batch_sources)):
            p = probs[i]
            pred_idx = p.argmax().item()
            top5p, top5i = torch.topk(p, k=5)
            results.append(
                {
                    "predicted_class": CIFAR10_CLASSES[pred_idx],
                    "confidence": round(p.max().item(), 4),
                    "class_idx": pred_idx,
                    "top_k": [
                        {
                            "class": CIFAR10_CLASSES[idx.item()],
                            "probability": round(prob.item(), 4),
                            "rank": j + 1,
                        }
                        for j, (prob, idx) in enumerate(zip(top5p, top5i))
                    ],
                }
            )

        print(
            f"  Processed {min(start+batch_size, len(image_list))}"
            f"/{len(image_list)}",
            end="\r",
        )

    print()
    return results


def predict_folder(
    folder_path: str,
    weights_path: str,
    output_dir: str = "./inference_results",
    batch_size: int = 32,
) -> list:
    """
    Run inference on all images in a folder.
    Saves results to JSON and a visualization grid.

    Supported formats: .jpg .jpeg .png .bmp .tiff .webp
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_name, _ = load_model(weights_path, device)
    os.makedirs(output_dir, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    paths = [str(p) for p in Path(folder_path).iterdir() if p.suffix.lower() in exts]

    if not paths:
        print(f"No images found in {folder_path}")
        return []

    print(f"\nRunning inference on {len(paths)} images...")
    results = predict_batch(model, paths, device, batch_size)

    for i, r in enumerate(results):
        r["file"] = os.path.basename(paths[i])

    # Save JSON
    out_json = os.path.join(output_dir, "predictions.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Predictions saved: {out_json}")

    # Visualize first 16
    _visualize_batch(paths[:16], results[:16], model_name, output_dir)

    return results


def _visualize_batch(paths, results, model_name, output_dir):
    n = min(16, len(paths))
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes.flatten()

    for i in range(n):
        img = Image.open(paths[i]).convert("RGB")
        axes[i].imshow(img)
        r = results[i]
        label = f'{r["predicted_class"]}\n{r["confidence"]*100:.1f}%'
        axes[i].set_title(label, fontsize=9)
        axes[i].axis("off")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.suptitle(f"Inference Results — {model_name}", fontsize=12)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "inference_grid.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"Visualization saved: {out_path}")


def visualize_prediction(image_source, result: dict, save_path: str = None):
    """
    Show a single image with its prediction and confidence bar chart.
    """
    img = load_image(image_source)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Image
    ax1.imshow(img)
    ax1.set_title(
        f'Predicted: {result["predicted_class"].upper()}\n'
        f'Confidence: {result["confidence"]*100:.1f}%',
        fontsize=12,
        fontweight="bold",
    )
    ax1.axis("off")

    # Top-5 bar chart
    top5 = result["top_k"][:5]
    classes = [t["class"] for t in top5]
    probs = [t["probability"] * 100 for t in top5]
    colors = ["#6C63FF" if i == 0 else "#A0A3B1" for i in range(len(top5))]

    bars = ax2.barh(classes[::-1], probs[::-1], color=colors[::-1])
    ax2.set_xlabel("Confidence (%)")
    ax2.set_title("Top-5 Predictions", fontsize=11)
    ax2.set_xlim(0, 100)

    for bar, prob in zip(bars, probs[::-1]):
        ax2.text(
            bar.get_width() + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{prob:.1f}%",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()
    plt.close()


def predict(
    image_source,
    weights_path: str = "outputs/vit_small_patch16_224_best.pth",
    top_k: int = 5,
    visualize: bool = True,
    save_path: str = None,
) -> dict:
    """
    Single-image inference — main entry point for Colab use.

    Args:
        image_source: path, URL, PIL image, or numpy array
        weights_path: path to best.pth checkpoint
        top_k:        number of top predictions
        visualize:    show image + bar chart
        save_path:    save visualization to this path

    Returns:
        dict with predicted_class, confidence, top_k list

    Example:
        from inference import predict
        r = predict('dog.jpg', weights_path='outputs/vit_small_patch16_224_best.pth')
        print(r['predicted_class'], r['confidence'])
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, _ = load_model(weights_path, device)
    result = predict_single(model, image_source, device, top_k)

    print(f'\nPrediction : {result["predicted_class"].upper()}')
    print(f'Confidence : {result["confidence"]*100:.2f}%')
    print(f"\nTop-{top_k}:")
    for t in result["top_k"]:
        bar = "█" * int(t["probability"] * 30)
        print(f'  {t["rank"]}. {t["class"]:<12} {t["probability"]*100:5.1f}%  {bar}')

    if visualize:
        visualize_prediction(image_source, result, save_path)

    return result


def main():
    parser = argparse.ArgumentParser(description="ViT CIFAR-10 Inference")
    parser.add_argument("--weights", required=True, help="Path to best.pth checkpoint")
    parser.add_argument("--image", default=None, help="Path to single image")
    parser.add_argument("--url", default=None, help="Image URL")
    parser.add_argument("--folder", default=None, help="Folder of images")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--output_dir", default="./inference_results")
    args = parser.parse_args()

    if args.image:
        predict(
            args.image,
            weights_path=args.weights,
            top_k=args.top_k,
            save_path=os.path.join(args.output_dir, "prediction.png"),
        )

    elif args.url:
        predict(
            args.url,
            weights_path=args.weights,
            top_k=args.top_k,
            save_path=os.path.join(args.output_dir, "prediction.png"),
        )

    elif args.folder:
        predict_folder(
            args.folder, weights_path=args.weights, output_dir=args.output_dir
        )

    else:
        print("Provide --image, --url, or --folder")
        sys.exit(1)


if __name__ == "__main__":
    main()
