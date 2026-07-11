import os
import json
import argparse

import torch
from torch import nn

from src.config import DEFAULT_CFG
from src.setup import set_seed
from src.setup import get_device

from src.data import build_dataloaders
from train import train
from evaluate import full_evaluation

from src.utils import visualize_attention
from src.utils import print_comparison_table


def run(model_name: str, cfg: dict):
    device = get_device()
    set_seed(cfg["seed"])
    os.makedirs(cfg["output_dir"], exist_ok=True)

    print(f'\n{"="*55}')
    print(f"Running: {model_name}")
    print(f'{"="*55}')

    train_loader, val_loader, test_loader = build_dataloaders(cfg)
    model, log, ckpt_path, train_time = train(
        cfg, model_name, device, train_loader, val_loader
    )

    # Load best checkpoint for evaluation
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    criterion = nn.CrossEntropyLoss()
    result = full_evaluation(
        model,
        test_loader,
        criterion,
        device,
        model_name,
        cfg["output_dir"],
        train_time,
        log,
    )

    visualize_attention(model, test_loader, device, model_name, cfg["output_dir"])

    return result


def main():
    parser = argparse.ArgumentParser(description="ViT CIFAR-10 Fine-tuning")
    parser.add_argument(
        "--model", default="vit_small_patch16_224", help="timm model name"
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument(
        "--compare", action="store_true", help="Compare vit_small and vit_base"
    )
    parser.add_argument("--output_dir", default="./outputs")
    args = parser.parse_args()

    cfg = {
        **DEFAULT_CFG,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "output_dir": args.output_dir,
    }

    if args.compare:
        models = ["vit_small_patch16_224", "vit_base_patch16_224"]
        results = []
        for m in models:
            results.append(run(m, cfg))
        print_comparison_table(results)

        # Save comparison JSON
        with open(os.path.join(cfg["output_dir"], "comparison.json"), "w") as f:
            json.dump(results, f, indent=2)
    else:
        result = run(args.model, cfg)
        print_comparison_table([result])

    print(f'\nAll outputs saved to: {cfg["output_dir"]}/')


if __name__ == "__main__":
    main()
