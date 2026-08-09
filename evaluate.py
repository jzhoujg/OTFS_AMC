"""Evaluate a trained OTFS / OFDM recognition model.

Computes overall accuracy, per-class accuracy and the confusion matrix on a
test/val split, and optionally saves a confusion-matrix figure.

Example::

    python evaluate.py --data-path ./data --weights ./weights/best.pth
"""

import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from dataset import SignalDataset
from model import build_model
from utils import read_split_data, read_test_data, evaluate


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate OTFS/OFDM AMR model")
    p.add_argument("--data-path", type=str, default="./data",
                   help="root dir with <class>/*.mat (val split used unless --test-only)")
    p.add_argument("--weights", type=str, required=True, help=".pth weights to load")
    p.add_argument("--seq-len", type=int, default=2560)
    p.add_argument("--num-classes", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--val-rate", type=float, default=0.2,
                   help="used to reproduce the same val split as train.py")
    p.add_argument("--test-only", action="store_true",
                   help="treat every file under data-path as the test set")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--save-cm", type=str, default="",
                   help="if set, save the confusion-matrix figure to this path")
    return p.parse_args()


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.test_only:
        paths, labels = read_test_data(args.data_path)
    else:
        # Reproduce the same split that train.py used (same default seed).
        _, _, paths, labels = read_split_data(args.data_path, val_rate=args.val_rate)

    loader = DataLoader(
        SignalDataset(paths, labels),
        batch_size=args.batch_size, shuffle=False,
        collate_fn=SignalDataset.collate_fn)

    model = build_model(num_classes=args.num_classes, seq_len=args.seq_len).to(device)
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()

    loss, acc, confmat = evaluate(model, loader, device, num_classes=args.num_classes)
    print(f"\noverall accuracy: {acc:.4f}   loss: {loss:.4f}\n")

    # Class names (if class_indices.json exists).
    names = None
    if os.path.exists("class_indices.json"):
        with open("class_indices.json") as f:
            idx2name = json.load(f)
        names = [idx2name[str(i)] for i in range(args.num_classes)]

    if confmat is not None:
        print("confusion matrix (rows=true, cols=pred):")
        header = "        " + "  ".join(f"{(names[i] if names else i):>8}" for i in range(confmat.shape[1]))
        print(header)
        for i in range(confmat.shape[0]):
            row = confmat[i].tolist()
            total = sum(row) or 1
            row_str = "  ".join(f"{v:>8}" for v in row)
            print(f"{(names[i] if names else i):>8} {row_str}  | acc={row[i]/total:.4f}")

    if args.save_cm and confmat is not None:
        _save_confusion_matrix(confmat, names, args.save_cm)


def _save_confusion_matrix(confmat, names, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"[warn] matplotlib unavailable ({e}); skip saving figure.")
        return
    cm = confmat.float()
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(cm.shape[1]))
    ax.set_yticks(range(cm.shape[0]))
    if names:
        ax.set_xticklabels(names)
        ax.set_yticklabels(names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{confmat[i, j].item():.0f}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"saved confusion matrix to {path}")


if __name__ == "__main__":
    main(parse_args())
