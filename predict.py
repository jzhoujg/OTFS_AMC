"""Predict the air-interface (OTFS / OFDM) of a single ``.mat`` capture.

Example::

    python predict.py --weights ./weights/best.pth --input ./1.mat
"""

import argparse
import json
import os

import torch

from dataset import load_signal
from model import build_model


def parse_args():
    p = argparse.ArgumentParser(description="Predict OTFS/OFDM for one .mat file")
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--input", type=str, required=True, help="path to a .mat with sig_rec")
    p.add_argument("--seq-len", type=int, default=2560)
    p.add_argument("--num-classes", type=int, default=2)
    p.add_argument("--device", type=str, default="cuda:0")
    return p.parse_args()


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = build_model(num_classes=args.num_classes, seq_len=args.seq_len).to(device)
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()

    sig = load_signal(args.input).to(device).unsqueeze(0)  # [1, L] complex

    names = None
    if os.path.exists("class_indices.json"):
        with open("class_indices.json") as f:
            idx2name = json.load(f)
        names = [idx2name[str(i)] for i in range(args.num_classes)]

    with torch.no_grad():
        logits = model(sig)
        probs = torch.softmax(logits, dim=1)[0]
        pred = int(torch.argmax(probs).item())

    print(f"input : {args.input}")
    print(f"pred  : {pred}" + (f" ({names[pred]})" if names else ""))
    for i, pr in enumerate(probs.tolist()):
        label = names[i] if names else i
        print(f"  {label:>8}: {pr:.4f}")


if __name__ == "__main__":
    main(parse_args())
