"""Train the OTFS / OFDM recognition model.

Example::

    python train.py --data-path ./data --epochs 50 --batch-size 128 --lr 0.01

The dataset under ``--data-path`` must be ImageFolder-style::

    data/ofdm/*.mat
    data/otfs/*.mat

Use ``dataset.py``/``prepare_data.py`` to assemble it from the MATLAB output.
"""

import argparse
import math
import os

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

from dataset import SignalDataset
from model import build_model
from utils import read_split_data, train_one_epoch, evaluate


def parse_args():
    p = argparse.ArgumentParser(description="Train OTFS/OFDM AMR model")
    p.add_argument("--data-path", type=str, default="./data",
                   help="root dir with <class>/*.mat sub-folders")
    p.add_argument("--seq-len", type=int, default=2560,
                   help="received-signal length (must match the transmitter)")
    p.add_argument("--num-classes", type=int, default=2)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--lrf", type=float, default=0.01,
                   help="final LR ratio for cosine annealing (min_lr = lr*lrf)")
    p.add_argument("--val-rate", type=float, default=0.2)
    p.add_argument("--weights", type=str, default="",
                   help="path to .pth to resume from (empty = train from scratch)")
    p.add_argument("--out-dir", type=str, default="./weights")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # Optional TensorBoard logging; degrade gracefully if not installed.
    tb_writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter()
    except Exception as e:  # pragma: no cover
        print(f"[info] TensorBoard unavailable ({e}); logging to stdout only.")

    train_paths, train_labels, val_paths, val_labels = read_split_data(
        args.data_path, val_rate=args.val_rate)

    train_loader = DataLoader(
        SignalDataset(train_paths, train_labels),
        batch_size=args.batch_size, shuffle=True, pin_memory=True,
        num_workers=args.num_workers, collate_fn=SignalDataset.collate_fn)
    val_loader = DataLoader(
        SignalDataset(val_paths, val_labels),
        batch_size=args.batch_size, shuffle=False, pin_memory=True,
        num_workers=args.num_workers, collate_fn=SignalDataset.collate_fn)

    model = build_model(num_classes=args.num_classes, seq_len=args.seq_len).to(device)

    if args.weights:
        assert os.path.exists(args.weights), f"weights '{args.weights}' not found."
        state = torch.load(args.weights, map_location=device)
        print(model.load_state_dict(state, strict=False))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5e-5)
    # Cosine annealing schedule (https://arxiv.org/abs/1812.01187).
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    best_acc = 0.0
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(
            model, optimizer, train_loader, device, epoch)
        scheduler.step()

        val_loss, val_acc, _ = evaluate(
            model, val_loader, device, epoch=epoch, num_classes=args.num_classes)

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} lr={lr_now:.5f}")

        if tb_writer is not None:
            tb_writer.add_scalar("train_loss", train_loss, epoch)
            tb_writer.add_scalar("train_acc", train_acc, epoch)
            tb_writer.add_scalar("val_loss", val_loss, epoch)
            tb_writer.add_scalar("val_acc", val_acc, epoch)
            tb_writer.add_scalar("lr", lr_now, epoch)

        # Save best + latest.
        torch.save(model.state_dict(), os.path.join(args.out_dir, "latest.pth"))
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best.pth"))
            print(f"  -> new best val_acc={best_acc:.4f}, saved best.pth")

    print(f"training finished. best val_acc={best_acc:.4f}")


if __name__ == "__main__":
    main(parse_args())
