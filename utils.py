"""Shared helpers: dataset splitting and train/eval loops."""

import json
import os
import random
import sys

import torch
from tqdm import tqdm


def read_split_data(root: str, val_rate: float = 0.2, seed: int = 0):
    """Walk an ImageFolder-style directory of ``.mat`` files.

    Each sub-directory of ``root`` is one class (e.g. ``ofdm/``, ``otfs/``).
    Returns ``(train_paths, train_labels, val_paths, val_labels)`` and writes a
    ``class_indices.json`` mapping ``{class_index: class_name}`` next to CWD.
    """
    random.seed(seed)
    assert os.path.isdir(root), f"dataset root '{root}' does not exist."

    classes = sorted(c for c in os.listdir(root)
                     if os.path.isdir(os.path.join(root, c)))
    assert classes, f"no class folders found under '{root}'."
    class_indices = {name: i for i, name in enumerate(classes)}

    with open("class_indices.json", "w") as f:
        json.dump({str(i): name for name, i in class_indices.items()}, f, indent=4)

    train_paths, train_labels = [], []
    val_paths, val_labels = [], []
    every_class_num = []

    for name in classes:
        cls = class_indices[name]
        files = [os.path.join(root, name, fn)
                 for fn in os.listdir(os.path.join(root, name))
                 if fn.endswith(".mat")]
        every_class_num.append(len(files))
        val_files = set(random.sample(files, k=int(len(files) * val_rate)))
        for fp in files:
            if fp in val_files:
                val_paths.append(fp)
                val_labels.append(cls)
            else:
                train_paths.append(fp)
                train_labels.append(cls)

    total = sum(every_class_num)
    print(f"found {total} samples ({len(classes)} classes: {classes})")
    print(f"  train: {len(train_paths)}    val: {len(val_paths)}")
    return train_paths, train_labels, val_paths, val_labels


def read_test_data(root: str):
    """Return ``(paths, labels)`` for every ``.mat`` under ``root/<class>/``."""
    assert os.path.isdir(root), f"dataset root '{root}' does not exist."
    classes = sorted(c for c in os.listdir(root)
                     if os.path.isdir(os.path.join(root, c)))
    class_indices = {name: i for i, name in enumerate(classes)}
    paths, labels = [], []
    for name in classes:
        cls = class_indices[name]
        for fn in os.listdir(os.path.join(root, name)):
            if fn.endswith(".mat"):
                paths.append(os.path.join(root, name, fn))
                labels.append(cls)
    return paths, labels


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_fn = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    sample_num = 0
    optimizer.zero_grad()

    pbar = tqdm(data_loader, file=sys.stdout)
    for step, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        sample_num += images.shape[0]

        pred = model(images)
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels).sum()

        loss = loss_fn(pred, labels)
        loss.backward()
        accu_loss += loss.detach()

        if not torch.isfinite(loss):
            print("WARNING: non-finite loss, ending training:", loss.item())
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

        pbar.set_description(
            f"[train epoch {epoch}] loss: {accu_loss.item()/(step+1):.3f} "
            f"acc: {accu_num.item()/sample_num:.3f}"
        )
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch=0, num_classes=None):
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    accu_num = torch.zeros(1).to(device)
    accu_loss = torch.zeros(1).to(device)
    sample_num = 0

    if num_classes is not None:
        confmat = torch.zeros(num_classes, num_classes, dtype=torch.long)
    else:
        confmat = None

    pbar = tqdm(data_loader, file=sys.stdout)
    for step, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        sample_num += images.shape[0]

        pred = model(images)
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels).sum()
        accu_loss += loss_fn(pred, labels)

        if confmat is not None:
            for t, p in zip(labels.tolist(), pred_classes.tolist()):
                confmat[t, p] += 1

        pbar.set_description(
            f"[valid epoch {epoch}] loss: {accu_loss.item()/(step+1):.3f} "
            f"acc: {accu_num.item()/sample_num:.3f}"
        )
    loss = accu_loss.item() / (step + 1)
    acc = accu_num.item() / sample_num
    return loss, acc, confmat
