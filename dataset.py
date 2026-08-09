"""Dataset utilities for OTFS / OFDM signal recognition.

Each sample is a MATLAB ``.mat`` file containing a ``sig_rec`` field -- a
complex vector of length ``model.SEQ_LEN`` (2560) produced by the transmitters
in ``transmitter/``. The dataset is laid out ImageFolder-style::

    data/
        ofdm/   *.mat   (label 0)
        otfs/   *.mat   (label 1)

so that ``utils.read_split_data`` can walk the class folders directly.
"""

import os

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset


def load_signal(path: str) -> torch.Tensor:
    """Load one ``.mat`` file and return its ``sig_rec`` as a complex tensor."""
    mat = sio.loadmat(path)
    sig = np.asarray(mat["sig_rec"]).squeeze()
    if np.iscomplexobj(sig):
        sig = sig.astype(np.complex128)
    else:
        # Some MATLAB exports store real/imag separately or as float; coerce.
        sig = sig.astype(np.complex128)
    return torch.from_numpy(sig)  # [L], torch.complex128


class SignalDataset(Dataset):
    """A dataset of ``.mat`` signal files with integer class labels."""

    def __init__(self, file_paths, labels):
        assert len(file_paths) == len(labels)
        self.file_paths = list(file_paths)
        self.labels = list(labels)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        sig = load_signal(self.file_paths[idx])  # complex [L]
        return sig, int(self.labels[idx])

    @staticmethod
    def collate_fn(batch):
        """Stack complex signals and labels into a mini-batch.

        ``torch.stack`` handles complex tensors, but we spell it out so the
        DataLoader never trips on the complex dtype.
        """
        sigs, labels = zip(*batch)
        sigs = torch.stack(sigs, dim=0)          # [B, L] complex
        labels = torch.as_tensor(labels, dtype=torch.long)
        return sigs, labels
