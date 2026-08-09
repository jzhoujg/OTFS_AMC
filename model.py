"""Deep learning model for OTFS / OFDM automatic modulation recognition.

This file implements the network described in our VTC2023-Spring paper:

    J. Zhou, X. Liao and Z. Gao, "Deep Learning-Based Automatic Modulation
    Recognition in OTFS and OFDM systems," 2023 IEEE 97th Vehicular Technology
    Conference (VTC2023-Spring), Florence, Italy, 2023, pp. 1-5.

The classifier tells apart two coexisting air-interfaces -- OTFS and OFDM --
from a short span of the received complex baseband signal. The architecture is
a compact multi-layer CNN (conv-5 style) followed by a Squeeze-and-Excitation
(SE) attention block that recalibrates channel-wise features. The SE block is
what lets the model cope with Doppler-spread fading, as reported in the paper.

Input  : complex tensor of shape ``[batch, seq_len]`` (default ``seq_len = 2560``).
         Each sample is the concatenated received samples produced by
         ``transmitter/otfs_syn.m`` / ``transmitter/ofdm_syn.m`` (field
         ``sig_rec``).
Output : real logits of shape ``[batch, num_classes]`` (default 2: OFDM / OTFS).
"""

import torch
import torch.nn as nn

# Received-signal length produced by the MATLAB transmitters (4 OTFS frames or
# 32 OFDM sub-frames, each 80 samples -> 2560). The fully-connected head is
# sized for this length; change SEQ_LEN and you must re-derive ``feat_dim``.
SEQ_LEN = 2560


class SEModule(nn.Module):
    """Squeeze-and-Excitation channel attention.

    Squeezes spatial dims via global average pooling, then learns a per-channel
    gate with a bottleneck FC (``channels -> channels/reduction -> channels``)
    and re-weights the input. Using ``AdaptiveAvgPool2d`` keeps this block
    independent of the (frequency) width of the feature map.
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, hidden)
        self.act1 = nn.PReLU()
        self.fc2 = nn.Linear(hidden, channels)
        self.act2 = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, H, W]
        b, c, _, _ = x.shape
        s = self.pool(x).view(b, c)
        s = self.act1(self.fc1(s))
        s = self.act2(self.fc2(s))
        return x * s.view(b, c, 1, 1)


class OTFS_OFDM_CNN(nn.Module):
    """CNN + SE attention classifier for OTFS / OFDM signal recognition.

    The complex input is split into its real / imaginary parts and arranged as
    two "rows" of a single-channel 2-D map ``[B, 1, 2, seq_len]``; the first
    conv spans both rows so real/imag are mixed from the very first layer.
    """

    def __init__(self, num_classes: int = 2, seq_len: int = SEQ_LEN, in_channels: int = 1):
        super().__init__()
        self.seq_len = seq_len

        # Conv block 1: mix real/imag (kernel height 2), large receptive field
        # along the time axis (width 128, stride 4).
        self.features_1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=(2, 128), stride=4, padding=0),
            nn.BatchNorm2d(64),
            nn.PReLU(),
        )
        # Conv block 2: further downsample the time axis.
        self.features_2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=(1, 8), stride=8, padding=0),
            nn.BatchNorm2d(32),
            nn.PReLU(),
        )
        self.se = SEModule(channels=32, reduction=4)

        # Flattened feature width after the two conv blocks (see _feat_dim).
        feat_dim = self._feat_dim(seq_len)
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.PReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes),
        )

        self.apply(self._init_weights)

    def _feat_dim(self, seq_len: int) -> int:
        """Compute the flattened width produced by the conv front-end.

        With the default kernels/strides and ``seq_len = 2560`` this is 2432
        (32 channels x 1 x 76). We derive it from a dummy forward so the head is
        always consistent with ``seq_len``.
        """
        with torch.no_grad():
            x = torch.zeros(1, 1, 2, seq_len)
            x = self.features_1(x)
            x = self.features_2(x)
            x = self.se(x)
            return x.numel()

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.Conv2d, nn.Conv3d)):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def to_2d(self, x: torch.Tensor) -> torch.Tensor:
        """Convert a complex ``[B, seq_len]`` tensor to ``[B, 1, 2, seq_len]``.

        Also accepts ``[B, seq_len, 1]`` for backward compatibility with the
        original demo. Handles ``batch == 1`` correctly (no over-squeeze).
        """
        if x.dim() == 3:
            x = x.view(x.shape[0], -1)          # [B, L, 1] -> [B, L]
        assert x.shape[1] == self.seq_len, (
            f"expected seq_len={self.seq_len}, got {x.shape[1]}"
        )
        real = x.real
        imag = x.imag
        two = torch.stack([real, imag], dim=1)   # [B, 2, L]
        return two.unsqueeze(1)                  # [B, 1, 2, L]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.to_2d(x)
        x = self.features_1(x)
        x = self.features_2(x)
        x = self.se(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x


def build_model(num_classes: int = 2, seq_len: int = SEQ_LEN, **kwargs) -> OTFS_OFDM_CNN:
    """Factory used by the training / evaluation scripts."""
    return OTFS_OFDM_CNN(num_classes=num_classes, seq_len=seq_len, **kwargs)


if __name__ == "__main__":
    # Quick smoke test on the bundled sample (a single OTFS frame).
    import numpy as np
    import scipy.io as sio

    mat = sio.loadmat("./1.mat")
    sig = np.asarray(mat["sig_rec"]).squeeze()        # complex, [L]
    sig = torch.from_numpy(sig).to(torch.complex128)  # [L]
    sig = sig.unsqueeze(0)                             # [1, L]

    model = OTFS_OFDM_CNN(num_classes=2)
    logits = model(sig)
    print("input shape :", tuple(sig.shape))
    print("output shape:", tuple(logits.shape))
    print("logits      :", logits.detach())
