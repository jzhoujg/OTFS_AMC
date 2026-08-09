"""Extended experiment: multi-modulation recognition on OTFS delay-Doppler frames.

This is a *beyond-paper* extension. While the main repository only tells OTFS and
OFDM apart (binary), this model classifies the **underlying modulation order**
(BPSK / QPSK / 8PSK / 16-QAM / 64-QAM / 256-QAM / ...) from the OTFS received
signal, treating it as a stack of delay-Doppler frames and processing it with a
3-D CNN.

Status: research-grade. It is provided for transparency and as a clean starting
point; it was **not** validated end-to-end in the release environment.

Input : complex tensor ``[batch, L]`` where ``L = num_frames * n_sc * n_ti``.
        With the binary transmitter's framing (``4 * 80 * 8``) this is ``L = 2560``.
Output: real logits ``[batch, num_classes]`` (default 8).
"""

import torch
import torch.nn as nn


class OTFS3DCNN(nn.Module):
    """3-D CNN over the (frames x subcarriers x time-indices) delay-Doppler grid.

    Real and imaginary parts are stacked as the two input channels. An
    ``AdaptiveAvgPool3d(1)`` before the classifier makes the head independent of
    the exact frame geometry, so the model builds for any ``num_frames/n_sc/n_ti``.
    """

    def __init__(self, num_classes: int = 8, num_frames: int = 4, n_sc: int = 80, n_ti: int = 8):
        super().__init__()
        self.num_frames = num_frames
        self.n_sc = n_sc
        self.n_ti = n_ti
        self.seq_len = num_frames * n_sc * n_ti

        self.features = nn.Sequential(
            nn.Conv3d(2, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.PReLU(),
            nn.MaxPool3d(kernel_size=2),                  # downsample 2x
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.PReLU(),
            nn.MaxPool3d(kernel_size=2),
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.PReLU(),
            nn.AdaptiveAvgPool3d(1),                      # [B, 256, 1, 1, 1]
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.PReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def to_3d(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, L]`` complex -> ``[B, 2, num_frames, n_sc, n_ti]`` real."""
        if x.dim() == 3:
            x = x.view(x.shape[0], -1)
        assert x.shape[1] == self.seq_len, (
            f"expected seq_len={self.seq_len} "
            f"(= {self.num_frames}*{self.n_sc}*{self.n_ti}), got {x.shape[1]}"
        )
        b = x.shape[0]
        real = x.real.view(b, 1, self.num_frames, self.n_sc, self.n_ti)
        imag = x.imag.view(b, 1, self.num_frames, self.n_sc, self.n_ti)
        return torch.cat([real, imag], dim=1)  # [B, 2, F, H, W]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.to_3d(x)
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    # Smoke test with the bundled 2560-sample OTFS capture (4x80x8 framing).
    import numpy as np
    import scipy.io as sio

    mat = sio.loadmat("../../1.mat")
    sig = np.asarray(mat["sig_rec"]).squeeze()
    sig = torch.from_numpy(sig.astype(np.complex128)).unsqueeze(0)  # [1, 2560]
    model = OTFS3DCNN(num_classes=8)
    logits = model(sig)
    print("input shape :", tuple(sig.shape))
    print("output shape:", tuple(logits.shape))  # expect [1, 8]
