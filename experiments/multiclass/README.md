# Extended experiment: multi-modulation recognition

> ⚠️ This is a **beyond-paper extension**, included for transparency. It is
> research-grade and was **not** re-validated end-to-end in the release
> environment. Use it as a clean starting point, not a finished baseline.

## What it does

The main repository solves a **binary** problem: tell OTFS and OFDM apart. This
folder goes one step further and classifies the **underlying modulation order**
of an OTFS signal — e.g. BPSK / QPSK / 8PSK / 16-QAM / 64-QAM / 256-QAM — from
the received delay-Doppler frames, using a 3-D CNN (`model.py`).

## Model

`OTFS3DCNN` reshapes the length-`L` complex capture into a 5-D tensor
`[B, 2, num_frames, n_sc, n_ti]` (real/imag as two channels), passes it through
three `Conv3d` blocks with batch-norm + pooling, and classifies with a small
FC head. An `AdaptiveAvgPool3d(1)` decouples the head from the exact frame
geometry, so any `num_frames × n_sc × n_ti` framing works.

```bash
python model.py        # smoke test on ../../1.mat (4×80×8 framing, 8 classes)
```

## Data

You can reuse the **same waveforms** produced by `transmitter/otfs_syn.m`: that
script already emits per-modulation folders (`b1_otfs_bpsk/`, `b2_otfs_qpsk/`,
…). For this experiment, simply label each sample by its **modulation** instead
of by the OTFS/OFDM scheme, e.g. organize into `data/bpsk/`, `data/qpsk/`, …

## Training / evaluation

The training and evaluation entry points in the repository root
(`train.py`, `evaluate.py`, `utils.py`, `dataset.py`) are written generically —
they read `<class>/*.mat` folders, so you can train this 3-D model by swapping
in `OTFS3DCNN` for `OTFS_OFDM_CNN` and setting `--num-classes` accordingly.
A dedicated runner is intentionally **not** provided, since the exact class set
and framing depend on which experiment you wish to reproduce.

## Provenance

The author's original working directory contained several exploratory 3-D
variants (different input lengths / kernel sizes). They were inconsistent and,
in places, dimensionally invalid for their stated inputs. This folder ships a
single cleaned, runnable model in their spirit rather than the raw variants.
