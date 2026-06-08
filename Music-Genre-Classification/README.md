# Music Genre Classification — Six Neural Network Architectures

COMP6252 Deep Learning Technologies | Coursework 1  

---

## Overview

10-class music genre classification on the [GTZAN dataset](http://marsyas.info/downloads/datasets.html) (999 usable samples, 10 genres).  
Two input modalities are explored:

- **Net1–Net4** — MEL spectrogram images (180×180 PNG)
- **Net5–Net6** — Raw audio MFCC sequences (10×40 tensor per clip)

Best result: **Net3 (CNN + BatchNorm) at 66.34%** in just 50 epochs.

---

## Results

| Model | Architecture | 50 ep | 100 ep |
|-------|-------------|-------|--------|
| Net1 | Fully Connected (2 hidden layers) | 47.52% | 36.63% |
| Net2 | CNN | 42.57% | 50.50% |
| Net3 | CNN + BatchNorm | **66.34%** | 64.36% |
| Net4 | CNN + BatchNorm + RMSprop | 58.42% | 64.36% |
| Net5 | Bidirectional LSTM (MFCC audio) | — | 64.36% (ep. 75) |
| Net6 | BiLSTM + Unconditional GAN augmentation | — | 61.39% (ep. 90) |

---

## Architectures

### Net1 — Fully Connected
Flattens the 3×180×180 image to 97,200 dims → FC(512) → FC(256) → FC(10).  
BatchNorm1d + Dropout(0.4) at each hidden layer. Optimizer: Adam + StepLR.

### Net2 — CNN
Two conv blocks (3→32→64 and 64→128→256) with 3×3 kernels, MaxPool, and Dropout2d.  
AdaptiveAvgPool to 1×1, then FC(256→128→10). Optimizer: Adam + StepLR.

### Net3 — CNN + Batch Normalisation
Net2 with BatchNorm2d added after every Conv2d layer.  
Optimizer: Adam (lr=5e-4) + CosineAnnealingLR. Best overall model.

### Net4 — CNN + BatchNorm + RMSprop
Identical architecture to Net3; optimizer swapped to RMSprop (lr=1e-3) + StepLR.

### Net5 — Bidirectional LSTM
Input: (10, 40) MFCC tensor per clip.  
BiLSTM (hidden=64) → mean pool → FC(128→64→10). 63,178 parameters total.  
Optimizer: Adam + StepLR. Early stopping (patience=20, max 150 epochs).

### Net6 — BiLSTM + GAN Augmentation
Same architecture as Net5, trained on real + GAN-generated MFCC samples.  
An unconditional GAN (Generator: 64→128→256→400, Discriminator: 400→256→128→1)  
was trained for 80 epochs in MFCC feature space. 699 synthetic samples were added  
to the training set (1,398 total). Pseudo-labels assigned cyclically — the main reason  
accuracy dropped vs Net5.

---

## Key Findings

- **BatchNorm** gave the largest single improvement: +23.77% over plain CNN at 50 epochs.
- **Adam + cosine annealing** converged faster than RMSprop — Net3 hit 66.34% at ep. 50 vs Net4's 58.42%.
- **Net5** (63K params, 243× smaller input) nearly matched Net3 (~0.5M params), showing temporal models are competitive on small audio datasets.
- **Unconditional GAN** augmentation hurt accuracy because synthetic samples had no class label; a conditional GAN would be the fix.

---

## Setup

```bash
pip install torch torchaudio torchvision soundfile matplotlib
```

Download the [GTZAN dataset](http://marsyas.info/downloads/datasets.html) and place the folders as:

```
Data/
├── images_original/   # MEL spectrogram PNGs (for Net1–Net4)
│   ├── blues/
│   ├── classical/
│   └── ...
└── genres_original/   # Raw WAV files (for Net5–Net6)
    ├── blues/
    ├── classical/
    └── ...
```

Then run all cells in `music_genre_classification.ipynb`.

> Note: `jazz.00054.wav` is corrupted and is automatically skipped.

---

## Data Split

All models use a **70/20/10** train/val/test split with `random_seed=42`:  
699 train | 199 validation | 101 test

---

## Dependencies

| Package | Purpose |
|---------|---------|
| PyTorch | Model training |
| torchaudio | MFCC extraction, audio resampling |
| torchvision | Image loading and transforms |
| soundfile | WAV file reading and corruption check |
| matplotlib | Training curves and result plots |
