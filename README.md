# TPE Predict Force

ResNet18-based regression model that predicts **contact force magnitude and angle** from grayscale particle contact-region images.

---

## Repository contents

| File | Description |
|---|---|
| `Contact_force_regression_label_produce.ipynb` | Crops per-contact images from synthetic simulations and builds the dataset (`labels.npy`) |
| `Contact_force_regression_TRAINING.ipynb` | Two-phase ResNet18 training: warmup (frozen backbone) then full fine-tuning |

---

## Data format

`Contact_force_regression_label_produce.ipynb` expects the following layout on disk:

```
<img_parent_dir>/
    0/   ← integer-named subfolder
        img_00001.png
        img_00002.png
        ...
    1/
        ...

<label_parent_dir>/
    0/
        mags.npy          ← shape (N, C)  force magnitudes
        angles_inner.npy  ← shape (N, C)  contact positions (radians)
        angles_tang.npy   ← shape (N, C)  force directions (radians)
    1/
        ...
```

It outputs:

```
<output_dir>/
    00001.png  ...  NNNNN.png   ← one crop per non-zero contact
    labels.npy                  ← shape (N_contacts, 2): [force_mag, force_angle]
```

---

## Model

- **Backbone:** ResNet18 (pretrained on ImageNet)
- **Head:** `Linear(512→256) → ReLU → Dropout(0.2) → Linear(256→2)`
- **Input:** 224 × 224, 3-channel (greyscale repeated), ImageNet-normalised
- **Output:** `[force_magnitude, force_angle]`
- **Loss:** `MSE(force) + MSE(angle)`

### Training phases

| Phase | Backbone | Optimizer | LR | Early stopping |
|---|---|---|---|---|
| Warmup | Frozen | Adam | 1 × 10⁻⁴ | patience = 20 |
| Fine-tuning | Unfrozen | Adam | 1 × 10⁻⁵ | patience = 20 |

Data is split **70 / 15 / 15** (train / val / test).

---

## Requirements

```
torch
torchvision
numpy
pandas
pillow
matplotlib
opencv-python
tqdm
```
