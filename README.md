# TPE Predict Force

This repository contains code for training a ResNet18-based regression model that predicts **contact force magnitude and angle** from grayscale particle contact-region images, like the following image:

 ![Demo results](sample_image.png)

To train the model, we generated multiple synthetic images of photoelastic disks subject to randomly generated vector forces at random contact points. Note that the force vectors are assigned such that each disk is in equilibrium. We then crop out the contact regions from these synthetic images and use them as training data, with the known force magnitudes and angles as labels. The trained model can then be applied to real experimental images to estimate the underlying contact forces.

Crucially, the method outperforms the traditional G^2 approximation in wet disks, leading to better results in the ensuing force inversion process.

<table><tr>
<td><img src="G2_results.png" alt="G2 results"/></td>
<td><img src="CNN_results.png" alt="CNN results"/></td>
</tr></table>

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

## Demo results

Predicted vs. true values on the held-out test set after fine-tuning:

![Demo results](demo_results.png)

---

## Requirements

```
pip install -r requirements.txt
```

| Package | Version |
|---|---|
| torch | 2.4.1 |
| torchvision | 0.19.1 |
| numpy | 1.23.5 |
| pandas | 1.4.1 |
| Pillow | 9.4.0 |
| matplotlib | 3.7.1 |
| opencv-python | 4.6.0.66 |
| tqdm | 4.65.0 |
