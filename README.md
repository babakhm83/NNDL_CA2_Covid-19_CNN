# COVID-19 Detection from Chest X-Rays: Reproducing an Efficient CNN Architecture

A reproduction and extension of a published CNN architecture for 3-class chest X-ray classification (Normal / Pneumonia / COVID-19), including a systematic ablation over learning-rate schedules, data augmentation strategies, and architectural variants, benchmarked against transfer-learning baselines (VGG16, MobileNetV2).

**Institution:** University of Tehran, School of Electrical and Computer Engineering
**Course:** Neural Networks and Deep Learning
**Author:** Babak Hosseini Mohtasham
**Full report:** [`CA2_Q1_report.pdf`](./CA2_Q1_report.pdf)
**Assignment specification:** [`NNDL_HW2.pdf`](./NNDL_HW2.pdf)
**Reference paper:** [`Complexity - 2021 - Reshi - An Efficient CNN Model for COVID‐19 Disease Detection Based on X‐Ray Image Classification.pdf`](<./Complexity - 2021 - Reshi - An Efficient CNN Model for COVID‐19 Disease Detection Based on X‐Ray Image Classification.pdf>) *(Reshi et al., Complexity 2021)*

---

## Overview

This project reproduces the CNN architecture from Reshi et al.'s "An Efficient CNN Model for COVID-19 Disease Detection Based on X-Ray Image Classification," then goes beyond a direct replication by systematically testing which training choices — learning-rate schedule, augmentation strategy, input color/resolution, and network width — actually drive performance on this task, and by comparing the reproduced architecture against transfer-learning baselines built on ImageNet-pretrained backbones.

**Objectives:**

1. Reproduce the paper's CNN architecture (a six-layer convolutional stack with the paper's specified filter widths, pooling, and batch normalization) for 3-class chest X-ray classification.
2. Implement class-balanced data loading and data augmentation (translation and rotation), including a variant that follows the paper's own augmentation strategy (horizontal flip + 90°/180°/270° rotations).
3. Systematically compare constant learning rates and multiple learning-rate schedules (cosine decay, exponential decay, cosine decay with warm restarts) to identify the most effective training configuration.
4. Explore architectural and preprocessing variants intended to improve on the base result: grayscale input, doubled convolutional filter counts, a decoder/upsampling variant, and higher input resolution.
5. Evaluate the best configuration on a held-out test set, and compare it against transfer-learning baselines using frozen, ImageNet-pretrained VGG16 and MobileNetV2 backbones.

## Methodology

| Component | Description |
|---|---|
| **Dataset** | Chest X-ray images spanning three classes — Normal, Pneumonia, COVID-19 — loaded from Kaggle (`prashant268/chest-xray-covid19-pneumonia`), class-balanced by subsampling to the smallest class count |
| **Base CNN architecture** | Six convolutional blocks with filter widths (64, 64, 128, 128, 256, 256), each followed by max pooling and batch normalization, following the reference paper's specified configuration |
| **Augmentation** | Random translation and rotation (Keras preprocessing layers), 5× oversampling factor; a separate variant reproduces the paper's own augmentation scheme (horizontal flip plus 90°/180°/270° rotations) |
| **Learning-rate study** | Constant rates (1e-4 to 1e-1) compared against cosine decay, exponential decay, and cosine decay with warm restarts, each swept over multiple hyperparameter settings |
| **Architectural variants** | Grayscale input, doubled filter counts, a decoder/upsampling block, and increased input resolution (300×300 vs. the paper's 150×150), each trained and evaluated independently |
| **Transfer-learning baselines** | VGG16 and MobileNetV2, both ImageNet-pretrained with frozen backbones and a new fully-connected classification head, trained under the same protocol for comparison |

## Repository Structure

| File | Description |
|---|---|
| [`NNDL_CA2_1.ipynb`](./NNDL_CA2_1.ipynb) | Complete implementation: dataset loading and balancing, data augmentation, the base CNN and its learning-rate/architectural ablations, final test-set evaluation, and the VGG16/MobileNetV2 baselines. All cells were executed and outputs preserved. |
| [`CA2_Q1_report.pdf`](./CA2_Q1_report.pdf) | Written report: methodology, results, and discussion |
| [`NNDL_HW2.pdf`](./NNDL_HW2.pdf) | Original assignment specification |
| [`Complexity - 2021 - Reshi - An Efficient CNN Model for COVID‐19 Disease Detection Based on X‐Ray Image Classification.pdf`](<./Complexity - 2021 - Reshi - An Efficient CNN Model for COVID‐19 Disease Detection Based on X‐Ray Image Classification.pdf>) | Reference paper: the CNN architecture this project reproduces and extends |

### Notebook outline

1. Load the dataset — class-balanced loading, class distribution inspection, image inspection
2. Data augmentation — translation/rotation augmentation, plus a paper-matching augmentation variant
3. Train the base CNN without augmentation, as a reference point
4. Constant learning rates — sweep over four fixed learning rates
5. Variable learning rates — cosine decay, exponential decay, and cosine decay with warm restarts
6. Train with the paper's own augmentation strategy
7. Improve performance — grayscale input, wider filters, a decoder variant, higher resolution
8. Final evaluation on the held-out test set
9. Transfer learning baselines — VGG16 and MobileNetV2

## Key Results

**The reproduced CNN reached 96.55% accuracy on the held-out test set**, using grayscale input, the constant learning rate of 0.01 identified as best in the ablation study, and the paper's augmentation strategy — successfully reproducing (and slightly refining) the reference paper's reported performance range for this architecture.

**Learning-rate schedule comparison (validation accuracy, augmented training data):**

| Configuration | Accuracy |
|---|---|
| No augmentation (reference point) | 96.07% |
| Constant lr = 1e-4 | 92.55% |
| Constant lr = 1e-3 | 96.27% |
| **Constant lr = 1e-2** | **96.48%** |
| Constant lr = 1e-1 | 94.41% |
| Cosine decay (α=0) | 87.16% |
| Exponential decay (rate=0.9) | 96.07% |
| Cosine decay with warm restarts | 90.06% |

A constant learning rate of 1e-2 outperformed every learning-rate schedule tested, including cosine decay and warm restarts — a useful, somewhat counterintuitive finding, since scheduled decay is often assumed to help, but did not on this task and architecture.

**Architectural variants gave modest, mixed improvements over the augmented baseline:** grayscale input reached 94.6% validation accuracy, doubled filter width 93.8%, the decoder/upsampling variant 94.8%, and the paper's own augmentation scheme combined with the base architecture reached 96.6% — ultimately the configuration carried forward to final test-set evaluation.

**The reproduced CNN outperformed both transfer-learning baselines on the held-out test set:**

| Model | Test Accuracy |
|---|---|
| **Reproduced CNN (paper architecture)** | **96.55%** |
| VGG16 (ImageNet-pretrained, frozen) | 95.69% |
| MobileNetV2 (ImageNet-pretrained, frozen) | 94.83% |

This is a notable result: a comparatively small, task-specific CNN trained from scratch outperformed two much larger ImageNet-pretrained backbones used as frozen feature extractors, suggesting the domain gap between natural images and chest X-rays limits how much generic ImageNet features transfer to this task without backbone fine-tuning.

## Reproducing the Results

The notebook downloads its dataset via the Kaggle API and was developed for Google Colab.

1. Open [`NNDL_CA2_1.ipynb`](./NNDL_CA2_1.ipynb) in Google Colab or Jupyter.
2. Configure Kaggle API credentials (`kaggle.json`) to download the `prashant268/chest-xray-covid19-pneumonia` dataset, fetched automatically within the notebook.
3. A GPU runtime is strongly recommended — the notebook trains well over a dozen model configurations across the learning-rate sweep, architectural ablations, and transfer-learning baselines, most for up to 100 epochs with early stopping.
4. All outputs (training curves, learning-rate schedule plots, augmented-image visualizations, and full evaluation metrics for every configuration) are already preserved in the notebook and can be reviewed directly on GitHub without re-execution.

## Notes on Scope

- Class balancing is performed by subsampling all classes down to the smallest class's image count, prioritizing a balanced evaluation over using the full dataset.
- The learning-rate and architectural ablations are conducted on a validation split; only the single best-performing configuration from that process is retrained and evaluated on the held-out test set, to avoid repeatedly touching test data during model selection.
- VGG16 and MobileNetV2 baselines use frozen, ImageNet-pretrained backbones with only a new classification head trained; full fine-tuning of these backbones was not explored and might change the comparison with the from-scratch CNN.
