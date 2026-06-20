# Satellite Image Segmentation using U-Net with EfficientNet-B3

## Overview

This project implements a multi-class semantic segmentation pipeline for high-resolution satellite imagery using a U-Net architecture with an EfficientNet-B3 encoder.

The model segments satellite images into six land-cover classes:

* Building
* Land
* Road
* Vegetation
* Water
* Unlabeled

The complete pipeline is developed in PyTorch and includes data preprocessing, patch generation, label encoding, augmentation, class imbalance handling, model training, evaluation, and visualization.

---

## Dataset

The project uses the Dubai Satellite Imagery Dataset.

### Patch Generation

To increase the number of training samples and preserve spatial details:

* Patch Size: 256 × 256
* Stride: 128 (overlapping patches)
* Total Patches Generated: 4464
* Training Patches: 3571
* Validation Patches: 893

---

## Data Preprocessing

### RGB Mask → Label Encoding

Satellite masks are provided as RGB images and converted into integer class labels.

| Class      | Label |
| ---------- | ----- |
| Building   | 0     |
| Land       | 1     |
| Road       | 2     |
| Vegetation | 3     |
| Water      | 4     |
| Unlabeled  | 5     |

---

## Data Augmentation

Albumentations is used to improve generalization.

* Horizontal Flip
* Vertical Flip
* Random Rotation (90°)
* Random Brightness & Contrast
* Gaussian Blur
* Gaussian Noise
* Random Gamma
* ImageNet Normalization

---

## Class Imbalance Analysis

Pixel distribution revealed a significant class imbalance.

| Class      | Percentage |
| ---------- | ---------- |
| Land       | 53.7%      |
| Building   | 13.3%      |
| Water      | 12.0%      |
| Vegetation | 10.4%      |
| Road       | 9.6%       |
| Unlabeled  | 1.0%       |

To address this issue:

* Computed inverse-frequency class weights
* Applied weighted CrossEntropy Loss
* Ignored unlabeled pixels during training

This significantly improved segmentation performance, especially for minority classes such as roads and buildings.

---

## Model Architecture

### U-Net + EfficientNet-B3

* Encoder: EfficientNet-B3
* Pretrained Weights: ImageNet
* Input Channels: 3
* Output Classes: 6

EfficientNet-B3 provides stronger feature extraction compared to the previous ResNet34 encoder.

---

## Loss Function

A hybrid loss function was used:

### Weighted CrossEntropy Loss

Handles class imbalance.

### Dice Loss

Improves overlap between predicted and ground-truth masks.

Final Loss:

Loss = 0.7 × CrossEntropy + 0.3 × Dice Loss

---

## Training Configuration

* Framework: PyTorch
* Optimizer: Adam
* Learning Rate: 1e-4
* Weight Decay: 1e-4
* Scheduler: ReduceLROnPlateau
* Batch Size: 16
* Epochs: 30
* Early Stopping Enabled

The best model is automatically saved based on validation loss.

---

## Evaluation Metrics

The model is evaluated using:

### Intersection over Union (IoU)

IoU = TP / (TP + FP + FN)

### Dice Score

Dice = 2TP / (2TP + FP + FN)

Per-class metrics are calculated by accumulating:

* True Positives (TP)
* False Positives (FP)
* False Negatives (FN)

---

## Results

### Per-Class Performance

| Class      | IoU    | Dice   |
| ---------- | ------ | ------ |
| Building   | 0.7933 | 0.8847 |
| Land       | 0.8742 | 0.9329 |
| Road       | 0.6990 | 0.8228 |
| Vegetation | 0.7728 | 0.8719 |
| Water      | 0.9479 | 0.9732 |

### Overall Performance

* Mean IoU: 0.8174
* Mean Dice Score: 0.8971

---

## Sample Predictions

<img width="1283" height="390" alt="image" src="https://github.com/user-attachments/assets/bf158081-d005-448e-8d71-792588a19a79" />
<img width="1283" height="390" alt="image" src="https://github.com/user-attachments/assets/602a22fd-4345-4357-a480-10daaa929e60" />
<img width="1283" height="390" alt="image" src="https://github.com/user-attachments/assets/32781efc-90d9-4d44-a059-8ba0c588f85d" />

---

## Tech Stack

* Python
* PyTorch
* segmentation_models_pytorch
* Albumentations
* NumPy
* OpenCV
* Matplotlib

---

## Future Work

* FastAPI Deployment
* Streamlit Web Interface
* Docker Containerization
* Hugging Face Spaces Deployment
* Real-time Satellite Image Segmentation

---

## Author

Manojit Das


Satellite Image Semantic Segmentation using Deep Learning
