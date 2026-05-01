# Satellite Image Segmentation using U-Net (ResNet34)

## Overview

This project implements a multi-class semantic segmentation pipeline for satellite imagery using a U-Net architecture with a ResNet34 encoder. The model segments images into classes such as buildings, land, roads, vegetation, water, and unlabeled regions.

The entire pipeline is built from scratch in PyTorch, covering data preprocessing, patch generation, label encoding, model training, evaluation, and visualization.

---

## Pipeline

### 1. Data Preparation

* Loaded Dubai satellite dataset (tile-based structure)
* Extracted image–mask pairs
* Generated fixed-size patches (256×256) from large images
* Created ~1300 training samples

### 2. Label Processing

* Converted RGB mask values into class labels
* Total classes: 6

  * Water
  * Land
  * Road
  * Building
  * Vegetation
  * Unlabeled

### 3. Data Augmentation

* Horizontal & vertical flips
* Random rotation
* Brightness/contrast adjustment
* Noise & blur
* Normalization using ImageNet statistics

---

## Model Architecture

* U-Net with ResNet34 encoder
* Encoder pretrained on ImageNet
* Input channels: 3
* Output classes: 6

---

## Loss Function

Combination of:

* Cross-Entropy Loss (stability)
* Dice Loss (improves segmentation overlap)

---

## Training Strategy

* Optimizer: Adam
* Learning rate: 1e-4
* Weight decay: 1e-5
* Learning rate scheduler: ReduceLROnPlateau
* Trained for 20–25 epochs
* Best model saved based on validation loss

---

## Results

* Mean IoU: **0.62**
* Dice Score: **0.72**

---

## Sample Outputs

The model demonstrates strong segmentation performance across different land cover types with good alignment to ground truth.

### Example 1

![Result](images/result1.png)

### Example 2

![Result](images/result2.png)

### Example 3

![Result](images/result3.png)

---

## Key Learnings

* Patch-based training improves performance on large satellite images
* Combining Dice + CrossEntropy improves segmentation quality
* Data augmentation significantly reduces overfitting
* Learning rate scheduling stabilizes training

---

## Tech Stack

* Python
* PyTorch
* segmentation_models_pytorch
* Albumentations
* OpenCV
* Matplotlib

---

## Future Improvements

* Deploy model using Hugging Face Spaces
* Improve accuracy using overlapping patches
* Experiment with deeper encoders (ResNet50, EfficientNet)
* Apply class-weighted loss for class imbalance

---
