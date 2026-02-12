# 🖼️ SRGAN Super-Resolution with Transfer Learning (ResNet-101)

This project implements **Super Resolution Generative Adversarial Network (SRGAN)** combined with **Transfer Learning using ResNet-101** to enhance the spatial resolution of remote sensing imagery.

The system is developed as part of an undergraduate thesis research focusing on improving Landsat 8 satellite imagery resolution.

---

## 📌 Project Overview

This research applies SRGAN to upscale low-resolution remote sensing images by **4× spatial resolution**.

Although the model successfully increases image resolution size, experimental results show:

- PSNR values decreased compared to original images
- Visual output still contains noticeable noise
- Transfer learning from ResNet-101 was not fully optimal
- Spatial resolution successfully increased 4× from the original image

---

## 🧠 Methodology

- Dataset: Landsat 8 OLI imagery  
- Preprocessing: QGIS  
- Model: SRGAN Generator  
- Transfer Learning Backbone: ResNet-101  
- Evaluation Metrics:
  - PSNR (Peak Signal-to-Noise Ratio)
  - MSE (Mean Squared Error)
  - MAE (Mean Absolute Error)
- Visual analysis comparison

---

## 🏗️ Model Architecture

- 8 Residual Blocks
- 64 Feature Channels
- PixelShuffle Upsampling (4×)
- Generator-based Super Resolution

---
## 🚀 How to Run Locally

1. Clone this repository:

```bash
git clone https://github.com/YOUR_USERNAME/srgan-super-resolution.git
cd srgan-super-resolution
