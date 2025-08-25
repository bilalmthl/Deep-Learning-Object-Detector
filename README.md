# 🍎🍌🍊 Fruit Object Detector (CNN + Raspberry Pi)

A **Convolutional Neural Network (CNN)** built with **PyTorch** to classify fruits (apples, bananas, oranges, none) using images captured by a **Raspberry Pi Camera Module**.  
The model was trained on diverse conditions (lighting, angle, ripeness, cluttered backgrounds) and achieved **98% test accuracy**.  
Deployment is handled via a **Flask REST API** running on the Raspberry Pi.

---

## 🚀 Features
- **Data Collection**: Raspberry Pi Camera for capturing training/test datasets.
- **Training**: CUDA-accelerated ResNet18 CNN training on an **NVIDIA RTX 3060** using PyTorch.
- **Robust Generalization**: Data augmentation for lighting, orientation, and cluttered scenes.
- **Deployment**: Flask API for real-time predictions; integrates with Pi LCD display.
- **Test Accuracy**: 98% across validation datasets.
