# Comparing Attention, CNNs, and SSMs for Multi-Domain Image Recognition


## Abstract
Multi-domain image recognition presents a rich and challenging area for computer vision research, with applications ranging from natural disaster detection to medical and agricultural diagnostics and beyond. However, deploying such detection systems in resource-constrained environments has become increasingly challenging due to the advent of ever-deeper Convolutional Neural Network (CNN) architectures and resource-intensive Transformer architectures with their underlying attention mechanism. As state-of-the-art image classification becomes ever more resource-hungry, the search for efficient alternatives has gained momentum. State-space models (SSMs) have emerged as a viable lightweight alternative to Transformers, though their potential in vision tasks remains unexplored. In this research, we conduct a comparative, performance-focused, study of a leading SSM-based vision backbone, Vision Mamba (ViM), against well-established CNN and Transformer-based backbones. We evaluate their performance across three diverse tasks: wildfire detection, plant disease identification, and skin cancer diagnosis. Our results highlight that while CNN models consistently achieved higher accuracy, ViM models consume significantly less VRAM. Moreover, the ViM Tiny model(7.06M) matched the accuracy of the 12×larger DeiT Base model(85.80M) in the wildfire detection task. The ViM models also converged faster than their counterparts in the same parameter range for all tasks. Furthermore, knowledge distillation on the Vim Tiny model boosted its accuracy from 70.6% to 85.32% in the wildfire detection task, closely matching the accuracy of the best-performing EfficientNet-B7 (66M). These findings underscore the potential of SSM-based models for producing lightweight and performant vision systems for resource-constrained deployments.

**Keywords: State-Space Models, Vision Mamba, Mamba, Knowledge Distillation, Multi-Domain Applications, Attention-Based Models, and Convolutional Models**

## Datasets

The following publicly available datasets were used in this research to train and evaluate CNN, Transformer, and SSM-based models across multiple domains:

---

### 1. 🔥 **FLAME Dataset: Aerial Imagery for Wildfire Detection**
- **Description**: High-resolution aerial frames captured via UAVs for pile burn detection. The training set contains **39,375 labeled JPEG images** (~1.3 GB), and the test set contains **8,617 labeled images** (~5.2 GB), categorized into `fire` and `no-fire` folders.
- **Source**: [IEEE Dataport – FLAME Dataset](https://ieee-dataport.org/open-access/flame-dataset-aerial-imagery-pile-burn-detection-using-drones-uavs)
- **Usage**: Used for training, validation, and evaluation in **wildfire detection**.

---

### 2. 🩺 **Skin Cancer: Malignant vs. Benign**
- **Description**: A dermoscopic image dataset containing **2,637 training** and **660 test** images, labeled into `malignant` and `benign` folders. All images were resized to **224×224 pixels** for model compatibility.
- **Source**: [Kaggle – Skin Cancer Dataset](https://www.kaggle.com/datasets/fanconic/skin-cancer-malignant-vs-benign)
- **Usage**: Used for binary classification in **skin cancer detection**.

---

### 3. 🌿 **PlantVillage Dataset**
- **Description**: A large-scale plant disease image dataset with **15 different disease categories** across multiple crop species. All images were standardized to **224×224 resolution** and categorized by disease type.
- **Source**: [Kaggle – PlantVillage](https://www.kaggle.com/datasets/emmarex/plantdisease)
- **Usage**: Used for **multi-class classification** in plant disease detection tasks.


## Models Used

The following deep learning models were utilized in this study, grouped into three categories based on architectural paradigms:

| Convolutional Neural Networks (CNNs) | Transformer-Based          | State Space Models (SSMs)        |
|----------------------------------------|----------------------------------------|-------------------------------------|
| ResNet-18                              | ViT_B_16                                | VisionMamba-Tiny-Patch16           |
| ResNet-50                              | Swin Transformer                        | VisionMamba-Small-Patch16          |
| VGG-16                                  | PVT v2                                  | VisionMamba-Base-Patch16           |
| InceptionV3                            | DeiT-Base Distilled Patch16-224         |                                     |
| Xception                               | ConvNext Base                           |                                     |
| MobileNetV2                            | ConvNextv2 Base                         |                                     |
| EfficientNet B0                        |                                          |                                     |
| EfficientNet B7                        |                                          |                                     |

> ✅ *All models were fine-tuned and evaluated on the same datasets to ensure fair comparison across architectural types.*

## Evaluation Metrics

To assess model performance across multiple domains, we used the following metrics:

### 🔹 Performance Metrics
- **Accuracy** – Overall correctness of the model.
- **Precision** – How many of the predicted positives were actually correct.
- **Recall** – How many of the actual positives were correctly predicted.
- **F1-Score** – Harmonic mean of Precision and Recall.
- **AUC-ROC** – Measures model's ability to distinguish between classes.
- **Top-5 Accuracy** *(for PlantVillage)* – Measures whether the correct label is within the top 5 predictions.

### 🔹 Computational Efficiency Metrics
- **FLOPs (GFLOPs)** – Model complexity in terms of floating-point operations.
- **Total Parameters** – Number of parameters in the model.
- **Trainable Parameters** – Parameters updated during training.
- **Inference Time** – Time taken to predict one image.
- **Throughput (FPS)** – Number of images processed per second.
- **GPU Memory Utilization** – Peak memory used during inference.


### Mamba Training Details
The training and testing of the Mamba model are thoroughly documented in a separate repository. You can find it [here](https://github.com/avonoeia/VisionMamba)


