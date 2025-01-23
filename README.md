# Comparative Analysis of Attention-Based, Convolutional, and SSM-Based Models for Multi-Domain Image Classification


## Abstract
The increasing frequency and severity of environmental and societal challenges, such as natural disasters, medical diagnostics, and agricultural threats, require development of efficient and scalable detection and classification systems. Autonomous technologies with real-time processing capabilities offer significant potential but face challenges like network delays, adverse conditions, and bandwidth limitations for transporting image or video data. Lightweight and fast models deployed on edge devices, such as surveillance drones, portable diagnostic tools, or agricultural sensors, can address these constraints effectively. Most modern-day State-Of-The-Art (SOTA) object detection and classification architectures are primarily based on Transformers and its underlying concept of attention, Unfortunately, this significant improvement in accuracy comes at the cost of a substantial increase in computation. Recently, State-space models (SSMs) have emerged as a promising alternative to transformer-based architectures with their enhanced computational efficiency and reasoning capabilities. The application of SSMs have potential in areas where long-range dependence on data is crucial but computational efficiency is particularly important. This research explores the application of SSMs, particularly Vision Mamba, in diverse domains: wildfire detection, plant disease identification, and skin cancer diagnostics. Furthermore, empirical comparisons are made with Attention-Based and Convolutional models in these domains to better evaluate the capabilities of SSM-based Vision backbones. Finally, the feasibility of knowledge distillation in Vision Mamba is examined using the information gathered from the thorough comparisons. This research demonstrates that Mamba models achieve promising accuracy, comparable to some Transformer-based models, while operating at sub-quadratic time complexity. In the context of wildfire detection, knowledge distillation further improved VisionMamba Tiny’s accuracy from 70.6% to 85.32%, highlighting its potential for lightweight, high-performance applications in critical scenarios.

## Datasets
The following datasets were used in this research:

1. **The FLAME dataset: Aerial Imagery Pile burn detection using drones (UAVs)**
   - **Description**: The training dataset consists of 39,375 frames (1.3 GB) in JPEG format and some samples are shown in \ref{fig:flame-sample}. The test dataset consists of 8,617 frames (5.2 GB), also in JPEG format. Both the training and testing dataset images have been labeled in fire and no-fire folders.
   - **Source**: [https://ieee-dataport.org/open-access/flame-dataset-aerial-imagery-pile-burn-detection-using-drones-uavs]
   - **Usage**: Used for training, validating and testing wildfire detection

2. **Skin Cancer: Malignant vs. Benign**
   - **Description**:The training data comprises 2637 images while the test data contains 660 images. The training and test datasets have been labeled in malignant and benign folders and resized to 224 x 224 pixels. 
   - **Source**: [https://www.kaggle.com/datasets/fanconic/skin-cancer-malignant-vs-benign]
   - **Usage**: Used for training and testing skin cancer detection

3. **PlantVillage Dataset**
   - **Description**: Contains images of 15 different catgories of plant diseases. 
   - **Source**: [https://www.kaggle.com/datasets/emmarex/plantdisease]
   - **Usage**: Used for training, validating and testing plant disease classification
  

  ## Models Used
The following models were utilized in this research:

- DeiT-Base Distilled Patch16-224
- Swin Transformer
- PVT v2
- ViT_b_16
- ResNet18
- ResNet-50
- VGG16
- InceptionV3
- Xception
- MobileNetV2
- ConvNext Base
- ConvNextv2 Base
- EfficientNet B0
- EfficientNet B7
- VisionMamba-tiny-patch16
- VisionMamba-small-patch16
- VisionMamba-base-patch16


### Note
The training and testing of the Mamba model are thoroughly documented in a separate repository. You can find it [here](https://github.com/avonoeia/VisionMamba)


