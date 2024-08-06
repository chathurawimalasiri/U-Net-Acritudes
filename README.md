# U-Net-Architecture
This repository contains different types of U-Net architectures.

## Overview

The U-Net architecture is a convolutional network designed for biomedical image segmentation. The key idea behind U-Net is to supplement a usual contracting network by successive layers, where pooling operations are replaced by upsampling operators. This leads to an architecture that is more symmetric and allows for more precise localization.

In this repository, we provide implementations of the following U-Net architectures:

1. **U-Net**: The original U-Net architecture.
2. **Attention U-Net**: An enhanced U-Net with attention mechanisms to focus on important features.
3. **UNet++**: A nested U-Net architecture with dense skip connections to capture fine details.
4. **Graph U-Net**: A U-Net variant utilizing graph neural networks.
