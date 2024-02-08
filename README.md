# Vislocas
In this work, we developed Vislocas, which identifies potential protein mis-localization events from IHC images, to mark different cancer subtypes. Vislocas combines CNN and vision transformer to capture IHC image features at both global and local levels. Vislocas can be trained from scratch to create an end-to-end system that can identify protein subcellular localizations directly from IHC images.
## 1. Platform and Dependency
### 1.1 Platform
* Ubuntu 9.4.0-1ubuntu1~20.04.1
* RTX 2080 Ti(11GB) * 6
### 1.2 Dependency
|Requirements|Release|
|----|----|
|CUDA|11.3|
|Python|3.8.15|
|pytorch|1.11.0|
|torchvision|0.12.0|
|torchaudio|0.11.0|
|cudatoolkit|11.3|
|pandas|1.2.4|
|fvcore|0.1.5|
|opencv-python|4.6.0.66|
|timm|0.6.12|
|scipy|1.9.3|
|einops|0.6.0|
|matplotlib|3.5.1|
|scikit-learn|1.1.2|
|tensorboard|2.11.0|
|adabelief-pytorch|0.2.0|
## 2. Project Catalog Structure
### 2.1 datasets
> This folder stores the code files for data loading.
* ihc.py
    > This file includes ihc data loading code.
* build.py
    > This file includes building dataset code.
* loader.py
    > This file includes constructing loader code.
