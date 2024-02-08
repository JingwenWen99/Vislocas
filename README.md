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
### 2.2 prepareData
> This folder stores the code files for data preparing.
  #### 2.2.1 IF
    > This folder stores the code files used to analyse and process the IF labels. 
  #### 2.2.2 IHC
    > This folder stores the code files used to analyse, process and download the IHC data. 
  #### 2.2.3 pathology
    > This folder stores the code files used to analyse, process and download the pathology data. 
  #### 2.2.4 GraphLoc
    > This folder stores the code files used to analyse, process and download the GraphLoc benchmarking dataset. 
  #### 2.2.5 MSTLoc
    > This folder stores the code files used to analyse, process and download the MSTLoc benchmarking dataset. 
  #### 2.2.6 laceDNN
    > This folder stores the code files used to analyse, process and download the laceDNN benchmarking dataset. 
### 2.3 data
> Download and save the data annotation information to this folder.
### 2.4 models
> This folder stores model-related code files, including Visloacas model code, loss function code, and model training-related code.
* cvr_utils
    > This folder stores convolutional tokenizer and transformer encoder module code.
* cct.py
    > This file includes Visloacas model code.
* classifier_model.py
    > This file includes load model code.
* train_classifier.py
    > This file includes model training-related code.
* loss.py
    > This file includes loss function code.
* criterion.py
    > This file includes criterion-related code.
### 2.5 tools
> This folder stores code files for model training, prediction, biomarker prediction, etc.
* train.py
    > This file includes model training code.
* test.py
    > This file includes model testing code.
* multi-instance.py
    > This file includes the code that aggregates the image-level results into protein-level results.
* cancerTest.py
    > This file includes the codes for screening biomarkers of cancer subtypes.
* cal_metrics.py
    > This file includes the code that calculates the performance metrics.
### 2.6 utils
> This folder stores the optimiser, scheduler, checkpoint and other utilities code files.
* checkpoint.py
    > This file includes checkpoint code.
* config_defaults.py
    > This file includes the parameter configuration code.
* distributed.py
    > This file includes the code for distributed training.
* eval_metrics.py
    > This file includes the utilities code for calculating the performance metrics.
* optimizer.py
    > This file includes the optimizer code.
* scheduler.py
    > This file includes the scheduler code.
