### Download and save the data annotation information to this folder.  
### All benchmark data has been deposited at Zenode (<https://doi.org/10.5281/zenodo.10632698>) [![DOI](<https://zenodo.org/badge/DOI/10.5281/zenodo.10632698.svg>)](<https://doi.org/10.5281/zenodo.10632698>)

#### 1. The following files should be placed directly in the `data` directory.  
|File|Descriptrion|
|----|----|
|data1.csv|The whole benchmark dataset|
|data_train.csv|The full training set|
|data_test.csv|The independent set|
|data_train_split<sub>i</sub>_fold<sub>j</sub>.csv|The training set for the jth-fold cross-validation of the i-th division|
|data_val_split<sub>i</sub>_fold<sub>j</sub>.csv|The validation set for the jth-fold cross-validation of the i-th division|
|data_IHC_analysis.csv|Data statistics for the benchmark data set|
|annotations.csv|Location annotation information for IF images|
|tissueUrl.csv|All normal IHC images and their URLs in the HPA database.|
|pathologyUrl.csv|All pathology IHC images and their URLs in the HPA database.|
|normalWithAnnotation.csv|IHC data information with matching IF labels|
|normalLabeled.csv|IHC data with labels|

#### 2. The following files should be placed directly in the `data/cancer` directory.  
|File|Descriptrion|
|----|----|
|normalGlioma.csv|The normal data for glioma|
|patholotyGlioma.csv|The patholoty data for glioma|
|normalMelanoma.csv|The normal data for melanoma|
|patholotyMelanoma.csv|The patholoty data for melanoma|
|normalSkinCancer.csv|The normal data for skin cancer|
|patholotySkinCancer.csv|The patholoty data for skin cancer|
|screenedNormalData.csv|Normal data filtered to the same image quality as the benchmark dataset|
|screenedPathologyData.csv|Pathology tissue data filtered to the same image quality as the benchmark dataset|
