import numpy as np
import pandas as pd

from PIL import Image


RNG_SEED = 0
np.random.seed(RNG_SEED)


dataDir = "data/"
cancerDir = "data/cancer/"
imageDir = "dataset/IHC/"
imageDir2 = "dataset/IHC/"
locationList = ['cytoplasm', 'cytoskeleton', 'endoplasmic reticulum', 'golgi apparatus', 'lysosomes', 'mitochondria',
                'nucleoli', 'nucleus', 'plasma membrane', 'vesicles']

subtypeList = ['Normal tissue, NOS',
                'Adenocarcinoma primary or metastatic', 'Adenocarcinoma, Low grade', 'Adenocarcinoma, Medium grade', 'Adenocarcinoma, High grade',
                'Adenocarcinoma, NOS', 'Adenocarcinoma, metastatic, NOS', 'Adenocarcinoma, uncertain malignant potential', 'Adenoma, NOS',
                'Carcinoid, malignant, NOS', 'Carcinoma, NOS', 'Carcinoma, Embryonal, NOS', 'Carcinoma, endometroid', 'Carcinoma, Hepatocellular, NOS', 'Carcinoma, metastatic, NOS',
                'Cystadenocarcinoma, mucinous, NOS', 'Cystadenocarcinoma, serous, NOS',
                'Glioma, malignant, NOS', 'Glioma, malignant, Low grade', 'Glioma, malignant, High grade',
                'Malignant melanoma, NOS', 'Malignant melanoma, Metastatic site',
                'Squamous cell carcinoma, NOS',
                'Squamous cell carcinoma, metastatic, NOS',
                'Urothelial carcinoma, NOS', 'Urothelial carcinoma, Low grade', 'Urothelial carcinoma, High grade']
cancerList = ['Breast cancer', 'Carcinoid', 'Cervical cancer', 'Colorectal cancer', 'Endometrial cancer', 'Glioma', 'Head and neck cancer',
            'Liver cancer', 'Lung cancer', 'Lymphoma', 'Melanoma', 'Ovarian cancer', 'Pancreatic cancer', 'Prostate cancer', 'Renal cancer',
            'Skin cancer', 'Stomach cancer', 'Testis cancer', 'Thyroid cancer', 'Urothelial cancer']
tissueList = ['Adipose tissue', 'Adrenal gland', 'Appendix', 'Bone marrow', 'Breast', 'Bronchus', 'Caudate', 'Cerebellum', 'Cerebral cortex',
            'Cervix', 'Colon', 'Duodenum', 'Endometrium 1', 'Endometrium 2', 'Epididymis', 'Esophagus', 'Fallopian tube', 'Gallbladder',
            'Heart muscle', 'Hippocampus', 'Kidney', 'Liver', 'Lung', 'Lymph node', 'Nasopharynx', 'Oral mucosa', 'Ovary', 'Pancreas',
            'Parathyroid gland', 'Placenta', 'Prostate', 'Rectum', 'Salivary gland', 'Seminal vesicle', 'Skeletal muscle', 'Skin 1', 'Skin 2',
            'Small intestine', 'Smooth muscle', 'Soft tissue 1', 'Soft tissue 2', 'Spleen', 'Stomach 1', 'Stomach 2', 'Testis', 'Thyroid gland',
            'Tonsil', 'Urinary bladder', 'Vagina']

LymphomaSubtypes = ["Malignant lymphoma, non-Hodgkin's type, NOS", "Malignant lymphoma, non-Hodgkin's type, Low grade",
            "Malignant lymphoma, non-Hodgkin's type, High grade", "Hodgkin's disease, NOS", "Hodgkin's lymphoma, nodular sclerosis"]
GliomaSubtypes = ["Glioma, malignant, NOS", "Glioma, malignant, Low grade", "Glioma, malignant, High grade", "Glioblastoma, NOS"]
ThyroidCancerSubtypes = ["Papillary adenocarcinoma, NOS", "Papillary adenoma metastatic", "Follicular adenoma carcinoma, NOS"]
BreastCancerSubtypes = ["Duct carcinoma", "Intraductal carcinoma, in situ", "Lobular carcinoma", "Lobular carcinoma, in situ"]
CervicalCancerSubtypes = ["Squamous cell carcinoma, NOS", "Adenocarcinoma, NOS", "Adenocarcinoma, Low grade"]
EndometrialCancerSubtypes = ["Adenocarcinoma, NOS", "Adenocarcinoma, metastatic, NOS"]
OvarianCancerSubtypes = ["Cystadenocarcinoma, serous, NOS", "Cystadenocarcinoma, mucinous, NOS", "Carcinoma, endometroid"]
ColorectalCancerSubtypes = ["Adenocarcinoma, NOS", "Colon", "Rectum"]
StomachCancerSubtypes = ["Adenocarcinoma, NOS", "Adenocarcinoma, High grade"]
RenalCancerSubtypes = ["Adenocarcinoma, NOS", "Inflammation, NOS", "Carcinoma, NOS", "Adenocarcinoma, uncertain malignant potential",
            "Neoplasm, malignant, NOS", "Carcinoid, malignant, NOS"]
UrothelialCancerSubtypes = ["Urothelial carcinoma, NOS", "Urothelial carcinoma, Low grade", "Urothelial carcinoma, High grade"]
LiverCancerSubtypes = ["Carcinoma, Hepatocellular, NOS", "Cholangiocarcinoma"]
TestisCancerSubtypes = ["Seminoma, NOS", "Carcinoma, Embryonal, NOS"]
ProstateCancerSubtypes = ["Adenocarcinoma, NOS", "Adenocarcinoma, Low grade", "Adenocarcinoma, Medium grade", "Adenocarcinoma, High grade"]
LungCancerSubtypes = ["Adenocarcinoma, NOS", "Adenocarcinoma, metastatic, NOS", "Adenocarcinoma primary or metastatic",
            "Squamous cell carcinoma, NOS", "Squamous cell carcinoma, metastatic, NOS"]
MelanomaSubtypes = ["Malignant melanoma, NOS", "Malignant melanoma, Metastatic site"]
SkinCancerSubtypes = ["Basal cell carcinoma", "BCC, low aggressive", "BCC, high aggressive",
            "Squamous cell carcinoma, NOS", "Squamous cell carcinoma in situ, NOS", "Squamous cell carcinoma, metastatic, NOS"]


""" ['Protein Name', 'Protein Id', 'Antibody Id', 'Tissue', 'Organ', 'Cell Type',
    'Sex', 'Age', 'Patient Id', 'Staining Level', 'Intensity Level', 'Quantity', 'Location', 'SnomedParameters', 'URL'] """
def pathologyAnalysis(PathologyData):

    print("Pathology数据条数：", len(PathologyData))
    protein_count = PathologyData['Protein Id'].value_counts(sort=False).rename('Protein Id count').rename_axis('Protein Id').reset_index()
    # print(protein_count)
    antibody_count = PathologyData['Antibody Id'].value_counts(sort=False).rename('Antibody Id count').rename_axis('Antibody Id').reset_index()
    # print(antibody_count)
    tissue_count = PathologyData['Tissue'].value_counts().rename('Cancer count').rename_axis('Cancer').reset_index()
    # print(tissue_count)
    organ_count = PathologyData['Organ'].value_counts().rename('Organ count').rename_axis('Organ').reset_index()
    # print(organ_count)
    cellType_count = PathologyData['Cell Type'].value_counts().rename('Cell Type count').rename_axis('Cell Type').reset_index()
    # print(cellType_count)
    sex_count = PathologyData['Sex'].value_counts().rename('Sex count').rename_axis('Sex').reset_index()
    # print(sex_count)
    age_count = PathologyData['Age'].value_counts(bins=range(0, 100, 5)).rename('Age count').rename_axis('Age').reset_index()
    # print(age_count)
    patient_count = PathologyData['Patient Id'].value_counts().rename('Patient Id count').rename_axis('Patient Id').reset_index()
    # print(patient_count)
    stainingLevel_count = PathologyData['Staining Level'].value_counts().rename('Staining Level count').rename_axis('Staining Level').reset_index()
    # print(stainingLevel_count)
    intensityLevel_count = PathologyData['Intensity Level'].value_counts().rename('Intensity Level count').rename_axis('Intensity Level').reset_index()
    # print(intensityLevel_count)
    quantity_count = PathologyData['Quantity'].value_counts().rename('Quantity count').rename_axis('Quantity').reset_index()
    # print(quantity_count)
    location_count = PathologyData['Location'].value_counts().rename('Location count').rename_axis('Location').reset_index()
    # print(location_count)
    snomedParameters_count = PathologyData['SnomedParameters'].value_counts().rename('SnomedParameters count').rename_axis('SnomedParameters').reset_index()
    # print(snomedParameters_count)
    snomed_count = PathologyData['SnomedParameters'].str.split(";", expand=True).stack().value_counts().rename('Snomed count').rename_axis('Snomed').reset_index()
    # print(snomed_count)

    analysis = pd.concat([protein_count, antibody_count, tissue_count, organ_count, cellType_count, stainingLevel_count, intensityLevel_count, quantity_count, location_count, sex_count, age_count, patient_count, snomedParameters_count, snomed_count], axis=1)
    print(analysis)

    return analysis


def cancerAnalysis(filePath):
    PathologyData = pd.read_csv(filePath, header=0, index_col=0)
    for cancer in cancerList:
        cancerData = PathologyData[PathologyData['Tissue'] == cancer]
        cancerAnalysis = pathologyAnalysis(cancerData)
        cancerAnalysis.to_csv(dataDir + cancer +  "Analysis.csv", index=True, mode='w')


def getDetectedData():
    pathologyFilePath = dataDir + "pathologyUrl.csv"
    normalFilePath = dataDir + "normalWithAnnotation.csv"
    normalData = pd.read_csv(normalFilePath, header=0, index_col=0)
    pathologyData = pd.read_csv(pathologyFilePath, header=0, index_col=0)
    print(normalData)
    print(pathologyData)

    normalData = normalData[normalData['Staining Level'].str.contains('high|medium|low')]
    pathologyData = pathologyData[pathologyData['Staining Level'].str.contains('High|Medium|Low')]

    normalProtein = normalData['Protein Id'].drop_duplicates()
    pathologyProtein = pathologyData['Protein Id'].drop_duplicates()
    intersection = pd.Series(list(set(normalProtein).intersection(set(pathologyProtein))))

    normalData = normalData[normalData['Protein Id'].isin(intersection)]
    pathologyData = pathologyData[pathologyData['Protein Id'].isin(intersection)]

    print(normalData)
    print(pathologyData)

    normalData.to_csv(dataDir + "detectedNormalData.csv", index=True, mode='w')
    pathologyData.to_csv(dataDir + "detectedPathologyData.csv", index=True, mode='w')


def deleteWrongData(dataPath, deletedDataPath, condition="normal"):
    data = pd.read_csv(dataPath, header=0, index_col=0)

    if condition == "normal":
        data0 = data[data['Organ'].isin(['Brain', 'Skin'])]
    else:
        data0 = data[data['Tissue'].isin(['Glioma', 'Melanoma', 'Skin cancer'])]

    wrongData = []
    for index, row in data0.iterrows():
        if row["Pair Idx"].split('-')[0] == "N":
            im_path = imageDir + "normal/" + row.URL
        elif row["Pair Idx"].split('-')[0] == "P":
            if index >= 5500000:
                im_path = imageDir + "pathology/" + row.URL
            else:
                im_path = imageDir2 + "pathology/" + row.URL
        im = Image.open(im_path)
        # print(index, im_path, len(im.split()))
        if index % 1000 == 0:
            print(index)
        if len(im.split()) != 3:
            print(index, im_path, len(im.split()))
            wrongData.append(index)

    print("wrongData:", wrongData)
    data = data.drop(wrongData, axis=0)
    print(data)
    data.to_csv(deletedDataPath, index=True, mode='w')


def screenData():
    pathologyFilePath = dataDir + "detectedPathologyData.csv"
    normalFilePath = dataDir + "detectedNormalData.csv"
    normalData = pd.read_csv(normalFilePath, header=0, index_col=0)
    pathologyData = pd.read_csv(pathologyFilePath, header=0, index_col=0)
    print(normalData)
    print(pathologyData)

    # data-6029
    normalData = normalData[normalData[locationList].sum(axis=1) > 0]
    normalData = normalData[(normalData['Intensity Level'].str.contains('strong')) &
        (normalData['Quantity'].str.contains(r'>75%'))]
    pathologyData = pathologyData[(pathologyData['Intensity Level'].str.contains('Strong')) &
        (pathologyData['Quantity'].str.contains(r'>75%'))]

    groups = normalData.groupby(by=['Protein Id', 'Tissue'])['Protein Name'].count()
    groups = groups[groups >= 3].reset_index()
    normalData = pd.merge(normalData.reset_index(), groups[['Protein Id', 'Tissue']], on=['Protein Id', 'Tissue'], how='right').set_index('index')

    groups = pathologyData.groupby(by=['Protein Id', 'Tissue'])['Protein Name'].count()
    groups = groups[groups >= 3].reset_index()
    pathologyData = pd.merge(pathologyData.reset_index(), groups[['Protein Id', 'Tissue']], on=['Protein Id', 'Tissue'], how='right').set_index('index')

    normalProtein = normalData['Protein Id'].drop_duplicates()
    pathologyProtein = pathologyData['Protein Id'].drop_duplicates()
    intersection = pd.Series(list(set(normalProtein).intersection(set(pathologyProtein))))

    normalData = normalData[normalData['Protein Id'].isin(intersection)]
    pathologyData = pathologyData[pathologyData['Protein Id'].isin(intersection)]

    print(normalData)
    print(pathologyData)

    normalData.to_csv(cancerDir + "screenedNormalData.csv", index=True, mode='w')
    pathologyData.to_csv(cancerDir + "screenedPathologyData.csv", index=True, mode='w')


def genLymphomaData(normalFilePath, pathologyFilePath):
    print("Lymphoma")
    normalData = pd.read_csv(normalFilePath, header=0, index_col=0)
    pathologyData = pd.read_csv(pathologyFilePath, header=0, index_col=0)

    normalData = normalData[normalData['Tissue'].isin(['Lymph node', 'Spleen', 'Tonsil'])]
    pathologyData = pathologyData[pathologyData['Tissue'].isin(['Lymphoma'])]

    normalProtein = normalData['Protein Id'].drop_duplicates()
    pathologyProtein = pathologyData['Protein Id'].drop_duplicates()
    intersection = pd.Series(list(set(normalProtein).intersection(set(pathologyProtein))))

    print(intersection)

    normalData = normalData[normalData['Protein Id'].isin(intersection)]
    pathologyData = pathologyData[pathologyData['Protein Id'].isin(intersection)]

    pathologyData[LymphomaSubtypes] = 0
    for subtype in LymphomaSubtypes:
        pathologyData.loc[pathologyData['SnomedParameters'].str.contains(subtype), subtype] = 1

    # print(normalData)
    # print(pathologyData)
    print(len(normalData))
    print(len(pathologyData))

    subtypes = pathologyData[LymphomaSubtypes].groupby(by=pathologyData['Protein Id']).max().sum()
    print(subtypes)
    print(pathologyData[LymphomaSubtypes].sum())

    normalData.to_csv(cancerDir + "normalLymphoma.csv", index=True, mode='w')
    pathologyData.to_csv(cancerDir + "pathologyLymphoma.csv", index=True, mode='w')


def genGliomaData(normalFilePath, pathologyFilePath):
    print("Glioma")
    normalData = pd.read_csv(normalFilePath, header=0, index_col=0)
    pathologyData = pd.read_csv(pathologyFilePath, header=0, index_col=0)

    normalData = normalData[normalData['Organ'].isin(['Brain'])]
    pathologyData = pathologyData[pathologyData['Tissue'].isin(['Glioma'])]

    normalProtein = normalData['Protein Id'].drop_duplicates()
    pathologyProtein = pathologyData['Protein Id'].drop_duplicates()
    intersection = pd.Series(list(set(normalProtein).intersection(set(pathologyProtein))))

    print(intersection)

    normalData = normalData[normalData['Protein Id'].isin(intersection)]
    pathologyData = pathologyData[pathologyData['Protein Id'].isin(intersection)]

    pathologyData[GliomaSubtypes] = 0
    for subtype in GliomaSubtypes:
        pathologyData.loc[pathologyData['SnomedParameters'].str.contains(subtype), subtype] = 1

    # print(normalData)
    # print(pathologyData)
    print(len(normalData))
    print(len(pathologyData))

    subtypes = pathologyData[GliomaSubtypes].groupby(by=pathologyData['Protein Id']).max().sum()
    print(subtypes)
    print(pathologyData[GliomaSubtypes].sum())

    normalData.to_csv(cancerDir + "normalGlioma.csv", index=True, mode='w')
    pathologyData.to_csv(cancerDir + "pathologyGlioma.csv", index=True, mode='w')


def genThyroidCancerData(normalFilePath, pathologyFilePath):
    print("ThyroidCancer")
    normalData = pd.read_csv(normalFilePath, header=0, index_col=0)
    pathologyData = pd.read_csv(pathologyFilePath, header=0, index_col=0)

    normalData = normalData[normalData['Tissue'].isin(['Thyroid gland'])]
    pathologyData = pathologyData[pathologyData['Tissue'].isin(['Thyroid cancer'])]

    normalProtein = normalData['Protein Id'].drop_duplicates()
    pathologyProtein = pathologyData['Protein Id'].drop_duplicates()
    intersection = pd.Series(list(set(normalProtein).intersection(set(pathologyProtein))))

    print(intersection)

    normalData = normalData[normalData['Protein Id'].isin(intersection)]
    pathologyData = pathologyData[pathologyData['Protein Id'].isin(intersection)]

    pathologyData[ThyroidCancerSubtypes] = 0
    for subtype in ThyroidCancerSubtypes:
        pathologyData.loc[pathologyData['SnomedParameters'].str.contains(subtype), subtype] = 1

    # print(normalData)
    # print(pathologyData)
    print(len(normalData))
    print(len(pathologyData))

    subtypes = pathologyData[ThyroidCancerSubtypes].groupby(by=pathologyData['Protein Id']).max().sum()
    print(subtypes)
    print(pathologyData[ThyroidCancerSubtypes].sum())

    normalData.to_csv(cancerDir + "normalThyroidCancer.csv", index=True, mode='w')
    pathologyData.to_csv(cancerDir + "pathologyThyroidCancer.csv", index=True, mode='w')


def genBreastCancerData(normalFilePath, pathologyFilePath):
    print("BreastCancer")
    normalData = pd.read_csv(normalFilePath, header=0, index_col=0)
    pathologyData = pd.read_csv(pathologyFilePath, header=0, index_col=0)

    normalData = normalData[normalData['Tissue'].isin(['Breast'])]
    pathologyData = pathologyData[pathologyData['Tissue'].isin(['Breast cancer'])]

    normalProtein = normalData['Protein Id'].drop_duplicates()
    pathologyProtein = pathologyData['Protein Id'].drop_duplicates()
    intersection = pd.Series(list(set(normalProtein).intersection(set(pathologyProtein))))

    print(intersection)

    normalData = normalData[normalData['Protein Id'].isin(intersection)]
    pathologyData = pathologyData[pathologyData['Protein Id'].isin(intersection)]

    pathologyData[BreastCancerSubtypes] = 0
    for subtype in BreastCancerSubtypes:
        pathologyData.loc[pathologyData['SnomedParameters'].str.contains(subtype), subtype] = 1

    # print(normalData)
    # print(pathologyData)
    print(len(normalData))
    print(len(pathologyData))

    subtypes = pathologyData[BreastCancerSubtypes].groupby(by=pathologyData['Protein Id']).max().sum()
    print(subtypes)
    print(pathologyData[BreastCancerSubtypes].sum())

    normalData.to_csv(cancerDir + "normalBreastCancer.csv", index=True, mode='w')
    pathologyData.to_csv(cancerDir + "pathologyBreastCancer.csv", index=True, mode='w')


def genCervicalCancerData(normalFilePath, pathologyFilePath):
    print("CervicalCancer")
    normalData = pd.read_csv(normalFilePath, header=0, index_col=0)
    pathologyData = pd.read_csv(pathologyFilePath, header=0, index_col=0)

    normalData = normalData[normalData['Tissue'].isin(['Cervix'])]
    pathologyData = pathologyData[pathologyData['Tissue'].isin(['Cervical cancer'])]

    normalProtein = normalData['Protein Id'].drop_duplicates()
    pathologyProtein = pathologyData['Protein Id'].drop_duplicates()
    intersection = pd.Series(list(set(normalProtein).intersection(set(pathologyProtein))))

    print(intersection)

    normalData = normalData[normalData['Protein Id'].isin(intersection)]
    pathologyData = pathologyData[pathologyData['Protein Id'].isin(intersection)]

    pathologyData[CervicalCancerSubtypes] = 0
    for subtype in CervicalCancerSubtypes:
        pathologyData.loc[pathologyData['SnomedParameters'].str.contains(subtype), subtype] = 1

    # print(normalData)
    # print(pathologyData)
    print(len(normalData))
    print(len(pathologyData))

    subtypes = pathologyData[CervicalCancerSubtypes].groupby(by=pathologyData['Protein Id']).max().sum()
    print(subtypes)
    print(pathologyData[CervicalCancerSubtypes].sum())

    normalData.to_csv(cancerDir + "normalCervicalCancer.csv", index=True, mode='w')
    pathologyData.to_csv(cancerDir + "pathologyCervicalCancer.csv", index=True, mode='w')


def genEndometrialCancerData(normalFilePath, pathologyFilePath):
    print("EndometrialCancer")
    normalData = pd.read_csv(normalFilePath, header=0, index_col=0)
    pathologyData = pd.read_csv(pathologyFilePath, header=0, index_col=0)

    normalData = normalData[normalData['Tissue'].isin(['Endometrium 1', 'Endometrium 2'])]
    pathologyData = pathologyData[pathologyData['Tissue'].isin(['Endometrial cancer'])]

    normalProtein = normalData['Protein Id'].drop_duplicates()
    pathologyProtein = pathologyData['Protein Id'].drop_duplicates()
    intersection = pd.Series(list(set(normalProtein).intersection(set(pathologyProtein))))

    print(intersection)

    normalData = normalData[normalData['Protein Id'].isin(intersection)]
    pathologyData = pathologyData[pathologyData['Protein Id'].isin(intersection)]

    pathologyData[EndometrialCancerSubtypes] = 0
    for subtype in EndometrialCancerSubtypes:
        pathologyData.loc[pathologyData['SnomedParameters'].str.contains(subtype), subtype] = 1

    # print(normalData)
    # print(pathologyData)
    print(len(normalData))
    print(len(pathologyData))

    subtypes = pathologyData[EndometrialCancerSubtypes].groupby(by=pathologyData['Protein Id']).max().sum()
    print(subtypes)
    print(pathologyData[EndometrialCancerSubtypes].sum())

    normalData.to_csv(cancerDir + "normalEndometrialCancer.csv", index=True, mode='w')
    pathologyData.to_csv(cancerDir + "pathologyEndometrialCancer.csv", index=True, mode='w')


def genOvarianCancerData(normalFilePath, pathologyFilePath):
    print("OvarianCancer")
    normalData = pd.read_csv(normalFilePath, header=0, index_col=0)
    pathologyData = pd.read_csv(pathologyFilePath, header=0, index_col=0)

    normalData = normalData[normalData['Tissue'].isin(['Ovary'])]
    pathologyData = pathologyData[pathologyData['Tissue'].isin(['Ovarian cancer'])]

    normalProtein = normalData['Protein Id'].drop_duplicates()
    pathologyProtein = pathologyData['Protein Id'].drop_duplicates()
    intersection = pd.Series(list(set(normalProtein).intersection(set(pathologyProtein))))

    print(intersection)

    normalData = normalData[normalData['Protein Id'].isin(intersection)]
    pathologyData = pathologyData[pathologyData['Protein Id'].isin(intersection)]

    pathologyData[OvarianCancerSubtypes] = 0
    for subtype in OvarianCancerSubtypes:
        pathologyData.loc[pathologyData['SnomedParameters'].str.contains(subtype), subtype] = 1

    # print(normalData)
    # print(pathologyData)
    print(len(normalData))
    print(len(pathologyData))

    subtypes = pathologyData[OvarianCancerSubtypes].groupby(by=pathologyData['Protein Id']).max().sum()
    print(subtypes)
    print(pathologyData[OvarianCancerSubtypes].sum())

    normalData.to_csv(cancerDir + "normalOvarianCancer.csv", index=True, mode='w')
    pathologyData.to_csv(cancerDir + "pathologyOvarianCancer.csv", index=True, mode='w')


def genColorectalCancerData(normalFilePath, pathologyFilePath):
    print("ColorectalCancer")
    normalData = pd.read_csv(normalFilePath, header=0, index_col=0)
    pathologyData = pd.read_csv(pathologyFilePath, header=0, index_col=0)

    normalData = normalData[normalData['Tissue'].isin(['Colon', 'Rectum'])]
    pathologyData = pathologyData[pathologyData['Tissue'].isin(['Colorectal cancer'])]

    normalProtein = normalData['Protein Id'].drop_duplicates()
    pathologyProtein = pathologyData['Protein Id'].drop_duplicates()
    intersection = pd.Series(list(set(normalProtein).intersection(set(pathologyProtein))))

    print(intersection)

    normalData = normalData[normalData['Protein Id'].isin(intersection)]
    pathologyData = pathologyData[pathologyData['Protein Id'].isin(intersection)]

    pathologyData[ColorectalCancerSubtypes] = 0
    for subtype in ColorectalCancerSubtypes:
        pathologyData.loc[pathologyData['SnomedParameters'].str.contains(subtype), subtype] = 1

    # print(normalData)
    # print(pathologyData)
    print(len(normalData))
    print(len(pathologyData))

    subtypes = pathologyData[ColorectalCancerSubtypes].groupby(by=pathologyData['Protein Id']).max().sum()
    print(subtypes)
    print(pathologyData[ColorectalCancerSubtypes].sum())

    normalData.to_csv(cancerDir + "normalColorectalCancer.csv", index=True, mode='w')
    pathologyData.to_csv(cancerDir + "pathologyColorectalCancer.csv", index=True, mode='w')


def genStomachCancerData(normalFilePath, pathologyFilePath):
    print("StomachCancer")
    normalData = pd.read_csv(normalFilePath, header=0, index_col=0)
    pathologyData = pd.read_csv(pathologyFilePath, header=0, index_col=0)

    normalData = normalData[normalData['Tissue'].isin(['Stomach 1', 'Stomach 2'])]
    pathologyData = pathologyData[pathologyData['Tissue'].isin(['Stomach cancer'])]

    normalProtein = normalData['Protein Id'].drop_duplicates()
    pathologyProtein = pathologyData['Protein Id'].drop_duplicates()
    intersection = pd.Series(list(set(normalProtein).intersection(set(pathologyProtein))))

    print(intersection)

    normalData = normalData[normalData['Protein Id'].isin(intersection)]
    pathologyData = pathologyData[pathologyData['Protein Id'].isin(intersection)]

    pathologyData[StomachCancerSubtypes] = 0
    for subtype in StomachCancerSubtypes:
        pathologyData.loc[pathologyData['SnomedParameters'].str.contains(subtype), subtype] = 1

    # print(normalData)
    # print(pathologyData)
    print(len(normalData))
    print(len(pathologyData))

    subtypes = pathologyData[StomachCancerSubtypes].groupby(by=pathologyData['Protein Id']).max().sum()
    print(subtypes)
    print(pathologyData[StomachCancerSubtypes].sum())

    normalData.to_csv(cancerDir + "normalStomachCancer.csv", index=True, mode='w')
    pathologyData.to_csv(cancerDir + "pathologyStomachCancer.csv", index=True, mode='w')


def genRenalCancerData(normalFilePath, pathologyFilePath):
    print("RenalCancer")
    normalData = pd.read_csv(normalFilePath, header=0, index_col=0)
    pathologyData = pd.read_csv(pathologyFilePath, header=0, index_col=0)

    normalData = normalData[normalData['Tissue'].isin(['Kidney'])]
    pathologyData = pathologyData[pathologyData['Tissue'].isin(['Renal cancer'])]

    normalProtein = normalData['Protein Id'].drop_duplicates()
    pathologyProtein = pathologyData['Protein Id'].drop_duplicates()
    intersection = pd.Series(list(set(normalProtein).intersection(set(pathologyProtein))))

    print(intersection)

    normalData = normalData[normalData['Protein Id'].isin(intersection)]
    pathologyData = pathologyData[pathologyData['Protein Id'].isin(intersection)]

    pathologyData[RenalCancerSubtypes] = 0
    for subtype in RenalCancerSubtypes:
        pathologyData.loc[pathologyData['SnomedParameters'].str.contains(subtype), subtype] = 1

    # print(normalData)
    # print(pathologyData)
    print(len(normalData))
    print(len(pathologyData))

    subtypes = pathologyData[RenalCancerSubtypes].groupby(by=pathologyData['Protein Id']).max().sum()
    print(subtypes)
    print(pathologyData[RenalCancerSubtypes].sum())

    normalData.to_csv(cancerDir + "normalRenalCancer.csv", index=True, mode='w')
    pathologyData.to_csv(cancerDir + "pathologyRenalCancer.csv", index=True, mode='w')


def genUrothelialCancerData(normalFilePath, pathologyFilePath):
    print("UrothelialCancer")
    normalData = pd.read_csv(normalFilePath, header=0, index_col=0)
    pathologyData = pd.read_csv(pathologyFilePath, header=0, index_col=0)

    normalData = normalData[normalData['Tissue'].isin(['Urinary bladder'])]
    pathologyData = pathologyData[pathologyData['Tissue'].isin(['Urothelial cancer'])]

    normalProtein = normalData['Protein Id'].drop_duplicates()
    pathologyProtein = pathologyData['Protein Id'].drop_duplicates()
    intersection = pd.Series(list(set(normalProtein).intersection(set(pathologyProtein))))

    print(intersection)

    normalData = normalData[normalData['Protein Id'].isin(intersection)]
    pathologyData = pathologyData[pathologyData['Protein Id'].isin(intersection)]

    pathologyData[UrothelialCancerSubtypes] = 0
    for subtype in UrothelialCancerSubtypes:
        pathologyData.loc[pathologyData['SnomedParameters'].str.contains(subtype), subtype] = 1

    # print(normalData)
    # print(pathologyData)
    print(len(normalData))
    print(len(pathologyData))

    subtypes = pathologyData[UrothelialCancerSubtypes].groupby(by=pathologyData['Protein Id']).max().sum()
    print(subtypes)
    print(pathologyData[UrothelialCancerSubtypes].sum())

    normalData.to_csv(cancerDir + "normalUrothelialCancer.csv", index=True, mode='w')
    pathologyData.to_csv(cancerDir + "pathologyUrothelialCancer.csv", index=True, mode='w')


def genLiverCancerData(normalFilePath, pathologyFilePath):
    print("LiverCancer")
    normalData = pd.read_csv(normalFilePath, header=0, index_col=0)
    pathologyData = pd.read_csv(pathologyFilePath, header=0, index_col=0)

    normalData = normalData[normalData['Tissue'].isin(['Liver'])]
    pathologyData = pathologyData[pathologyData['Tissue'].isin(['Liver cancer'])]

    normalProtein = normalData['Protein Id'].drop_duplicates()
    pathologyProtein = pathologyData['Protein Id'].drop_duplicates()
    intersection = pd.Series(list(set(normalProtein).intersection(set(pathologyProtein))))

    print(intersection)

    normalData = normalData[normalData['Protein Id'].isin(intersection)]
    pathologyData = pathologyData[pathologyData['Protein Id'].isin(intersection)]

    pathologyData[LiverCancerSubtypes] = 0
    for subtype in LiverCancerSubtypes:
        pathologyData.loc[pathologyData['SnomedParameters'].str.contains(subtype), subtype] = 1

    # print(normalData)
    # print(pathologyData)
    print(len(normalData))
    print(len(pathologyData))

    subtypes = pathologyData[LiverCancerSubtypes].groupby(by=pathologyData['Protein Id']).max().sum()
    print(subtypes)
    print(pathologyData[LiverCancerSubtypes].sum())

    normalData.to_csv(cancerDir + "normalLiverCancer.csv", index=True, mode='w')
    pathologyData.to_csv(cancerDir + "pathologyLiverCancer.csv", index=True, mode='w')


def genTestisCancerData(normalFilePath, pathologyFilePath):
    print("TestisCancer")
    normalData = pd.read_csv(normalFilePath, header=0, index_col=0)
    pathologyData = pd.read_csv(pathologyFilePath, header=0, index_col=0)

    normalData = normalData[normalData['Tissue'].isin(['Testis'])]
    pathologyData = pathologyData[pathologyData['Tissue'].isin(['Testis cancer'])]

    normalProtein = normalData['Protein Id'].drop_duplicates()
    pathologyProtein = pathologyData['Protein Id'].drop_duplicates()
    intersection = pd.Series(list(set(normalProtein).intersection(set(pathologyProtein))))

    print(intersection)

    normalData = normalData[normalData['Protein Id'].isin(intersection)]
    pathologyData = pathologyData[pathologyData['Protein Id'].isin(intersection)]

    pathologyData[TestisCancerSubtypes] = 0
    for subtype in TestisCancerSubtypes:
        pathologyData.loc[pathologyData['SnomedParameters'].str.contains(subtype), subtype] = 1

    # print(normalData)
    # print(pathologyData)
    print(len(normalData))
    print(len(pathologyData))

    subtypes = pathologyData[TestisCancerSubtypes].groupby(by=pathologyData['Protein Id']).max().sum()
    print(subtypes)
    print(pathologyData[TestisCancerSubtypes].sum())

    normalData.to_csv(cancerDir + "normalTestisCancer.csv", index=True, mode='w')
    pathologyData.to_csv(cancerDir + "pathologyTestisCancer.csv", index=True, mode='w')


def genProstateCancerData(normalFilePath, pathologyFilePath):
    print("ProstateCancer")
    normalData = pd.read_csv(normalFilePath, header=0, index_col=0)
    pathologyData = pd.read_csv(pathologyFilePath, header=0, index_col=0)

    normalData = normalData[normalData['Tissue'].isin(['Prostate'])]
    pathologyData = pathologyData[pathologyData['Tissue'].isin(['Prostate cancer'])]

    normalProtein = normalData['Protein Id'].drop_duplicates()
    pathologyProtein = pathologyData['Protein Id'].drop_duplicates()
    intersection = pd.Series(list(set(normalProtein).intersection(set(pathologyProtein))))

    print(intersection)

    normalData = normalData[normalData['Protein Id'].isin(intersection)]
    pathologyData = pathologyData[pathologyData['Protein Id'].isin(intersection)]

    pathologyData[ProstateCancerSubtypes] = 0
    for subtype in ProstateCancerSubtypes:
        pathologyData.loc[pathologyData['SnomedParameters'].str.contains(subtype), subtype] = 1

    # print(normalData)
    # print(pathologyData)
    print(len(normalData))
    print(len(pathologyData))

    subtypes = pathologyData[ProstateCancerSubtypes].groupby(by=pathologyData['Protein Id']).max().sum()
    print(subtypes)
    print(pathologyData[ProstateCancerSubtypes].sum())

    normalData.to_csv(cancerDir + "normalProstateCancer.csv", index=True, mode='w')
    pathologyData.to_csv(cancerDir + "pathologyProstateCancer.csv", index=True, mode='w')


def genLungCancerData(normalFilePath, pathologyFilePath):
    print("LungCancer")
    normalData = pd.read_csv(normalFilePath, header=0, index_col=0)
    pathologyData = pd.read_csv(pathologyFilePath, header=0, index_col=0)

    normalData = normalData[normalData['Tissue'].isin(['Lung'])]
    pathologyData = pathologyData[pathologyData['Tissue'].isin(['Lung cancer'])]

    normalProtein = normalData['Protein Id'].drop_duplicates()
    pathologyProtein = pathologyData['Protein Id'].drop_duplicates()
    intersection = pd.Series(list(set(normalProtein).intersection(set(pathologyProtein))))

    print(intersection)

    normalData = normalData[normalData['Protein Id'].isin(intersection)]
    pathologyData = pathologyData[pathologyData['Protein Id'].isin(intersection)]

    pathologyData[LungCancerSubtypes] = 0
    for subtype in LungCancerSubtypes:
        pathologyData.loc[pathologyData['SnomedParameters'].str.contains(subtype), subtype] = 1

    # print(normalData)
    # print(pathologyData)
    print(len(normalData))
    print(len(pathologyData))

    subtypes = pathologyData[LungCancerSubtypes].groupby(by=pathologyData['Protein Id']).max().sum()
    print(subtypes)
    print(pathologyData[LungCancerSubtypes].sum())

    normalData.to_csv(cancerDir + "normalLungCancer.csv", index=True, mode='w')
    pathologyData.to_csv(cancerDir + "pathologyLungCancer.csv", index=True, mode='w')


def genMelanomaData(normalFilePath, pathologyFilePath):
    print("Melanoma")
    normalData = pd.read_csv(normalFilePath, header=0, index_col=0)
    pathologyData = pd.read_csv(pathologyFilePath, header=0, index_col=0)

    normalData = normalData[normalData['Organ'].isin(['Skin'])]
    pathologyData = pathologyData[pathologyData['Tissue'].isin(['Melanoma'])]

    normalProtein = normalData['Protein Id'].drop_duplicates()
    pathologyProtein = pathologyData['Protein Id'].drop_duplicates()
    intersection = pd.Series(list(set(normalProtein).intersection(set(pathologyProtein))))

    print(intersection)

    normalData = normalData[normalData['Protein Id'].isin(intersection)]
    pathologyData = pathologyData[pathologyData['Protein Id'].isin(intersection)]

    pathologyData[MelanomaSubtypes] = 0
    for subtype in MelanomaSubtypes:
        pathologyData.loc[pathologyData['SnomedParameters'].str.contains(subtype), subtype] = 1

    # print(normalData)
    # print(pathologyData)
    print(len(normalData))
    print(len(pathologyData))

    subtypes = pathologyData[MelanomaSubtypes].groupby(by=pathologyData['Protein Id']).max().sum()
    print(subtypes)
    print(pathologyData[MelanomaSubtypes].sum())

    normalData.to_csv(cancerDir + "normalMelanoma.csv", index=True, mode='w')
    pathologyData.to_csv(cancerDir + "pathologyMelanoma.csv", index=True, mode='w')


def genSkinCancerData(normalFilePath, pathologyFilePath):
    print("SkinCancer")
    normalData = pd.read_csv(normalFilePath, header=0, index_col=0)
    pathologyData = pd.read_csv(pathologyFilePath, header=0, index_col=0)

    normalData = normalData[normalData['Organ'].isin(['Skin'])]
    pathologyData = pathologyData[pathologyData['Tissue'].isin(['Skin cancer'])]

    normalProtein = normalData['Protein Id'].drop_duplicates()
    pathologyProtein = pathologyData['Protein Id'].drop_duplicates()
    intersection = pd.Series(list(set(normalProtein).intersection(set(pathologyProtein))))

    print(intersection)

    normalData = normalData[normalData['Protein Id'].isin(intersection)]
    pathologyData = pathologyData[pathologyData['Protein Id'].isin(intersection)]

    pathologyData[SkinCancerSubtypes] = 0
    for subtype in SkinCancerSubtypes:
        pathologyData.loc[pathologyData['SnomedParameters'].str.contains(subtype), subtype] = 1

    # print(normalData)
    # print(pathologyData)
    print(len(normalData))
    print(len(pathologyData))

    subtypes = pathologyData[SkinCancerSubtypes].groupby(by=pathologyData['Protein Id']).max().sum()
    print(subtypes)
    print(pathologyData[SkinCancerSubtypes].sum())

    normalData.to_csv(cancerDir + "normalSkinCancer.csv", index=True, mode='w')
    pathologyData.to_csv(cancerDir + "pathologySkinCancer.csv", index=True, mode='w')


def genData(normalFilePath, pathologyFilePath):
    genLymphomaData(normalFilePath, pathologyFilePath)
    genGliomaData(normalFilePath, pathologyFilePath)
    genThyroidCancerData(normalFilePath, pathologyFilePath)
    genBreastCancerData(normalFilePath, pathologyFilePath)
    genCervicalCancerData(normalFilePath, pathologyFilePath)
    genEndometrialCancerData(normalFilePath, pathologyFilePath)
    genOvarianCancerData(normalFilePath, pathologyFilePath)
    genColorectalCancerData(normalFilePath, pathologyFilePath)
    genStomachCancerData(normalFilePath, pathologyFilePath)
    genRenalCancerData(normalFilePath, pathologyFilePath)
    genUrothelialCancerData(normalFilePath, pathologyFilePath)
    genLiverCancerData(normalFilePath, pathologyFilePath)
    genTestisCancerData(normalFilePath, pathologyFilePath)
    genProstateCancerData(normalFilePath, pathologyFilePath)
    genLungCancerData(normalFilePath, pathologyFilePath)
    genGliomaData(normalFilePath, pathologyFilePath)
    genMelanomaData(normalFilePath, pathologyFilePath)
    genSkinCancerData(normalFilePath, pathologyFilePath)


def showTissue(normalFilePath, pathologyFilePath, savePath):
    normalData = pd.read_csv(normalFilePath, header=0, index_col=0)
    print(normalData)
    pathologyData = pd.read_csv(pathologyFilePath, header=0, index_col=0)
    print(pathologyData)

    normal = normalData[['Organ', 'Tissue']].drop_duplicates().sort_values(by=['Organ', 'Tissue'])
    normal['condition'] = 'normal'
    pathology = pathologyData[['Organ', 'Tissue']].drop_duplicates().sort_values(by=['Organ', 'Tissue'])
    pathology['condition'] = 'pathology'

    print(normal)
    print(pathology)
    print(len(normal))
    print(len(pathology))

    tissueAnalysis = pd.concat([normal, pathology], axis=0)
    tissueAnalysis = tissueAnalysis.sort_values(by=['Organ', 'condition', 'Tissue'])

    tissueAnalysis.to_csv(savePath, index=None, mode='w')


def showSnomed(normalFilePath, pathologyFilePath):
    normalData = pd.read_csv(normalFilePath, header=0, index_col=0)
    print(normalData)
    pathologyData = pd.read_csv(pathologyFilePath, header=0, index_col=0)
    print(pathologyData)

    analysisList = []
    for tissue in tissueList:
        tissueData = normalData[normalData['Tissue'] == tissue]

        tissue_count = tissueData['Tissue'].value_counts().rename('Tissue count').rename_axis('Tissue').reset_index()
        organ_count = tissueData['Organ'].value_counts().rename('Organ count').rename_axis('Organ').reset_index()
        snomedParameters_count = tissueData['SnomedParameters'].value_counts().rename('SnomedParameters count').rename_axis('SnomedParameters').reset_index()
        snomed_count = tissueData['SnomedParameters'].str.split(";", expand=True).stack().value_counts().rename('Snomed count').rename_axis('Snomed').reset_index()

        analysis = pd.concat([organ_count, tissue_count, snomed_count, snomedParameters_count], axis=1)
        analysis['condition'] = 'normal'
        print(analysis)
        analysisList.append(analysis)

    for cancer in cancerList:
        cancerData = pathologyData[pathologyData['Tissue'] == cancer]

        tissue_count = cancerData['Tissue'].value_counts().rename('Tissue count').rename_axis('Tissue').reset_index()
        organ_count = cancerData['Organ'].value_counts().rename('Organ count').rename_axis('Organ').reset_index()
        snomedParameters_count = cancerData['SnomedParameters'].value_counts().rename('SnomedParameters count').rename_axis('SnomedParameters').reset_index()
        snomed_count = cancerData['SnomedParameters'].str.split(";", expand=True).stack().value_counts().rename('Snomed count').rename_axis('Snomed').reset_index()

        analysis = pd.concat([organ_count, tissue_count, snomed_count, snomedParameters_count], axis=1)
        analysis['condition'] = 'cancer'
        print(analysis)
        analysisList.append(analysis)

    snomedAnalysis = pd.concat(analysisList, axis=0)
    print(snomedAnalysis)

    snomedAnalysis.to_csv(dataDir + "SnomedAnalysis.csv", index=None, mode='w')



def subtypesAnalysis(filePath, savePath):
    PathologyData = pd.read_csv(filePath, header=0, index_col=0)
    print(PathologyData)
    PathologyData[subtypeList] = 0
    for subtype in subtypeList:
        PathologyData.loc[PathologyData['SnomedParameters'].str.contains(subtype), subtype] = 1
    print(PathologyData)
    PathologyData.to_csv(savePath, index=True, mode='w')


def dataMatch(normalFilePath, subtypeFilePath, normalSavePath, subtypeSavePath):
    normalData = pd.read_csv(normalFilePath, header=0, index_col=0)
    print(normalData)
    subtypeData = pd.read_csv(subtypeFilePath, header=0, index_col=0)
    subtypeData = subtypeData[subtypeData[subtypeList[1:]].sum(axis=1) > 0]
    print(subtypeData)

    normalGroup = normalData[['Protein Id', 'Organ']].drop_duplicates()
    subtypeGroup = subtypeData[['Protein Id', 'Organ']].drop_duplicates()
    intersected_df = pd.merge(normalGroup, subtypeGroup, how='inner')
    print(normalGroup)
    print(subtypeGroup)
    print(intersected_df)

    normalData = pd.merge(normalData.reset_index(), intersected_df, how='right').set_index('index')
    subtypeData = pd.merge(subtypeData.reset_index(), intersected_df, how='right').set_index('index')
    print(normalData)
    print(subtypeData)

    normalData.to_csv(normalSavePath, index=True, mode='w')
    subtypeData.to_csv(subtypeSavePath, index=True, mode='w')


def dataScreening(normalFilePath, subtypeFilePath, normalSavePath, subtypeSavePath):
    normalData = pd.read_csv(normalFilePath, header=0, index_col=0)
    print(normalData)
    subtypeData = pd.read_csv(subtypeFilePath, header=0, index_col=0)
    print(subtypeData)

    normalData = normalData[
        (normalData['Staining Level'].str.contains('high')) &
        (normalData['Quantity'].str.contains(r'>75%'))
    ]

    subtypeData = subtypeData[
        (subtypeData['Staining Level'].str.contains('High')) &
        (subtypeData['Quantity'].str.contains(r'>75%'))
    ]

    print(normalData)
    print(subtypeData)

    normalGroup = normalData[['Protein Id', 'Organ']].drop_duplicates()
    subtypeGroup = subtypeData[['Protein Id', 'Organ']].drop_duplicates()
    intersected_df = pd.merge(normalGroup, subtypeGroup, how='inner')
    print(normalGroup)
    print(subtypeGroup)
    print(intersected_df)

    normalData = pd.merge(normalData.reset_index(), intersected_df, how='right').set_index('index')
    subtypeData = pd.merge(subtypeData.reset_index(), intersected_df, how='right').set_index('index')
    print(normalData)
    print(subtypeData)

    normalData.to_csv(normalSavePath, index=True, mode='w')
    subtypeData.to_csv(subtypeSavePath, index=True, mode='w')


def dataSplit(normalFilePath, subtypeFilePath):
    subtypeData = pd.read_csv(subtypeFilePath, header=0, index_col=0)
    print(subtypeData)
    print(subtypeData[subtypeList].sum(axis=0))




if __name__ == '__main__':

    getDetectedData()
    screenData()

    normalFilePath = cancerDir + "screenedNormalData.csv"
    pathologyFilePath = cancerDir + "screenedPathologyData.csv"
    pathologyAnalysis(cancerDir + "screenedPathologyData.csv")

    genData(normalFilePath, pathologyFilePath)
