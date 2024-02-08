import numpy as np
import pandas as pd

from PIL import Image


RNG_SEED = 0
np.random.seed(RNG_SEED)


# dataDir = "data/data-6029-5locations/"
# dataDir = "data/data-check/"
dataDir = "data/"
# imageDir = "E:/data/IHC/"
imageDir = "G:/data/IHC/"
imageDir2 = "H:/data/IHC/"
# locationList = ['actin filaments', 'centrosome',
#     'cytosol', 'endoplasmic reticulum', 'golgi apparatus', 'intermediate filaments',
#     'microtubules', 'mitochondria', 'nuclear membrane', 'nucleoli', 'nucleoplasm',
#     'plasma membrane', 'vesicles']
# locationList = ['centrosome', 'cytosol', 'endoplasmic reticulum', 'golgi apparatus',
#     'mitochondria', 'plasma membrane', 'vesicles', 'cytoskeleton', 'nucleus']
# locationList = ['endoplasmic reticulum', 'golgi apparatus', 'mitochondria', 'plasma membrane', 'vesicles', 'nucleus', 'cytoplasm']
locationList = ['cytoplasm', 'cytoskeleton', 'endoplasmic reticulum', 'golgi apparatus', 'lysosomes', 'mitochondria',
                'nucleoli', 'nucleus', 'plasma membrane', 'vesicles']

""" ['Protein Name', 'Protein Id', 'Antibody Id', 'Reliability Verification',
    'Tissue', 'Organ', 'Cell Type', 'Staining Level', 'Intensity Level', 'Quantity', 'Location',
    'Sex', 'Age', 'Patient Id', 'SnomedParameters', 'URL'] """
def tissueAnalysis(filePath, savePath):
    tissueData = pd.read_csv(filePath, header=0)

    print("Tissue数据条数：", len(tissueData))
    protein_count = tissueData['Protein Id'].value_counts(sort=False).rename('Protein Id count').rename_axis('Protein Id').reset_index()
    # print(protein_count)
    antibody_count = tissueData['Antibody Id'].value_counts(sort=False).rename('Antibody Id count').rename_axis('Antibody Id').reset_index()
    # print(antibody_count)
    reliability_count = tissueData['Reliability Verification'].value_counts().rename('Reliability Verification count')
    reliability_count = pd.concat([reliability_count, tissueData['Reliability Verification'].value_counts(normalize=True).rename('Reliability Verification ratio')], axis=1)
    reliability_count = reliability_count.rename_axis('Reliability Verification').reset_index()
    # print(reliability_count)
    tissue_count = tissueData['Tissue'].value_counts().rename('Tissue count').rename_axis('Tissue').reset_index()
    # print(tissue_count)
    organ_count = tissueData['Organ'].value_counts().rename('Organ count').rename_axis('Organ').reset_index()
    # print(organ_count)

    tissueCell = tissueData[['Protein Id', 'Antibody Id', 'Tissue', 'Organ', 'Cell Type', 'Staining Level', 'Intensity Level', 'Quantity', 'Location']].drop_duplicates()
    # print(tissueCell)
    info_cellType = tissueCell['Cell Type'].str.split(";", expand=True).stack().rename('Cell Type').reset_index()
    # print(info_cellType)
    info_stainingLevel = tissueCell['Staining Level'].str.split(";", expand=True).stack().reset_index(drop=True).rename('Staining Level')
    # print(info_stainingLevel)
    info_intensityLevel = tissueCell['Intensity Level'].str.split(";", expand=True).stack().reset_index(drop=True).rename('Intensity Level')
    # print(info_intensityLevel)
    info_quantity = tissueCell['Quantity'].str.split(";", expand=True).stack().reset_index(drop=True).rename('Quantity')
    # print(info_quantity)
    info_location = tissueCell['Location'].str.split(";", expand=True).stack().reset_index(drop=True).rename('Location')
    # print(info_location)

    info_tissueCell = info_cellType.join([info_stainingLevel, info_intensityLevel, info_quantity, info_location]).set_index('level_0').drop(['level_1'], axis=1)
    # print(info_tissueCell)
    tissueCell = tissueCell.drop(['Cell Type', 'Staining Level', 'Intensity Level', 'Quantity', 'Location'], axis=1).join(info_tissueCell)
    # print(tissueCell)
    print("Tissue Cell Count:", len(tissueCell))

    cellType_count = tissueCell['Cell Type'].value_counts().rename('Cell Type count').rename_axis('Cell Type').reset_index()
    # print(cellType_count)
    stainingLevel_count = tissueCell['Staining Level'].value_counts().rename('Staining Level count').rename_axis('Staining Level').reset_index()
    # print(stainingLevel_count)
    intensityLevel_count = tissueCell['Intensity Level'].value_counts().rename('Intensity Level count').rename_axis('Intensity Level').reset_index()
    # print(intensityLevel_count)
    quantity_count = tissueCell['Quantity'].value_counts().rename('Quantity count').rename_axis('Quantity').reset_index()
    # print(quantity_count)
    location_count = tissueCell['Location'].value_counts().rename('Location count').rename_axis('Location').reset_index()
    # print(location_count)

    sex_count = tissueData['Sex'].value_counts().rename('Sex count').rename_axis('Sex').reset_index()
    # print(sex_count)
    age_count = tissueData['Age'].value_counts(bins=range(0, 100, 5)).rename('Age count').rename_axis('Age').reset_index()
    # print(age_count)
    patient_count = tissueData['Patient Id'].value_counts().rename('Patient Id count').rename_axis('Patient Id').reset_index()
    # print(patient_count)
    snomedParameters_count = tissueData['SnomedParameters'].value_counts().rename('SnomedParameters count').rename_axis('SnomedParameters').reset_index()
    # print(snomedParameters_count)
    snomed_count = tissueData['SnomedParameters'].str.split(";", expand=True).stack().value_counts().rename('Snomed count').rename_axis('Snomed').reset_index()
    # print(snomed_count)

    analysis = pd.concat([protein_count, antibody_count, reliability_count, tissue_count, organ_count, cellType_count, stainingLevel_count, intensityLevel_count, quantity_count, location_count, sex_count, age_count, patient_count, snomedParameters_count, snomed_count], axis=1)
    # print(analysis)
    analysis.to_csv(savePath, index=False, mode='w')

""" ['Protein Name', 'Protein Id', 'Antibody Id', 'Tissue', 'Organ', 'Cell Type',
    'Sex', 'Age', 'Patient Id', 'Staining Level', 'Intensity Level', 'Quantity', 'Location', 'SnomedParameters', 'URL'] """
def pathologyAnalysis(filePath, savePath):
    PathologyData = pd.read_csv(filePath, header=0)

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
    # print(analysis)
    analysis.to_csv(savePath, index=False, mode='w')


def getAnnotation(filePath1, filePath2, savePath):
    normalData = pd.read_csv(filePath1, header=0)
    badUrl = pd.read_csv("data/normal_bad_url.csv", header=None)
    normalData = normalData.drop(normalData[normalData['Protein Id'].isin(badUrl[0]) & normalData['Antibody Id'].isin(badUrl[1]) & normalData['Tissue'].isin(badUrl[2]) & normalData['Organ'].isin(badUrl[3]) & normalData['URL'].isin(badUrl[4])].index)
    normalData['URL'] = normalData['Protein Id'] + "/" + normalData['Organ'] + "/" + normalData['Tissue'] + "/" + normalData['Antibody Id'] + "/" + normalData['URL'].str.split('/', expand=True)[4]
    annotationData = pd.read_csv(filePath2, header=0)

    """ 统一列名称和组织 """
    annotationData['IF Organ'] = annotationData['organ']
    organDic = {
        'Lymphoid': 'Bone marrow & lymphoid tissues',
        'Myeloid': 'Bone marrow & lymphoid tissues',
        'Mesenchymal': 'Connective & soft tissue',
        'Female reproductive system': 'Female tissues',
        'Kidney & Urinary bladder': 'Kidney & urinary bladder',
        'Liver & Gallbladder': 'Liver & gallbladder',
        'Lung': 'Respiratory system'}
    annotationData['organ'].replace(organDic, inplace=True)
    annotationData.rename(columns={'proteinName': 'Protein Name', 'proteinId': 'Protein Id', 'antibodyId': 'Antibody Id', 'verification': 'IF Verification', 'organ': 'Organ'}, inplace=True)

    """ 去除annotation中位置标注为空的数据 """
    annotationData = annotationData.dropna(axis=0, subset = ['locations'])

    """ 按['Protein Name', 'Protein Id', 'Antibody Id', 'IF Verification', 'Organ']将数据进行合并 """
    grouped = annotationData.groupby(['Protein Name', 'Protein Id', 'Antibody Id', 'IF Verification', 'Organ'], as_index=False).max()
    integrate = pd.merge(annotationData[['Protein Name', 'Protein Id', 'Antibody Id', 'IF Verification', 'Organ']], grouped, how='left').set_index(annotationData.index)
    dup_row = integrate.drop(['locations'], axis=1).duplicated()
    annotationData = integrate[dup_row == False].reset_index(drop=True)

    """ 划分亚细胞位置，将部分亚细胞位置进行合并 """
    # print(annotationData)
    # annotationData.to_csv("data/anno35.csv", index=True, mode='w')
    # subcellularLocs = ['actin filaments', 'aggresome', 'cell junctions', 'centriolar satellite', 'centrosome', 'cleavage furrow', 'cytokinetic bridge',
#     'cytoplasmic bodies', 'cytosol', 'endoplasmic reticulum', 'endosomes', 'focal adhesion sites', 'golgi apparatus', 'intermediate filaments', 'kinetochore',
#     'lipid droplets', 'lysosomes', 'microtubule ends', 'microtubules', 'midbody', 'midbody ring', 'mitochondria', 'mitotic chromosome', 'mitotic spindle',
#     'nuclear bodies', 'nuclear membrane', 'nuclear speckles', 'nucleoli', 'nucleoli fibrillar center', 'nucleoli rim', 'nucleoplasm', 'peroxisomes',
#     'plasma membrane', 'rods & rings', 'vesicles']
    annotationData['actin filaments'] = annotationData[['actin filaments', 'cleavage furrow', 'focal adhesion sites']].max(axis=1)
    annotationData['cytosol'] = annotationData[['aggresome', 'cytoplasmic bodies', 'cytosol', 'rods & rings']].max(axis=1)
    annotationData['centrosome'] = annotationData[['centriolar satellite', 'centrosome']].max(axis=1)
    annotationData['microtubules'] = annotationData[['cytokinetic bridge', 'microtubule ends', 'microtubules', 'midbody', 'midbody ring', 'mitotic spindle']].max(axis=1)
    annotationData['plasma membrane'] = annotationData[['cell junctions', 'plasma membrane']].max(axis=1)
    annotationData['vesicles'] = annotationData[['endosomes', 'lipid droplets', 'lysosomes', 'peroxisomes', 'vesicles']].max(axis=1)
    annotationData['lysosomes'] = 0
    # annotationData['vesicles'] = annotationData[['endosomes', 'lipid droplets', 'peroxisomes', 'vesicles']].max(axis=1)
    annotationData['nucleoplasm'] = annotationData[['kinetochore', 'mitotic chromosome', 'nuclear bodies', 'nuclear speckles', 'nucleoplasm']].max(axis=1)
    annotationData['nucleoli'] = annotationData[['nucleoli', 'nucleoli fibrillar center', 'nucleoli rim']].max(axis=1)
    # annotationData = annotationData.drop(['aggresome', 'cell junctions', 'centriolar satellite', 'cleavage furrow', 'cytokinetic bridge',
    #     'cytoplasmic bodies', 'endosomes', 'focal adhesion sites', 'kinetochore',
    #     'lipid droplets', 'lysosomes', 'microtubule ends', 'midbody', 'midbody ring', 'mitotic chromosome', 'mitotic spindle',
    #     'nuclear bodies', 'nuclear speckles', 'nucleoli fibrillar center', 'nucleoli rim', 'peroxisomes',
    #     'rods & rings'], axis=1)
    annotationData = annotationData.drop(['aggresome', 'cell junctions', 'centriolar satellite', 'cleavage furrow', 'cytokinetic bridge',
        'cytoplasmic bodies', 'endosomes', 'focal adhesion sites', 'kinetochore',
        'lipid droplets', 'microtubule ends', 'midbody', 'midbody ring', 'mitotic chromosome', 'mitotic spindle',
        'nuclear bodies', 'nuclear speckles', 'nucleoli fibrillar center', 'nucleoli rim', 'peroxisomes',
        'rods & rings'], axis=1)


    annotationData['cytoskeleton'] = annotationData[['actin filaments', 'microtubules', 'intermediate filaments']].max(axis=1)
    annotationData['nucleus'] = annotationData[['nuclear membrane', 'nucleoli', 'nucleoplasm']].max(axis=1)
    annotationData['nucleoli'] = 0
    # annotationData['nucleus'] = annotationData[['nuclear membrane', 'nucleoplasm']].max(axis=1)
    # annotationData = annotationData.drop(['actin filaments', 'microtubules', 'intermediate filaments',
    #     'nuclear membrane', 'nucleoli', 'nucleoplasm'], axis=1)
    annotationData = annotationData.drop(['actin filaments', 'microtubules', 'intermediate filaments',
        'nuclear membrane', 'nucleoplasm'], axis=1)

    annotationData['cytoplasm'] = annotationData[['centrosome', 'cytosol', 'cytoskeleton']].max(axis=1)
    annotationData['cytoskeleton'] = 0
    # annotationData['cytoplasm'] = annotationData[['centrosome', 'cytosol']].max(axis=1)
    # annotationData = annotationData.drop(['centrosome', 'cytosol', 'cytoskeleton'], axis=1)
    annotationData = annotationData.drop(['centrosome', 'cytosol'], axis=1)

    # annotationData['endoplasmic reticulum'] = annotationData[['endoplasmic reticulum', 'golgi apparatus', 'vesicles', 'plasma membrane']].max(axis=1)
    annotationData['endoplasmic reticulum'] = annotationData[['endoplasmic reticulum', 'golgi apparatus', 'vesicles']].max(axis=1)
    annotationData['golgi apparatus'] = 0
    annotationData['vesicles'] = 0
    # annotationData['plasma membrane'] = 0

    temp = annotationData['IF Organ']
    annotationData = annotationData.drop(labels=['IF Organ'], axis=1)
    annotationData.insert(6, 'IF Organ', temp)
    # print(annotationData)
    # print(list(annotationData))
    # annotationData.to_csv("data/anno13.csv", index=True, mode='w')

    """ 重置合并数据的'locations'标签 """
    # print(annotationData[annotationData['locations'].str.count(';')+1 != annotationData.iloc[:,6:41].sum(axis=1)])
    # merged = annotationData['locations'].str.count(';')+1 != annotationData.iloc[:,6:19].sum(axis=1)
    # location_list = list(annotationData)[5:19]
    # for item in annotationData[merged].iloc[:,6:19].itertuples():
    #     locs = []
    #     for idx in range(1, len(item)):
    #         if item[idx] == 1:
    #             locs.append(location_list[idx])
    #     locs = ';'.join(locs)
    #     annotationData.iat[item[0], 5] = locs
    merged = annotationData['locations'].str.count(';')+1 != annotationData.iloc[:,7:].sum(axis=1)
    location_list = list(annotationData)[6:]
    for item in annotationData[merged].iloc[:,7:].itertuples():
        locs = []
        for idx in range(1, len(item)):
            if item[idx] == 1:
                locs.append(location_list[idx])
        locs = ';'.join(locs)
        annotationData.iat[item[0], 5] = locs
    # print(annotationData['locations'].str.count(';')+1)
    # print(annotationData.iloc[:,6:41].sum(axis=1))
    # print(annotationData[annotationData['locations'].str.count(';')+1 != annotationData.iloc[:,6:41].sum(axis=1)])

    print(list(annotationData))
    annotationData = annotationData.reindex(columns=(list(annotationData)[:-len(locationList)] + locationList))
    print(list(annotationData))

    newAnnotation = pd.merge(normalData, annotationData, how='left')
    # print(newAnnotation)
    # print(list(newAnnotation))
    # print(normalData)
    newAnnotation.to_csv(savePath, index=True, mode='w')
    return newAnnotation


def getPairIdx(normalPath, pathologyPath):
    normalData = pd.read_csv(normalPath, header=0, index_col=0)
    pathologyData = pd.read_csv(pathologyPath, header=0, index_col=0)

    normalData = normalData[['Protein Name', 'Protein Id', 'Antibody Id', 'Reliability Verification', 'Tissue', 'Organ', 'Cell Type', 'Staining Level', 'Intensity Level', 'Quantity', 'Location', 'Sex', 'Age', 'Patient Id', 'SnomedParameters', 'URL', 'IF Verification', 'locations', 'IF Organ'] + locationList]
    pathologyData = pathologyData[['Protein Name', 'Protein Id', 'Antibody Id', 'Tissue', 'Organ', 'Cell Type', 'Sex', 'Age', 'Patient Id', 'Staining Level', 'Intensity Level', 'Quantity', 'Location', 'SnomedParameters', 'URL']]
    pathologyData['URL'] = pathologyData['Protein Id'] + "/" + pathologyData['Organ'] + "/" + pathologyData['Tissue'] + "/" + pathologyData['Antibody Id'] + "/" + pathologyData['URL'].str.split('/', expand=True)[4]

    normalGroup = normalData[['Protein Name', 'Protein Id', 'Antibody Id', 'Tissue', 'Organ']].drop_duplicates().reset_index(drop=True)
    normalGroup["Pair Idx"] = "N-" + normalGroup.index.astype('str')
    normalData = pd.merge(normalData, normalGroup, how="left", on=['Protein Name', 'Protein Id', 'Antibody Id', 'Tissue', 'Organ'])

    pathologyGroup = pathologyData[['Protein Name', 'Protein Id', 'Antibody Id', 'Tissue', 'Organ']].drop_duplicates().reset_index(drop=True)
    pathologyGroup["Pair Idx"] = "P-" + pathologyGroup.index.astype('str')
    pathologyData = pd.merge(pathologyData, pathologyGroup, how="left", on=['Protein Name', 'Protein Id', 'Antibody Id', 'Tissue', 'Organ'])

    normalData.to_csv(normalPath, index=True, mode='w')
    pathologyData.to_csv(pathologyPath, index=True, mode='w')


def deleteWrongData(dataPath, deletedDataPath, condition="normal"):
    data = pd.read_csv(dataPath, header=0, index_col=0)

    wrongData = []
    for index, row in data.iterrows():
        if row["Pair Idx"].split('-')[0] == "N":
            im_path = imageDir + "normal/" + row.URL
        elif row["Pair Idx"].split('-')[0] == "P":
            if index >= 5500000:
                im_path = imageDir + "pathology/" + row.URL
            else:
                im_path = imageDir2 + "pathology/" + row.URL
        # if condition == "normal":
        #     im_path = imageDir + condition + "/" + row.URL
        # else:
        #     if index >= 5500000:
        #         im_path = imageDir + condition + "/" + row.URL
        #     else:
        #         im_path = imageDir2 + condition + "/" + row.URL
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


""" ['Protein Name', 'Protein Id', 'Antibody Id', 'Reliability Verification',
    'Tissue', 'Organ', 'Cell Type', 'Staining Level', 'Intensity Level', 'Quantity', 'Location',
    'Sex', 'Age', 'Patient Id', 'SnomedParameters', 'URL', 'IF Verification', 'locations', ...] """
def annotationAnalysis(annotationData, savePath1, savePath2):
    normalLabeled = annotationData[annotationData['locations'].notnull()]
    print(normalLabeled)
    normalLabeled.to_csv(savePath1, index=True, mode='w')

    print("Labeled Normal Tissue数据条数：", len(normalLabeled))
    protein_count = normalLabeled['Protein Id'].value_counts(sort=False).rename('Protein Id count').rename_axis('Protein Id').reset_index()
    antibody_count = normalLabeled['Antibody Id'].value_counts(sort=False).rename('Antibody Id count').rename_axis('Antibody Id').reset_index()
    reliability_count = normalLabeled['Reliability Verification'].value_counts().rename('Reliability Verification count')
    reliability_count = pd.concat([reliability_count, normalLabeled['Reliability Verification'].value_counts(normalize=True).rename('Reliability Verification ratio')], axis=1)
    reliability_count = reliability_count.rename_axis('Reliability Verification').reset_index()
    # print(reliability_count)

    IF_reliability_count = normalLabeled['IF Verification'].value_counts().rename('IF Verification count')
    IF_reliability_count = pd.concat([IF_reliability_count, normalLabeled['IF Verification'].value_counts(normalize=True).rename('IF Verification ratio')], axis=1)
    IF_reliability_count = IF_reliability_count.rename_axis('IF Verification').reset_index()
    # print(IF_reliability_count)

    tissue_count = normalLabeled['Tissue'].value_counts().rename('Tissue count').rename_axis('Tissue').reset_index()
    organ_count = normalLabeled['Organ'].value_counts().rename('Organ count').rename_axis('Organ').reset_index()

    tissueCell = normalLabeled[['Protein Id', 'Antibody Id', 'Tissue', 'Organ', 'Cell Type', 'Staining Level', 'Intensity Level', 'Quantity', 'Location']].drop_duplicates()
    info_cellType = tissueCell['Cell Type'].str.split(";", expand=True).stack().rename('Cell Type').reset_index()
    info_stainingLevel = tissueCell['Staining Level'].str.split(";", expand=True).stack().reset_index(drop=True).rename('Staining Level')
    info_intensityLevel = tissueCell['Intensity Level'].str.split(";", expand=True).stack().reset_index(drop=True).rename('Intensity Level')
    info_quantity = tissueCell['Quantity'].str.split(";", expand=True).stack().reset_index(drop=True).rename('Quantity')
    info_location = tissueCell['Location'].str.split(";", expand=True).stack().reset_index(drop=True).rename('Location')

    info_tissueCell = info_cellType.join([info_stainingLevel, info_intensityLevel, info_quantity, info_location]).set_index('level_0').drop(['level_1'], axis=1)
    tissueCell = tissueCell.drop(['Cell Type', 'Staining Level', 'Intensity Level', 'Quantity', 'Location'], axis=1).join(info_tissueCell)
    print("Tissue Cell Count:", len(tissueCell))

    cellType_count = tissueCell['Cell Type'].value_counts().rename('Cell Type count').rename_axis('Cell Type').reset_index()
    stainingLevel_count = tissueCell['Staining Level'].value_counts().rename('Staining Level count').rename_axis('Staining Level').reset_index()
    intensityLevel_count = tissueCell['Intensity Level'].value_counts().rename('Intensity Level count').rename_axis('Intensity Level').reset_index()
    quantity_count = tissueCell['Quantity'].value_counts().rename('Quantity count').rename_axis('Quantity').reset_index()
    location_count = tissueCell['Location'].value_counts().rename('Location count').rename_axis('Location').reset_index()

    sex_count = normalLabeled['Sex'].value_counts().rename('Sex count').rename_axis('Sex').reset_index()
    age_count = normalLabeled['Age'].value_counts(bins=range(0, 100, 5)).rename('Age count').rename_axis('Age').reset_index()
    patient_count = normalLabeled['Patient Id'].value_counts().rename('Patient Id count').rename_axis('Patient Id').reset_index()
    snomedParameters_count = normalLabeled['SnomedParameters'].value_counts().rename('SnomedParameters count').rename_axis('SnomedParameters').reset_index()
    snomed_count = normalLabeled['SnomedParameters'].str.split(";", expand=True).stack().value_counts().rename('Snomed count').rename_axis('Snomed').reset_index()

    locations_count = normalLabeled[locationList].sum().rename('Locations count').rename_axis('Locations').reset_index()
    print(locations_count)

    analysis = pd.concat([protein_count, antibody_count, reliability_count, IF_reliability_count, tissue_count, organ_count, cellType_count, locations_count, stainingLevel_count, intensityLevel_count, quantity_count, location_count, sex_count, age_count, patient_count, snomedParameters_count, snomed_count], axis=1)
    print(analysis)
    analysis.to_csv(savePath2, index=False, mode='w')


def dataScreening(filePath1, filePath2, pathologyFilePath):
    normalLabeledData = pd.read_csv(filePath1, header=0, index_col=0)
    normalData = pd.read_csv(filePath2, header=0, index_col=0)
    # print(normalLabeledData)
    # print(normalData)
    # normalLabeledData = normalLabeledData.drop(['Cell Type', 'Intensity Level', 'Quantity', 'Location', 'Sex', 'Age', 'Patient Id'], axis=1)
    # normalData = normalData.drop(['Cell Type', 'Intensity Level', 'Quantity', 'Location', 'Sex', 'Age', 'Patient Id'], axis=1)
    normalLabeledData = normalLabeledData.drop(['Cell Type', 'Location', 'Sex', 'Age', 'Patient Id'], axis=1)
    normalData = normalData.drop(['Cell Type', 'Location', 'Sex', 'Age', 'Patient Id'], axis=1)
    # normalLabeledData = normalLabeledData.drop(['Cell Type', 'Intensity Level', 'Location', 'Sex', 'Age', 'Patient Id'], axis=1)
    # normalData = normalData.drop(['Cell Type', 'Intensity Level', 'Location', 'Sex', 'Age', 'Patient Id'], axis=1)
    print("normalLabeledData:", normalLabeledData)
    print("normalData:", normalData)

    # data1 = normalLabeledData[(normalLabeledData['Reliability Verification'] == 'enhanced') & (normalLabeledData['IF Verification'] == 'enhanced') &
    #     (normalLabeledData['Staining Level'].str.contains('high'))]
    # data1 = normalLabeledData[(normalLabeledData['Reliability Verification'].isin(['enhanced', 'supported', 'approved'])) &
    # data1 = normalLabeledData[
    #     (normalLabeledData['Reliability Verification'].isin(['enhanced', 'supported', 'approved'])) &
    #     (normalLabeledData['IF Verification'].isin(['enhanced'])) &
    #     (normalLabeledData['Staining Level'].str.contains('high|medium')) &
    #     (normalLabeledData['Quantity'].str.contains(r'>75%'))]

    # data1 = normalLabeledData[
    #     # (normalLabeledData['Reliability Verification'].isin(['enhanced', 'supported', 'approved'])) &
    #     (normalLabeledData['IF Verification'].isin(['enhanced'])) &
    #     (normalLabeledData['Staining Level'].str.contains('high')) &
    #     (normalLabeledData['Quantity'].str.contains(r'>75%|75%-25%'))]
    #     # (normalLabeledData['Quantity'].str.contains(r'>75%'))]

    # data1 = normalLabeledData[
    #     (normalLabeledData['Reliability Verification'].isin(['enhanced', 'supported'])) &
    #     (normalLabeledData['IF Verification'].isin(['enhanced', 'supported'])) &
    #     (normalLabeledData['Staining Level'].str.contains('high')) &
    #     (normalLabeledData['Quantity'].str.contains(r'>75%'))]
    # data1 = normalLabeledData[
    #     (normalLabeledData['IF Verification'].isin(['enhanced', 'supported'])) &
    #     (normalLabeledData['Staining Level'].str.contains('high|medium')) &
    #     (normalLabeledData['Quantity'].str.contains(r'>75%'))]

    # data1 = normalLabeledData[
    #     (normalLabeledData['Reliability Verification'].isin(['enhanced', 'supported'])) &
    #     (normalLabeledData['IF Verification'].isin(['enhanced'])) &
    #     (normalLabeledData['Staining Level'].str.contains('high')) &
    #     (normalLabeledData['Quantity'].str.contains(r'>75%'))]
    # proteins = data1['Protein Id'].drop_duplicates()
    # data1 = normalLabeledData[normalLabeledData['Protein Id'].isin(proteins)]
    # data1 = data1[(data1['Staining Level'].str.contains('high|medium')) &
    #     (data1['Quantity'].str.contains(r'>75%|75%-25%'))]
    #11111
    # data1 = normalLabeledData[
    #     (normalLabeledData['Reliability Verification'].isin(['enhanced', 'supported', 'approved'])) &
    #     (normalLabeledData['IF Verification'].isin(['enhanced'])) &
    #     (normalLabeledData['Staining Level'].str.contains('high|medium')) &
    #     (normalLabeledData['Quantity'].str.contains(r'>75%'))]

    # data1 = normalLabeledData[
    #     (normalLabeledData['IF Verification'].isin(['enhanced'])) &
    #     (normalLabeledData['Staining Level'].str.contains('high|medium'))]
    #     # (normalLabeledData['Staining Level'].str.contains('high|medium')) &
    #     # (normalLabeledData['Quantity'].str.contains(r'>75%'))]

    # # data-6854
    # data1 = normalLabeledData[
    #     (normalLabeledData['IF Verification'].isin(['enhanced'])) &
    #     (normalLabeledData['Staining Level'].str.contains('high'))]

    # data1 = normalLabeledData[
    #     (normalLabeledData['IF Verification'].isin(['enhanced'])) &
    #     (normalLabeledData['Staining Level'].str.contains('high'))]

    # groups = data1.groupby(by=['Protein Id', 'Antibody Id', 'Tissue'])['Protein Name'].count().reset_index()
    # print(groups)
    # data1 = pd.merge(normalLabeledData.reset_index(), groups[['Protein Id', 'Antibody Id', 'Tissue']], on=['Protein Id', 'Antibody Id', 'Tissue'], how='right').set_index('index')

    # print(data1)
    # data1 = data1[
    #     (data1['IF Verification'].isin(['enhanced'])) &
    #     (data1['Staining Level'].str.contains('high|medium'))]

    # # data-13383  data-10876
    # data1 = normalLabeledData[
    #     (normalLabeledData['IF Verification'].isin(['enhanced'])) &
    #     (normalLabeledData['Intensity Level'].str.contains('strong|moderate')) &
    #     (normalLabeledData['Quantity'].str.contains(r'>75%'))]
    #     # (normalLabeledData['Staining Level'].str.contains('high|medium')) &
    #     # (normalLabeledData['Quantity'].str.contains(r'>75%'))]

    # data-6029
    data1 = normalLabeledData[
        (normalLabeledData['IF Verification'].isin(['enhanced'])) &
        (normalLabeledData['Intensity Level'].str.contains('strong')) &
        (normalLabeledData['Quantity'].str.contains(r'>75%'))]


    # data1 = normalLabeledData[
    #     (normalLabeledData['IF Verification'].isin(['enhanced'])) &
    #     ~((normalLabeledData['Intensity Level'].str.contains('negative|weak')) |
    #     (normalLabeledData['Quantity'].str.contains(r'<25%|75%-25%')))]

    # data1 = normalLabeledData[
    #     (normalLabeledData['IF Verification'].isin(['enhanced'])) &
    #     ~(normalLabeledData['Staining Level'].str.contains('not detected|low'))]

    # data1 = normalLabeledData[
    #     (normalLabeledData['IF Verification'].isin(['enhanced', 'supported'])) &
    #     (normalLabeledData['Staining Level'].str.contains('high')) &
    #     (normalLabeledData['Intensity Level'].str.contains('strong')) &
    #     (normalLabeledData['Quantity'].str.contains(r'>75%'))]

    # data1 = normalLabeledData[
    #     (normalLabeledData['Reliability Verification'].isin(['enhanced', 'supported'])) &
    #     (normalLabeledData['IF Verification'].isin(['enhanced', 'supported'])) &
    #     (normalLabeledData['Staining Level'].str.contains('high')) &
    #     (normalLabeledData['Quantity'].str.contains(r'>75%|75%-25%'))]
    #     # (normalLabeledData['Quantity'].str.contains(r'>75%'))]

    # groups = data1.groupby(by=['Protein Id', 'Antibody Id', 'Tissue'])['Protein Name'].count()
    # groups = groups[groups >= 3].reset_index()
    # print(groups)
    # data1 = pd.merge(data1.reset_index(), groups[['Protein Id', 'Antibody Id', 'Tissue']], on=['Protein Id', 'Antibody Id', 'Tissue'], how='right').set_index('index')

    print(data1)
    print(data1[locationList].sum())
    proteins = data1[locationList].groupby(by=data1['Protein Id']).sum()
    proteins = proteins.replace(0, np.nan)
    trainNum = proteins.count(axis=0).values
    print("totalDataNum:", trainNum)
    print(trainNum.sum() / len(proteins))
    data1.to_csv(dataDir + 'data1.csv', index=True, mode='w')
    # data1.to_csv(dataDir + 'data4.csv', index=True, mode='w')

    # pathologyData = pd.read_csv(pathologyFilePath, header=0, index_col=0)
    # print("pathologyData:", pathologyData)
    # # pathologyData.to_csv(dataDir + 'pathology.csv', index=True, mode='w')
    # # pathologyData = pathologyData.drop(['Cell Type', 'Intensity Level', 'Quantity', 'Location', 'Sex', 'Age', 'Patient Id'], axis=1)
    # # pathologyData = pathologyData.drop(['Cell Type', 'Location', 'Sex', 'Age', 'Patient Id'], axis=1)
    # pathologyData = pathologyData.drop(['Cell Type', 'Intensity Level', 'Location', 'Sex', 'Age', 'Patient Id'], axis=1)

    # """ 可靠性为enhanced的数据（高可靠性数据集），且Staining Level为high或medium """
    # # data_A = normalLabeledData[(normalLabeledData['Reliability Verification'] == 'enhanced') & (normalLabeledData['IF Verification'] == 'enhanced') &
    # data_A = normalLabeledData[(normalLabeledData['IF Verification'] == 'enhanced') &
    #     (normalLabeledData['Staining Level'].str.contains('high|medium')) &
    #     (normalLabeledData['Quantity'].str.contains(r'>75%'))]
    #     # (normalLabeledData['Quantity'].str.contains(r'>75%|75%-25%'))]
    #     # (normalLabeledData['Staining Level'].str.contains('high|medium'))]
    # print("data_A:", data_A)

    # """ 可靠性较低一些的数据（IHC为enhanced和supported，IF为enhanced和supported），且Staining Level为high或medium """
    # # data_B = normalLabeledData[(normalLabeledData['Reliability Verification'].isin(['enhanced', 'supported'])) &
    # data_B = normalLabeledData[
    #     (normalLabeledData['IF Verification'].isin(['enhanced', 'supported'])) &
    #     # (normalLabeledData['IF Verification'].isin(['enhanced'])) &
    #     (normalLabeledData['Staining Level'].str.contains('high|medium')) &
    #     (normalLabeledData['Quantity'].str.contains(r'>75%'))]
    #     # (normalLabeledData['Quantity'].str.contains(r'>75%|75%-25%'))]
    #     # (normalLabeledData['Staining Level'].str.contains('high|medium'))]
    # print("data_B:", data_B)

    # """ 可靠性较低一些的数据，除uncertain和未标记数据，且Staining Level为high或medium """
    # # data_C = normalLabeledData[(normalLabeledData['Reliability Verification'].isin(['enhanced', 'supported', 'approved'])) &
    # data_C = normalLabeledData[
    #     (normalLabeledData['IF Verification'].isin(['enhanced', 'supported', 'approved'])) &
    #     (normalLabeledData['Staining Level'].str.contains('high|medium')) &
    #     (normalLabeledData['Quantity'].str.contains(r'>75%'))]
    #     # (normalLabeledData['Quantity'].str.contains(r'>75%|75%-25%'))]
    #     # (normalLabeledData['Staining Level'].str.contains('high|medium'))]
    # print("data_C:", data_C)

    # protein_A = data_A['Protein Id'].drop_duplicates()
    # print("protein_A:", protein_A)
    # """ 与A中蛋白质无重叠的有标签正常组织数据 """
    # data_D = normalLabeledData[~normalLabeledData['Protein Id'].isin(protein_A)]
    # """ 与A中蛋白质无重叠的正常组织数据 """
    # data_D2 = normalData[~normalData['Protein Id'].isin(protein_A)]
    # """ 与A中蛋白质无重叠的癌症组织数据 """
    # data_D3 = pathologyData[~pathologyData['Protein Id'].isin(protein_A)]
    # print("data_D:", data_D)
    # print("data_D2:", data_D2)
    # print("data_D3:", data_D3)

    # protein_B = data_B['Protein Id'].drop_duplicates()
    # print("protein_B:", protein_B)
    # """ 与B中蛋白质无重叠的有标签正常组织数据 """
    # data_E = normalLabeledData[~normalLabeledData['Protein Id'].isin(protein_B)]
    # """ 与B中蛋白质无重叠的正常组织数据 """
    # data_E2 = normalData[~normalData['Protein Id'].isin(protein_B)]
    # """ 与B中蛋白质无重叠的癌症组织数据 """
    # data_E3 = pathologyData[~pathologyData['Protein Id'].isin(protein_B)]
    # print("data_E:", data_E)
    # print("data_E2:", data_E2)
    # print("data_E3:", data_E3)

    # protein_C = data_C['Protein Id'].drop_duplicates()
    # print("protein_C:", protein_C)
    # """ 与C中蛋白质无重叠的有标签正常组织数据 """
    # data_F = normalLabeledData[~normalLabeledData['Protein Id'].isin(protein_C)]
    # """ 与C中蛋白质无重叠的正常组织数据 """
    # data_F2 = normalData[~normalData['Protein Id'].isin(protein_C)]
    # """ 与C中蛋白质无重叠的癌症组织数据 """
    # data_F3 = pathologyData[~pathologyData['Protein Id'].isin(protein_C)]
    # print("data_F:", data_F)
    # print("data_F2:", data_F2)
    # print("data_F3:", data_F3)

    # data_A.to_csv(dataDir + 'data_A.csv', index=True, mode='w')
    # data_B.to_csv(dataDir + 'data_B.csv', index=True, mode='w')
    # data_C.to_csv(dataDir + 'data_C.csv', index=True, mode='w')
    # data_D.to_csv(dataDir + 'data_D.csv', index=True, mode='w')
    # data_D2.to_csv(dataDir + 'data_D2.csv', index=True, mode='w')
    # data_D3.to_csv(dataDir + 'data_D3.csv', index=True, mode='w')
    # data_E.to_csv(dataDir + 'data_E.csv', index=True, mode='w')
    # data_E2.to_csv(dataDir + 'data_E2.csv', index=True, mode='w')
    # data_E3.to_csv(dataDir + 'data_E3.csv', index=True, mode='w')
    # data_F.to_csv(dataDir + 'data_F.csv', index=True, mode='w')
    # data_F2.to_csv(dataDir + 'data_F2.csv', index=True, mode='w')
    # data_F3.to_csv(dataDir + 'data_F3.csv', index=True, mode='w')


# def pretrainDatasetPartitioning(normalDataPath, pathologyDataPath):
#     normalData = pd.read_csv(normalDataPath, header=0, index_col=0)
#     pathologyData = pd.read_csv(pathologyDataPath, header=0, index_col=0)

#     normalGroup = normalData['Pair Idx'].drop_duplicates()
#     pathologyGroup = pathologyData['Pair Idx'].drop_duplicates()

#     selectedNormalGroup = normalGroup.sample(n=4000, random_state=RNG_SEED)
#     selectedPathologyGroup = pathologyGroup.sample(n=500, random_state=RNG_SEED)

#     selectedNormalData = normalData[normalData['Pair Idx'].isin(selectedNormalGroup)]
#     selectedPathologyData = pathologyData[pathologyData['Pair Idx'].isin(selectedPathologyGroup)]

#     data = pd.concat([selectedNormalData, selectedPathologyData], axis=0)
#     data[locationList] = data[locationList].fillna(0)

#     print("normalGroup:", normalGroup)
#     print("pathologyGroup:", pathologyGroup)
#     print("selectedNormalGroup:", selectedNormalGroup)
#     print("selectedPathologyGroup:", selectedPathologyGroup)
#     print("selectedNormalData:", selectedNormalData)
#     print("selectedPathologyData:", selectedPathologyData)
#     print("data:", data)

#     data.to_csv(dataDir + 'data_pretrain.csv', index=True, mode='w')


# def datasetPartitioning(dataPath):
#     data = pd.read_csv(dataPath, header=0, index_col=0)
#     proteins = data[locationList].groupby(by=data['Protein Id']).sum()
#     proteins = proteins.replace(0, np.nan)
#     locations = proteins.T.stack(dropna=True)

#     trainNum = proteins.count(axis=0).values
#     print("totalDataNum:", trainNum)
#     # trainNum = [x//8 if x//8 >= 12 else 12 for x in trainNum]
#     # testNum = [x//3 for x in trainNum]
#     # trainNum = [x//24 if x//24 >= 4 else 4 for x in trainNum]
#     # testNum = [x//2 for x in trainNum]
#     # trainNum = [x//6 if x//6 >= 10 else 10 for x in trainNum]
#     # testNum = [x//10 if x//10 >= 1 else 1 for x in trainNum]
#     trainNum = [x//5 if x//5 >= 10 else 10 for x in trainNum]
#     testNum = [x//10 if x//10 >= 1 else 1 for x in trainNum]

#     trainProteins = []
#     valProteins = []

#     for i in range(len(locationList)):
#         prots = locations[locationList[i]].index
#         trainProteins.extend(prots[:trainNum[i]])
#         valProteins.extend(prots[-testNum[i]:])

#     # trainProteins = list(set(trainProteins))
#     # valProteins = [i for i in valProteins if i not in trainProteins]
#     valProteins = list(set(valProteins))
#     trainProteins = [i for i in trainProteins if i not in valProteins]

#     trainData = data[data['Protein Id'].isin(trainProteins)]
#     valData = data[data['Protein Id'].isin(valProteins)]

#     print("len(trainProteins)", len(trainProteins))
#     print("len(valProteins)", len(valProteins))
#     print("len(trainData)", len(trainData))
#     print("len(valData)", len(valData))
#     trainProteins = trainData[locationList].groupby(by=trainData['Protein Id']).sum()
#     trainProteins = trainProteins.replace(0, np.nan)
#     trainNum = trainProteins.count(axis=0).values
#     print("trainNum", trainNum)
#     valProteins = valData[locationList].groupby(by=valData['Protein Id']).sum()
#     valProteins = valProteins.replace(0, np.nan)
#     valNum = valProteins.count(axis=0).values
#     print("testNum", valNum)
#     print("trainData.sum()")
#     print(trainData[locationList].sum())
#     print("valData.sum()")
#     print(valData[locationList].sum())

#     trainData.to_csv(dataDir + 'data_train.csv', index=True, mode='w')
#     valData.to_csv(dataDir + 'data_val.csv', index=True, mode='w')
#     # trainData.to_csv(dataDir + 'data2.csv', index=True, mode='w')
#     # valData.to_csv(dataDir + 'data3.csv', index=True, mode='w')


# def splitData(dataPath):
#     allData = pd.read_csv(dataPath, header=0, index_col=0)
#     proteins = allData[locationList].groupby(by=allData['Protein Id']).sum().replace(0, np.nan)
#     locations = proteins.T.stack(dropna=True)
#     print(allData[locationList].sum().tolist())
#     proteinList = allData['Protein Id'].drop_duplicates()
#     proteinList = proteinList.sample(frac=1/10, random_state=0).to_list()

#     for loc in locationList:
#         if loc in locations:
#             if len(set(proteinList) & set(locations[loc].index.to_list())) == 0:
#                 proteinList.extend(locations[loc].sample(random_state=0).index)

#     testData = allData[allData['Protein Id'].isin(proteinList)]
#     trainData = allData[~allData['Protein Id'].isin(proteinList)]

#     print(allData)
#     print(testData)
#     print(testData[locationList].sum())
#     print(trainData)
#     print(trainData[locationList].sum())

#     testData.to_csv(dataDir + 'data_test.csv', index=True, mode='w')
#     trainData.to_csv(dataDir + 'data_train.csv', index=True, mode='w')

#     # trainData = pd.read_csv(dataDir + 'data_train.csv', header=0, index_col=0)
#     trainData['Fold'] = -1

#     for i in range(10):
#         unsplitData = trainData[trainData['Fold'] == -1]
#         # valProteins = []

#         # proteins = unsplitData[locationList].groupby(by=unsplitData['Protein Id']).sum().replace(0, np.nan)
#         # locations = proteins.T.stack(dropna=True)

#         # for loc in locationList:
#         #     if loc in locations:
#         #         prots = locations[loc].sample(frac=1/(10 - i), random_state=i).index
#         #         valProteins.extend(prots)

#         # valProteins = list(set(valProteins))
#         # print(len(valProteins))
#         # trainData.loc[trainData['Protein Id'].isin(valProteins), 'Fold'] = i

#         proteinList = unsplitData['Protein Id'].drop_duplicates()
#         proteinList = proteinList.sample(frac=1/(10 - i), random_state=i).to_list()
#         print(len(proteinList))
#         trainData.loc[trainData['Protein Id'].isin(proteinList), 'Fold'] = i

#     print(trainData)

#     # dataNums = []
#     for i in range(10):
#         trainFoldData = trainData[trainData['Fold'] != i].drop(['Fold'], axis=1)
#         valFoldData = trainData[trainData['Fold'] == i].drop(['Fold'], axis=1)
#         print(trainFoldData)
#         print(valFoldData)
#         # num = trainFoldData[locationList].sum().tolist()
#         # num.append(len(trainFoldData))
#         # dataNums.append(num)
#         trainFoldData.to_csv(dataDir + "data_train_fold%d.csv" % i, index=True, mode='w')
#         valFoldData.to_csv(dataDir + "data_val_fold%d.csv" % i, index=True, mode='w')
#     # num = trainData[locationList].sum().tolist()
#     # num.append(len(trainData))
#     # dataNums.append(num)
#     # dataNums = pd.DataFrame(dataNums)
#     # dataNums.columns = locationList + ["total"]
#     # print(dataNums)

#     # dataNums.to_csv(dataNumPath, index=True, mode='w')


def splitData(dataPath):
    allData = pd.read_csv(dataPath, header=0, index_col=0)
    # proteins = allData[locationList].groupby(by=allData['Protein Id']).sum().replace(0, np.nan)
    print(allData[locationList].sum().tolist())

    allData['Fold'] = -1
    for i in range(len(locationList)):
        unsplitData = allData[allData['Fold'] == -1]
        prot_cnt = unsplitData[locationList].groupby(by=allData['Protein Id']).sum().replace(0, np.nan).count(axis=0).values
        idx = np.argsort(prot_cnt)
        loc = locationList[idx[i]]
        locations = unsplitData[locationList].groupby(by=unsplitData['Protein Id']).sum().replace(0, np.nan).T.stack(dropna=True)
        if loc in locations:
            allData.loc[allData['Protein Id'].isin(locations[loc].index), 'Fold'] = -2
            prots = locations[loc].sample(frac=1/10, random_state=i).index
            if len(prots) == 0:
                prots = locations[loc].sample(random_state=i).index
            allData.loc[allData['Protein Id'].isin(prots), 'Fold'] = 0

    testData = allData[allData['Fold'] == 0].drop(['Fold'], axis=1)
    trainData = allData[allData['Fold'] != 0].drop(['Fold'], axis=1)

    train_prot_cnt = trainData[locationList].groupby(by=trainData['Protein Id']).sum().replace(0, np.nan).count(axis=0).values
    test_prot_cnt = testData[locationList].groupby(by=testData['Protein Id']).sum().replace(0, np.nan).count(axis=0).values

    print(allData)
    print(testData)
    print(testData[locationList].sum())
    print(test_prot_cnt)
    print(trainData)
    print(trainData[locationList].sum())
    print(train_prot_cnt)

    testData.to_csv(dataDir + 'data_test.csv', index=True, mode='w')
    trainData.to_csv(dataDir + 'data_train.csv', index=True, mode='w')

    # trainData = pd.read_csv(dataDir + 'data_train.csv', header=0, index_col=0)
    for k in range(5):
        trainData['Fold'] = -1

        print(locationList)
        for i in range(len(locationList)):
            unsplitData = trainData[trainData['Fold'] == -1]
            prot_cnt = unsplitData[locationList].groupby(by=unsplitData['Protein Id']).sum().replace(0, np.nan).count(axis=0).values
            idx = np.argsort(prot_cnt)
            loc = locationList[idx[i]]
            print(prot_cnt, idx, loc)
            locations = unsplitData[locationList].groupby(by=unsplitData['Protein Id']).sum().replace(0, np.nan).T.stack(dropna=True)
            if loc in locations:
                for j in range(5):
                    unsplitData = trainData[trainData['Fold'] == -1]
                    locations = unsplitData[locationList].groupby(by=unsplitData['Protein Id']).sum().replace(0, np.nan).T.stack(dropna=True)
                    prots = locations[loc].sample(frac=1/(5 - j), random_state=k*5*len(locationList)+i*5+j).index
                    trainData.loc[trainData['Protein Id'].isin(prots), 'Fold'] = j

        print(trainData)

        for i in range(5):
            trainFoldData = trainData[trainData['Fold'] != i].drop(['Fold'], axis=1)
            valFoldData = trainData[trainData['Fold'] == i].drop(['Fold'], axis=1)
            print(trainFoldData)
            print(valFoldData)
            trainFoldData.to_csv(dataDir + "data_train_split%d_fold%d.csv" % (k, i), index=True, mode='w')
            valFoldData.to_csv(dataDir + "data_val_split%d_fold%d.csv" % (k, i), index=True, mode='w')


def splitOrgan(dataPath):
    allData = pd.read_csv(dataPath, header=0, index_col=0)

    organList = ["Brain", "Connective & soft tissue", "Female tissues", "Gastrointestinal tract", "Kidney & urinary bladder", "Skin"]
    brainList = ["Caudate", "Cerebellum", "Cerebral cortex", "Hippocampus"]

    organData = allData[allData['Organ'].isin(organList)]
    organData.to_csv(dataDir + "data_organ.csv", index=True, mode='w')
    for i in range(len(organList)):
        trainFoldData = organData[organData['Organ'] != organList[i]]
        valFoldData = organData[organData['Organ'] == organList[i]]
        print(trainFoldData)
        print(valFoldData)
        trainFoldData.to_csv(dataDir + "data_train_split5_fold%d.csv" % (i), index=True, mode='w')
        valFoldData.to_csv(dataDir + "data_val_split5_fold%d.csv" % (i), index=True, mode='w')

    brainData = allData[allData['Organ'] == "Brain"]
    brainData.to_csv(dataDir + "data_brain.csv", index=True, mode='w')
    for i in range(len(brainList)):
        trainFoldData = brainData[brainData['Tissue'] != brainList[i]]
        valFoldData = brainData[brainData['Tissue'] == brainList[i]]
        print(trainFoldData)
        print(valFoldData)
        trainFoldData.to_csv(dataDir + "data_train_split6_fold%d.csv" % (i), index=True, mode='w')
        valFoldData.to_csv(dataDir + "data_val_split6_fold%d.csv" % (i), index=True, mode='w')


# def showSplitedData(dataPath):
#     allData = pd.read_csv(dataPath, header=0, index_col=0)
#     print(allData)
#     print(allData[locationList].sum())
#     allData['multiLabel'] = allData[locationList].sum(axis=1) > 1
#     multiLabelData = allData[allData['multiLabel']]
#     print("MultiLabel data Num / All data Num:", len(multiLabelData) / len(allData))
#     proteins = allData[locationList].groupby(by=allData['Protein Id'])
#     # print(proteins.sum())
#     # print(proteins.count())
#     protein_label_cnt = (proteins.sum() / proteins.count())
#     protein_dif = protein_label_cnt.replace(0, np.nan).replace(1, np.nan).dropna(axis=0, how='all').index.to_list()
#     print(protein_label_cnt[protein_label_cnt.index.isin(protein_dif)][:5])
#     print(len(protein_label_cnt[protein_label_cnt.index.isin(protein_dif)]))
#     proteinNum = proteins.sum().replace(0, np.nan).count(axis=0).values
#     print("totalDataNum:", proteinNum)
#     print(proteinNum.sum() / len(proteins))



# def dataLabelAnalysisFile(filePath):
#     data = pd.read_csv(filePath, header=0, index_col=0)
#     data['multiLabel'] = data[locationList].sum(axis=1) > 1
#     multiLabelData = data[data['multiLabel']]
#     print("Data: \n", data)
#     print("multiLabeledData: \n", multiLabelData)
#     print("multiLabeledData: \n", multiLabelData[locationList])
#     print("MultiLabel data Num / All data Num:", len(multiLabelData) / len(data))
#     print(data[locationList].sum())

#     proteins = data[locationList].groupby(by=data['Protein Id'])
#     protein_label_cnt = (proteins.sum() / proteins.count())
#     protein_dif = protein_label_cnt.replace(0, np.nan).replace(1, np.nan).dropna(axis=0, how='all').index.to_list()
#     print("protein_dif: \n", protein_label_cnt[protein_label_cnt.index.isin(protein_dif)])
#     print("protein_dif Num: ", len(protein_label_cnt[protein_label_cnt.index.isin(protein_dif)]))
#     print("protein_dif Num / Protein Num: ", len(protein_label_cnt[protein_label_cnt.index.isin(protein_dif)]) / len(proteins))

#     # dataNum = proteins.sum().replace(0, np.nan).count(axis=0).values
#     dataNum = proteins.sum().replace(0, np.nan).count(axis=0).values
#     print("totalDataNum: ", dataNum)
#     print(dataNum.sum(), " ", len(proteins))
#     multiLabelProteins = multiLabelData[locationList].groupby(by=data['Protein Id'])
#     print("MultiLabel Protein Num / Protein Num: ", len(multiLabelProteins) / len(proteins))
#     print("Label Num / Protein Num: ", dataNum.sum() / len(proteins))
#     print("---------------------------------------------------")
#     print()


def dataLabelAnalysisFile(filePath):
    data = pd.read_csv(filePath, header=0, index_col=0)
    data['multiLabel'] = data[locationList].sum(axis=1) > 1
    multiLabelData = data[data['multiLabel']]
    print("Data: \n", data)
    print("multiLabeledData: \n", multiLabelData)
    print("multiLabeledData: \n", multiLabelData[locationList])
    print("MultiLabel data Num / All data Num:", len(multiLabelData) / len(data))
    print(data[locationList].sum())

    proteins = data[locationList].groupby(by=data['Protein Id'])
    protein_label_cnt = (proteins.sum() / proteins.count())
    protein_dif = protein_label_cnt.replace(0, np.nan).replace(1, np.nan).dropna(axis=0, how='all').index.to_list()
    print("protein_dif: \n", protein_label_cnt[protein_label_cnt.index.isin(protein_dif)])
    print("protein_dif Num: ", len(protein_label_cnt[protein_label_cnt.index.isin(protein_dif)]))
    print("protein_dif Num / Protein Num: ", len(protein_label_cnt[protein_label_cnt.index.isin(protein_dif)]) / len(proteins))

    # dataNum = proteins.sum().replace(0, np.nan).count(axis=0).values
    dataNum = proteins.sum().replace(0, np.nan).count(axis=0).values
    print("totalDataNum: ", dataNum)
    print(dataNum.sum(), " ", len(proteins))
    # multiLabelProteins = multiLabelData[locationList].groupby(by=data['Protein Id'])
    proteinList = proteins.max().sum(axis=1)
    multiLabelProteins = proteinList[proteinList > 1]
    print("MultiLabel Protein Num: ", len(multiLabelProteins))
    print("MultiLabel Protein Num / Protein Num: ", len(multiLabelProteins) / len(proteins))
    print("Label Num: ", dataNum.sum())
    print("Label Num / Protein Num: ", dataNum.sum() / len(proteins))
    print("---------------------------------------------------")
    print()

    print([len(proteins)])
    print(dataNum.tolist())
    print([len(data)])
    print(data[locationList].sum().values.tolist())
    print([len(multiLabelData), len(multiLabelData) / len(data), len(multiLabelProteins), len(multiLabelProteins) / len(proteins), dataNum.sum(), dataNum.sum() / len(proteins), len(protein_label_cnt[protein_label_cnt.index.isin(protein_dif)]), len(protein_label_cnt[protein_label_cnt.index.isin(protein_dif)]) / len(proteins)])

    result = [len(proteins)] + dataNum.tolist() + [len(data)] + data[locationList].sum().values.tolist()
    result += [len(multiLabelData), len(multiLabelData) / len(data), len(multiLabelProteins), len(multiLabelProteins) / len(proteins), dataNum.sum(), dataNum.sum() / len(proteins), len(protein_label_cnt[protein_label_cnt.index.isin(protein_dif)]), len(protein_label_cnt[protein_label_cnt.index.isin(protein_dif)]) / len(proteins)]

    for i in range(1, len(locationList) + 1):
        newData = data[data[locationList].sum(axis=1) == i]
        newProteins = proteinList[proteinList == i]
        result += [len(newData), len(newData) / len(data), len(newProteins), len(newProteins) / len(proteins)]
        print(len(newProteins))

    return result
    # return [len(proteins)] + dataNum.tolist() + [len(data)] + data[locationList].sum().values.tolist() + [len(multiLabelData), len(multiLabelData) / len(data), len(multiLabelProteins), len(multiLabelProteins) / len(proteins), dataNum.sum(), dataNum.sum() / len(proteins), len(protein_label_cnt[protein_label_cnt.index.isin(protein_dif)]), len(protein_label_cnt[protein_label_cnt.index.isin(protein_dif)]) / len(proteins)]


def dataAnalysis():
    protein_cols = ['protein_' + loc for loc in ['all'] + locationList]
    image_cols = ['image_' + loc for loc in ['all'] + locationList]
    cols = ['Round', 'Fold', 'Split'] + protein_cols + image_cols + ['MultiLabel Image Num', 'MultiLabel Image Num / Image Num', 'MultiLabel Protein Num', 'MultiLabel Protein Num / Protein Num', 'Label Num', 'Label Num / Protein Num', 'protein_dif Num', 'protein_dif Num / Protein Num']
    cols += ["{} {}".format(i, col) for i in range(1, len(locationList) + 1) for col in ['Labels Image Num', 'Labels Image Ratio', 'Labels Protein Num', 'Labels Protein Ratio']]
    print(cols)
    df = pd.DataFrame(columns=cols)

    df.loc[len(df)] = [None, None, 'All'] + dataLabelAnalysisFile(dataDir + "data1_deleted.csv")
    # df.loc[len(df)] = [None, None, 'All'] + dataLabelAnalysisFile(dataDir + "data1.csv")
    df.loc[len(df)] = [None, None, 'Train'] + dataLabelAnalysisFile(dataDir + 'data_train.csv')
    df.loc[len(df)] = [None, None, 'Test'] + dataLabelAnalysisFile(dataDir + 'data_test.csv')

    for k in range(5):
        for i in range(5):
            train_path = dataDir + "data_train_split%d_fold%d.csv" % (k, i)
            val_path = dataDir + "data_val_split%d_fold%d.csv" % (k, i)
            df.loc[len(df)] = [k, i, 'Train'] + dataLabelAnalysisFile(train_path)
            df.loc[len(df)] = [k, i, 'Val'] + dataLabelAnalysisFile(val_path)

    organList = ["Brain", "Connective & soft tissue", "Female tissues", "Gastrointestinal tract", "Kidney & urinary bladder", "Skin"]
    brainList = ["Caudate", "Cerebellum", "Cerebral cortex", "Hippocampus"]

    df.loc[len(df)] = [None, None, 'organ'] + dataLabelAnalysisFile(dataDir + "data_organ.csv")

    for i in range(len(organList)):
        train_path = dataDir + "data_train_split5_fold%d.csv" % (i)
        val_path = dataDir + "data_val_split5_fold%d.csv" % (i)
        df.loc[len(df)] = [5, i, 'Train'] + dataLabelAnalysisFile(train_path)
        df.loc[len(df)] = [5, i, 'Val'] + dataLabelAnalysisFile(val_path)

    df.loc[len(df)] = [None, None, 'brain'] + dataLabelAnalysisFile(dataDir + "data_brain.csv")

    for i in range(len(brainList)):
        train_path = dataDir + "data_train_split6_fold%d.csv" % (i)
        val_path = dataDir + "data_val_split6_fold%d.csv" % (i)
        df.loc[len(df)] = [6, i, 'Train'] + dataLabelAnalysisFile(train_path)
        df.loc[len(df)] = [6, i, 'Val'] + dataLabelAnalysisFile(val_path)

    for k in range(7, 17):
        for i in range(5):
            train_path = dataDir + "data_train_split%d_fold%d.csv" % (k, i)
            val_path = dataDir + "data_val_split%d_fold%d.csv" % (k, i)
            df.loc[len(df)] = [k, i, 'Train'] + dataLabelAnalysisFile(train_path)
            df.loc[len(df)] = [k, i, 'Val'] + dataLabelAnalysisFile(val_path)

    print(df)
    df.to_csv(dataDir + "data_IHC_analysis.csv", index=None, mode='w')



# def dataLabelAnalysis():
#     normalLabeledData = pd.read_csv(dataDir + "normalLabeled.csv", header=0, index_col=0)
#     normalLabeledData = normalLabeledData.drop(['Cell Type', 'Intensity Level', 'Location', 'Sex', 'Age', 'Patient Id'], axis=1)
#     normalLabeledData['multiLabel'] = normalLabeledData[locationList].sum(axis=1) > 1
#     multiLabelData = normalLabeledData[normalLabeledData['multiLabel']]
#     print("normalLabeledData: \n", normalLabeledData)
#     print("multiLabeledData: \n", multiLabelData)
#     print("MultiLabel data Num / All data Num:", len(multiLabelData) / len(normalLabeledData))
#     print(normalLabeledData[locationList].sum())

#     proteins = normalLabeledData[locationList].groupby(by=normalLabeledData['Protein Id'])
#     protein_label_cnt = (proteins.sum() / proteins.count())
#     protein_dif = protein_label_cnt.replace(0, np.nan).replace(1, np.nan).dropna(axis=0, how='all').index.to_list()
#     print("protein_dif: \n", protein_label_cnt[protein_label_cnt.index.isin(protein_dif)])
#     print("protein_dif Num: ", len(protein_label_cnt[protein_label_cnt.index.isin(protein_dif)]))
#     print("protein_dif Num / Protein Num: ", len(protein_label_cnt[protein_label_cnt.index.isin(protein_dif)]) / len(proteins))

#     dataNum = proteins.sum().replace(0, np.nan).count(axis=0).values
#     print("totalDataNum: ", dataNum)
#     print("Label Num / Protein Num: ", dataNum.sum() / len(proteins))
#     print("---------------------------------------------------")
#     print()


#     data0 = normalLabeledData[
#         (normalLabeledData['IF Verification'].isin(['enhanced'])) &
#         (normalLabeledData['Staining Level'].str.contains('high')) &
#         (normalLabeledData['Quantity'].str.contains(r'>75%|75%-25%'))]
#     multiLabelData0 = data0[data0['multiLabel']]
#     print("data0: \n", data0)
#     print("multiLabeledData0: \n", multiLabelData0)
#     print("MultiLabel data0 Num / data0 Num:", len(multiLabelData0) / len(data0))
#     print(data0[locationList].sum())

#     proteins = data0[locationList].groupby(by=data0['Protein Id'])
#     protein_label_cnt = (proteins.sum() / proteins.count())
#     protein_dif = protein_label_cnt.replace(0, np.nan).replace(1, np.nan).dropna(axis=0, how='all').index.to_list()
#     print("protein_dif: \n", protein_label_cnt[protein_label_cnt.index.isin(protein_dif)])
#     print("protein_dif Num: ", len(protein_label_cnt[protein_label_cnt.index.isin(protein_dif)]))
#     print("protein_dif Num / Protein Num: ", len(protein_label_cnt[protein_label_cnt.index.isin(protein_dif)]) / len(proteins))

#     dataNum = proteins.sum().replace(0, np.nan).count(axis=0).values
#     print("totalDataNum: ", dataNum)
#     print("Label Num / Protein Num: ", dataNum.sum() / len(proteins))
#     print("---------------------------------------------------")
#     print()


#     data1 = normalLabeledData[(normalLabeledData['IF Verification'].isin(['enhanced']))]
#     multiLabelData1 = data1[data1['multiLabel']]
#     print("data1: \n", data1)
#     print("multiLabeledData1: \n", multiLabelData1)
#     print("MultiLabel data1 Num / data1 Num:", len(multiLabelData1) / len(data1))
#     print(data1[locationList].sum())

#     proteins = data1[locationList].groupby(by=data1['Protein Id'])
#     protein_label_cnt = (proteins.sum() / proteins.count())
#     protein_dif = protein_label_cnt.replace(0, np.nan).replace(1, np.nan).dropna(axis=0, how='all').index.to_list()
#     print("protein_dif: \n", protein_label_cnt[protein_label_cnt.index.isin(protein_dif)])
#     print("protein_dif Num: ", len(protein_label_cnt[protein_label_cnt.index.isin(protein_dif)]))
#     print("protein_dif Num / Protein Num: ", len(protein_label_cnt[protein_label_cnt.index.isin(protein_dif)]) / len(proteins))

#     dataNum = proteins.sum().replace(0, np.nan).count(axis=0).values
#     print("totalDataNum: ", dataNum)
#     print("Label Num / Protein Num: ", dataNum.sum() / len(proteins))
#     print("---------------------------------------------------")
#     print()


#     data2 = normalLabeledData[
#         (normalLabeledData['Reliability Verification'].isin(['enhanced', 'supported', 'approved'])) &
#         (normalLabeledData['IF Verification'].isin(['enhanced'])) &
#         (normalLabeledData['Staining Level'].str.contains('high|medium')) &
#         (normalLabeledData['Quantity'].str.contains(r'>75%'))]
#     multiLabelData2 = data2[data2['multiLabel']]
#     print("data2: \n", data2)
#     print("multiLabeledData2: \n", multiLabelData2)
#     print("MultiLabel data2 Num / data2 Num:", len(multiLabelData2) / len(data2))
#     print(data2[locationList].sum())

#     proteins = data2[locationList].groupby(by=data2['Protein Id'])
#     protein_label_cnt = (proteins.sum() / proteins.count())
#     protein_dif = protein_label_cnt.replace(0, np.nan).replace(1, np.nan).dropna(axis=0, how='all').index.to_list()
#     print("protein_dif: \n", protein_label_cnt[protein_label_cnt.index.isin(protein_dif)])
#     print("protein_dif Num: ", len(protein_label_cnt[protein_label_cnt.index.isin(protein_dif)]))
#     print("protein_dif Num / Protein Num: ", len(protein_label_cnt[protein_label_cnt.index.isin(protein_dif)]) / len(proteins))

#     dataNum = proteins.sum().replace(0, np.nan).count(axis=0).values
#     print("totalDataNum: ", dataNum)
#     print("Label Num / Protein Num: ", dataNum.sum() / len(proteins))
#     print("---------------------------------------------------")
#     print()


#     data3 = normalLabeledData[
#         (normalLabeledData['Reliability Verification'].isin(['enhanced', 'supported'])) &
#         (normalLabeledData['IF Verification'].isin(['enhanced', 'supported'])) &
#         (normalLabeledData['Staining Level'].str.contains('high')) &
#         (normalLabeledData['Quantity'].str.contains(r'>75%'))]
#     multiLabelData3 = data3[data3['multiLabel']]
#     print("data3: \n", data3)
#     print("multiLabeledData3: \n", multiLabelData3)
#     print("MultiLabel data3 Num / data3 Num:", len(multiLabelData3) / len(data3))
#     print(data3[locationList].sum())

#     proteins = data3[locationList].groupby(by=data3['Protein Id'])
#     protein_label_cnt = (proteins.sum() / proteins.count())
#     protein_dif = protein_label_cnt.replace(0, np.nan).replace(1, np.nan).dropna(axis=0, how='all').index.to_list()
#     print("protein_dif: \n", protein_label_cnt[protein_label_cnt.index.isin(protein_dif)])
#     print("protein_dif Num: ", len(protein_label_cnt[protein_label_cnt.index.isin(protein_dif)]))
#     print("protein_dif Num / Protein Num: ", len(protein_label_cnt[protein_label_cnt.index.isin(protein_dif)]) / len(proteins))

#     dataNum = proteins.sum().replace(0, np.nan).count(axis=0).values
#     print("totalDataNum: ", dataNum)
#     print("Label Num / Protein Num: ", dataNum.sum() / len(proteins))
#     print("---------------------------------------------------")
#     print()


#     data4 = normalLabeledData[
#         # (normalLabeledData['Reliability Verification'].isin(['enhanced', 'supported'])) &
#         (normalLabeledData['Reliability Verification'].isin(['enhanced', 'supported', 'approved'])) &
#         (normalLabeledData['IF Verification'].isin(['enhanced', 'supported'])) &
#         # (normalLabeledData['Staining Level'].str.contains('high|medium|low')) &
#         (normalLabeledData['Staining Level'].str.contains('high')) &
#         (normalLabeledData['Quantity'].str.contains(r'>75%'))]
#         # 1]
#     multiLabelData4 = data4[data4['multiLabel']]
#     print("data4: \n", data4)
#     print("multiLabeledData4: \n", multiLabelData4)
#     print("MultiLabel data4 Num / data4 Num:", len(multiLabelData4) / len(data4))
#     print(data4[locationList].sum())

#     proteins = data4[locationList].groupby(by=data4['Protein Id'])
#     protein_label_cnt = (proteins.sum() / proteins.count())
#     protein_dif = protein_label_cnt.replace(0, np.nan).replace(1, np.nan).dropna(axis=0, how='all').index.to_list()
#     print("protein_dif: \n", protein_label_cnt[protein_label_cnt.index.isin(protein_dif)])
#     print("protein_dif Num: ", len(protein_label_cnt[protein_label_cnt.index.isin(protein_dif)]))
#     print("protein_dif Num / Protein Num: ", len(protein_label_cnt[protein_label_cnt.index.isin(protein_dif)]) / len(proteins))

#     dataNum = proteins.sum().replace(0, np.nan).count(axis=0).values
#     print("totalDataNum: ", dataNum)
#     print("Label Num / Protein Num: ", dataNum.sum() / len(proteins))
#     print("---------------------------------------------------")
#     print()
#         # (normalLabeledData['Quantity'].str.contains(r'>75%'))]

#     # # data1 = normalLabeledData[(normalLabeledData['Reliability Verification'] == 'enhanced') & (normalLabeledData['IF Verification'] == 'enhanced') &
#     # #     (normalLabeledData['Staining Level'].str.contains('high'))]
#     # # data1 = normalLabeledData[(normalLabeledData['Reliability Verification'].isin(['enhanced', 'supported', 'approved'])) &
#     # # data1 = normalLabeledData[
#     # #     (normalLabeledData['Reliability Verification'].isin(['enhanced', 'supported', 'approved'])) &
#     # #     (normalLabeledData['IF Verification'].isin(['enhanced'])) &
#     # #     (normalLabeledData['Staining Level'].str.contains('high|medium')) &
#     # #     (normalLabeledData['Quantity'].str.contains(r'>75%'))]
#     # data1 = normalLabeledData[
#     #     # (normalLabeledData['Reliability Verification'].isin(['enhanced', 'supported', 'approved'])) &
#     #     (normalLabeledData['IF Verification'].isin(['enhanced'])) &
#     #     (normalLabeledData['Staining Level'].str.contains('high')) &
#     #     (normalLabeledData['Quantity'].str.contains(r'>75%|75%-25%'))]
#     #     # (normalLabeledData['Quantity'].str.contains(r'>75%'))]
#     # # data1 = normalLabeledData[
#     # #     (normalLabeledData['Reliability Verification'].isin(['enhanced', 'supported'])) &
#     # #     (normalLabeledData['IF Verification'].isin(['enhanced', 'supported'])) &
#     # #     (normalLabeledData['Staining Level'].str.contains('high')) &
#     # #     (normalLabeledData['Quantity'].str.contains(r'>75%'))]
#     # # data1 = normalLabeledData[
#     # #     (normalLabeledData['IF Verification'].isin(['enhanced', 'supported'])) &
#     # #     (normalLabeledData['Staining Level'].str.contains('high|medium')) &
#     # #     (normalLabeledData['Quantity'].str.contains(r'>75%'))]
#     # print(data1)
#     # print(data1[locationList].sum())
#     # proteins = data1[locationList].groupby(by=data1['Protein Id']).sum()
#     # proteins = proteins.replace(0, np.nan)
#     # trainNum = proteins.count(axis=0).values
#     # print("totalDataNum:", trainNum)
#     # print(trainNum.sum() / len(proteins))


# def countData():
#     for k in range(5):
#         dataNums = []
#         for i in range(5):
#             trainData = pd.read_csv(dataDir + "data_train_split%d_fold%d.csv" % (k, i), header=0, index_col=0)
#             num = trainData[locationList].sum().tolist()
#             num.append(len(trainData))
#             dataNums.append(num)
#         dataNums = pd.DataFrame(dataNums)
#         dataNums.columns = locationList + ["total"]
#         print(dataNums)
#         dataNums.to_csv(dataDir + "data_num_split%d.csv" % (k), index=True, mode='w')


#     organList = ["Brain", "Connective & soft tissue", "Female tissues", "Gastrointestinal tract", "Kidney & urinary bladder", "Skin"]
#     brainList = ["Caudate", "Cerebellum", "Cerebral cortex", "Hippocampus"]

#     dataNums = []
#     for i in range(len(organList)):
#         trainData = pd.read_csv(dataDir + "data_train_split5_fold%d.csv" % (i), header=0, index_col=0)
#         num = trainData[locationList].sum().tolist()
#         num.append(len(trainData))
#         dataNums.append(num)
#     dataNums = pd.DataFrame(dataNums)
#     dataNums.columns = locationList + ["total"]
#     print(dataNums)
#     dataNums.to_csv(dataDir + "data_num_split5.csv", index=True, mode='w')

#     dataNums = []
#     for i in range(len(brainList)):
#         trainData = pd.read_csv(dataDir + "data_train_split6_fold%d.csv" % (i), header=0, index_col=0)
#         num = trainData[locationList].sum().tolist()
#         num.append(len(trainData))
#         dataNums.append(num)
#     dataNums = pd.DataFrame(dataNums)
#     dataNums.columns = locationList + ["total"]
#     print(dataNums)
#     dataNums.to_csv(dataDir + "data_num_split6.csv", index=True, mode='w')

def reSampleData():
    # # nucleus下采样1/2
    # for k in range(5):
    #     allData = []
    #     for i in range(5):
    #         valData = pd.read_csv(dataDir + "data_val_split%d_fold%d.csv" % (k, i), header=0, index_col=0)
    #         print(valData[locationList].sum())
    #         sampledIdx = valData[valData['nucleus'] == 1].sample(frac=1/2, random_state=i).index
    #         valData.loc[sampledIdx, 'nucleus'] = 0
    #         valData = valData[valData[locationList].sum(axis=1) != 0]
    #         print(valData[locationList].sum())
    #         valData['Fold'] = i
    #         print(valData)
    #         allData.append(valData)

    #     allData = pd.concat(allData, axis=0)
    #     print(allData)

    #     for i in range(5):
    #         trainFoldData = allData[allData['Fold'] != i].drop(['Fold'], axis=1)
    #         valFoldData = allData[allData['Fold'] == i].drop(['Fold'], axis=1)
    #         print(trainFoldData)
    #         print(valFoldData)
    #         trainFoldData.to_csv(dataDir + "data_train_split%d_fold%d.csv" % (k + 7, i), index=True, mode='w')
    #         valFoldData.to_csv(dataDir + "data_val_split%d_fold%d.csv" % (k + 7, i), index=True, mode='w')


    for k in range(5):
        allData = []
        for i in range(5):
            valData = pd.read_csv(dataDir + "data_val_split%d_fold%d.csv" % (k, i), header=0, index_col=0)
            print(valData[locationList].sum())
            sampledIdx = valData[valData['nucleus'] == 1].sample(frac=2/3, random_state=i).index
            valData.loc[sampledIdx, 'nucleus'] = 0
            valData = valData[valData[locationList].sum(axis=1) != 0]
            print(valData[locationList].sum())
            valData['Fold'] = i
            print(valData)
            allData.append(valData)

        allData = pd.concat(allData, axis=0)
        print(allData)

        for i in range(5):
            trainFoldData = allData[allData['Fold'] != i].drop(['Fold'], axis=1)
            valFoldData = allData[allData['Fold'] == i].drop(['Fold'], axis=1)
            print(trainFoldData)
            print(valFoldData)
            trainFoldData.to_csv(dataDir + "data_train_split%d_fold%d.csv" % (k + 12, i), index=True, mode='w')
            valFoldData.to_csv(dataDir + "data_val_split%d_fold%d.csv" % (k + 12, i), index=True, mode='w')


def nucleusDataAnalysis():
    protein_cols = ['protein_' + loc for loc in ['all'] + locationList]
    image_cols = ['image_' + loc for loc in ['all'] + locationList]
    cols = ['Round', 'Fold', 'Split'] + protein_cols + image_cols + ['MultiLabel Image Num', 'MultiLabel Image Num / Image Num', 'MultiLabel Protein Num', 'MultiLabel Protein Num / Protein Num', 'Label Num', 'Label Num / Protein Num', 'protein_dif Num', 'protein_dif Num / Protein Num']
    cols += ["{} {}".format(i, col) for i in range(1, len(locationList) + 1) for col in ['Labels Image Num', 'Labels Image Ratio', 'Labels Protein Num', 'Labels Protein Ratio']]
    print(cols)
    df = pd.DataFrame(columns=cols)

    df.loc[len(df)] = [None, None, 'All'] + nucluesDataLabelAnalysisFile(dataDir + "data1_deleted.csv")
    # df.loc[len(df)] = [None, None, 'All'] + nucluesDataLabelAnalysisFile(dataDir + "data1.csv")
    df.loc[len(df)] = [None, None, 'Train'] + nucluesDataLabelAnalysisFile(dataDir + 'data_train.csv')
    df.loc[len(df)] = [None, None, 'Test'] + nucluesDataLabelAnalysisFile(dataDir + 'data_test.csv')

    for k in range(5):
        for i in range(5):
            train_path = dataDir + "data_train_split%d_fold%d.csv" % (k, i)
            val_path = dataDir + "data_val_split%d_fold%d.csv" % (k, i)
            df.loc[len(df)] = [k, i, 'Train'] + nucluesDataLabelAnalysisFile(train_path)
            df.loc[len(df)] = [k, i, 'Val'] + nucluesDataLabelAnalysisFile(val_path)

    organList = ["Brain", "Connective & soft tissue", "Female tissues", "Gastrointestinal tract", "Kidney & urinary bladder", "Skin"]
    brainList = ["Caudate", "Cerebellum", "Cerebral cortex", "Hippocampus"]

    for i in range(len(organList)):
        train_path = dataDir + "data_train_split5_fold%d.csv" % (i)
        val_path = dataDir + "data_val_split5_fold%d.csv" % (i)
        df.loc[len(df)] = [5, i, 'Train'] + nucluesDataLabelAnalysisFile(train_path)
        df.loc[len(df)] = [5, i, 'Val'] + nucluesDataLabelAnalysisFile(val_path)

    for i in range(len(brainList)):
        train_path = dataDir + "data_train_split6_fold%d.csv" % (i)
        val_path = dataDir + "data_val_split6_fold%d.csv" % (i)
        df.loc[len(df)] = [6, i, 'Train'] + nucluesDataLabelAnalysisFile(train_path)
        df.loc[len(df)] = [6, i, 'Val'] + nucluesDataLabelAnalysisFile(val_path)

    for k in range(7, 12):
        for i in range(5):
            train_path = dataDir + "data_train_split%d_fold%d.csv" % (k, i)
            val_path = dataDir + "data_val_split%d_fold%d.csv" % (k, i)
            df.loc[len(df)] = [k, i, 'Train'] + nucluesDataLabelAnalysisFile(train_path)
            df.loc[len(df)] = [k, i, 'Val'] + nucluesDataLabelAnalysisFile(val_path)

    print(df)
    df.to_csv(dataDir + "data_nucleus_analysis.csv", index=None, mode='w')


def nucluesDataLabelAnalysisFile(filePath):
    data = pd.read_csv(filePath, header=0, index_col=0)

    # data = data[data['nucleus'] == 1]
    nucleusData = data[data['nucleus'] == 1]
    nucleusProteins = nucleusData['Protein Id'].drop_duplicates()
    data = data[data['Protein Id'].isin(nucleusProteins)]

    data['multiLabel'] = data[locationList].sum(axis=1) > 1
    multiLabelData = data[data['multiLabel']]
    print("Data: \n", data)
    print("multiLabeledData: \n", multiLabelData)
    print("multiLabeledData: \n", multiLabelData[locationList])
    print("MultiLabel data Num / All data Num:", len(multiLabelData) / len(data))
    print(data[locationList].sum())

    proteins = data[locationList].groupby(by=data['Protein Id'])
    protein_label_cnt = (proteins.sum() / proteins.count())
    protein_dif = protein_label_cnt.replace(0, np.nan).replace(1, np.nan).dropna(axis=0, how='all').index.to_list()
    print("protein_dif: \n", protein_label_cnt[protein_label_cnt.index.isin(protein_dif)])
    print("protein_dif Num: ", len(protein_label_cnt[protein_label_cnt.index.isin(protein_dif)]))
    print("protein_dif Num / Protein Num: ", len(protein_label_cnt[protein_label_cnt.index.isin(protein_dif)]) / len(proteins))

    # dataNum = proteins.sum().replace(0, np.nan).count(axis=0).values
    dataNum = proteins.sum().replace(0, np.nan).count(axis=0).values
    print("totalDataNum: ", dataNum)
    print(dataNum.sum(), " ", len(proteins))
    # multiLabelProteins = multiLabelData[locationList].groupby(by=data['Protein Id'])
    proteinList = proteins.max().sum(axis=1)
    multiLabelProteins = proteinList[proteinList > 1]
    print("MultiLabel Protein Num: ", len(multiLabelProteins))
    print("MultiLabel Protein Num / Protein Num: ", len(multiLabelProteins) / len(proteins))
    print("Label Num: ", dataNum.sum())
    print("Label Num / Protein Num: ", dataNum.sum() / len(proteins))
    print("---------------------------------------------------")
    print()

    print([len(proteins)])
    print(dataNum.tolist())
    print([len(data)])
    print(data[locationList].sum().values.tolist())
    print([len(multiLabelData), len(multiLabelData) / len(data), len(multiLabelProteins), len(multiLabelProteins) / len(proteins), dataNum.sum(), dataNum.sum() / len(proteins), len(protein_label_cnt[protein_label_cnt.index.isin(protein_dif)]), len(protein_label_cnt[protein_label_cnt.index.isin(protein_dif)]) / len(proteins)])

    result = [len(proteins)] + dataNum.tolist() + [len(data)] + data[locationList].sum().values.tolist()
    result += [len(multiLabelData), len(multiLabelData) / len(data), len(multiLabelProteins), len(multiLabelProteins) / len(proteins), dataNum.sum(), dataNum.sum() / len(proteins), len(protein_label_cnt[protein_label_cnt.index.isin(protein_dif)]), len(protein_label_cnt[protein_label_cnt.index.isin(protein_dif)]) / len(proteins)]

    for i in range(1, len(locationList) + 1):
        newData = data[data[locationList].sum(axis=1) == i]
        newProteins = proteinList[proteinList == i]
        result += [len(newData), len(newData) / len(data), len(newProteins), len(newProteins) / len(proteins)]
        print(len(newProteins))

    return result
    # return [len(proteins)] + dataNum.tolist() + [len(data)] + data[locationList].sum().values.tolist() + [len(multiLabelData), len(multiLabelData) / len(data), len(multiLabelProteins), len(multiLabelProteins) / len(proteins), dataNum.sum(), dataNum.sum() / len(proteins), len(protein_label_cnt[protein_label_cnt.index.isin(protein_dif)]), len(protein_label_cnt[protein_label_cnt.index.isin(protein_dif)]) / len(proteins)]



if __name__ == '__main__':
    """ 1. 数据探索性分析 """
    # tissueFilePath = dataDir + "tissueUrl.csv"
    # tissueSavePath = dataDir + "tissueAnalysis.csv"
    # tissueAnalysis(tissueFilePath, tissueSavePath)
    # pathologyFilePath = dataDir + "pathologyUrl.csv"
    # pathologySavePath = dataDir + "pathologyAnalysis.csv"
    # pathologyAnalysis(pathologyFilePath, pathologySavePath)

    """ 2. 匹配图像数据与亚细胞位置标签,按['Protein Name', 'Protein Id', 'Antibody Id', 'IF Verification', 'Organ'] """
    # tissueFilePath = dataDir + "tissueUrl.csv"
    # locationFilePath = dataDir + "locationOneHot.csv"
    # newLocationSavePath = dataDir + "normalWithAnnotation.csv"
    # newAnnotation = getAnnotation(tissueFilePath, locationFilePath, newLocationSavePath)

    """ 3. 数据分组，并记录组Idx，按['Protein Name', 'Protein Id', 'Antibody Id', 'Tissue', 'Organ'] """
    # newLocationSavePath = dataDir + "normalWithAnnotation.csv"
    # pathologyFilePath = dataDir + "pathologyUrl.csv"
    # getPairIdx(newLocationSavePath, pathologyFilePath)

    """ 4. 另存有标签数据，并对有标签数据进行数据分析 """
    # newLocationSavePath = dataDir + "normalWithAnnotation.csv"
    # newAnnotation = pd.read_csv(newLocationSavePath, header=0, index_col=0)
    # labeledSavePath = dataDir + "normalLabeled.csv"
    # annotationSavePath = dataDir + "annotationAnalysis.csv"
    # annotationAnalysis(newAnnotation, labeledSavePath, annotationSavePath)

    """ 5. 数据筛选，按数据可靠性对数据集进行划分 """
    # labeledSavePath = dataDir + "normalLabeled.csv"
    # newLocationSavePath = dataDir + "normalWithAnnotation.csv"
    # pathologyFilePath = dataDir + "pathologyUrl.csv"
    # dataScreening(labeledSavePath, newLocationSavePath, pathologyFilePath)

    # normalDataPath = dataDir + 'data_E2.csv'
    # pathologyDataPath = dataDir + 'data_E3.csv'
    # pretrainDatasetPartitioning(normalDataPath, pathologyDataPath)
    # pretrainDataPath = dataDir + 'data_pretrain.csv'
    # deleteWrongData(pretrainDataPath, pretrainDataPath)

    """ 6. 去除数据集中错误的数据（图片通道数为1） """
    # dataPath = dataDir + "data_B.csv"
    # deletedDataPath = dataDir + "data_B_deleted.csv"
    # deleteWrongData(dataPath, deletedDataPath)

    # dataPath = dataDir + "data1.csv"
    # deletedDataPath = dataDir + "data1_deleted.csv"
    # deleteWrongData(dataPath, deletedDataPath)

    """ 7. 数据集划分，在蛋白质水平上对数据进行训练集、验证集划分 """
    # deletedDataPath = dataDir + "data_B_deleted.csv"
    # datasetPartitioning(deletedDataPath)
    # splitData()

    # deletedDataPath = dataDir + "data1_deleted.csv"
    # deletedDataPath = dataDir + "data1.csv"
    # datasetPartitioning(deletedDataPath)
    # splitData(deletedDataPath)

    # splitOrgan(deletedDataPath)
    # countData()

    # reSampleData()


    # print(dataDir + "data1_deleted.csv")
    # showSplitedData(dataDir + "data1_deleted.csv")
    # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    # print(dataDir + 'data_train.csv')
    # showSplitedData(dataDir + 'data_train.csv')
    # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    # print(dataDir + 'data_test.csv')
    # showSplitedData(dataDir + 'data_test.csv')
    # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    # for i in range(10):
    #     print(dataDir + "data_train_fold%d.csv" % i)
    #     showSplitedData(dataDir + "data_train_fold%d.csv" % i)
    #     print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    #     print(dataDir + "data_val_fold%d.csv" % i)
    #     showSplitedData(dataDir + "data_val_fold%d.csv" % i)
    #     print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    # dataLabelAnalysis()


    # dataLabelAnalysisFile(dataDir + "normalLabeled.csv")
    # dataLabelAnalysisFile(dataDir + "data1_deleted.csv")
    # dataLabelAnalysisFile(dataDir + 'data_train.csv')
    # dataLabelAnalysisFile(dataDir + 'data_test.csv')

    dataAnalysis()


    # nucleusDataAnalysis()


