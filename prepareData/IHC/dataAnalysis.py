import numpy as np
import pandas as pd

from PIL import Image


RNG_SEED = 0
np.random.seed(RNG_SEED)


dataDir = "data/"
imageDir = "dataset/IHC/"
imageDir2 = "dataset/IHC/"
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
    annotationData['actin filaments'] = annotationData[['actin filaments', 'cleavage furrow', 'focal adhesion sites']].max(axis=1)
    annotationData['cytosol'] = annotationData[['aggresome', 'cytoplasmic bodies', 'cytosol', 'rods & rings']].max(axis=1)
    annotationData['centrosome'] = annotationData[['centriolar satellite', 'centrosome']].max(axis=1)
    annotationData['microtubules'] = annotationData[['cytokinetic bridge', 'microtubule ends', 'microtubules', 'midbody', 'midbody ring', 'mitotic spindle']].max(axis=1)
    annotationData['plasma membrane'] = annotationData[['cell junctions', 'plasma membrane']].max(axis=1)
    annotationData['vesicles'] = annotationData[['endosomes', 'lipid droplets', 'lysosomes', 'peroxisomes', 'vesicles']].max(axis=1)
    annotationData['lysosomes'] = 0
    annotationData['nucleoplasm'] = annotationData[['kinetochore', 'mitotic chromosome', 'nuclear bodies', 'nuclear speckles', 'nucleoplasm']].max(axis=1)
    annotationData['nucleoli'] = annotationData[['nucleoli', 'nucleoli fibrillar center', 'nucleoli rim']].max(axis=1)
    annotationData = annotationData.drop(['aggresome', 'cell junctions', 'centriolar satellite', 'cleavage furrow', 'cytokinetic bridge',
        'cytoplasmic bodies', 'endosomes', 'focal adhesion sites', 'kinetochore',
        'lipid droplets', 'microtubule ends', 'midbody', 'midbody ring', 'mitotic chromosome', 'mitotic spindle',
        'nuclear bodies', 'nuclear speckles', 'nucleoli fibrillar center', 'nucleoli rim', 'peroxisomes',
        'rods & rings'], axis=1)


    annotationData['cytoskeleton'] = annotationData[['actin filaments', 'microtubules', 'intermediate filaments']].max(axis=1)
    annotationData['nucleus'] = annotationData[['nuclear membrane', 'nucleoli', 'nucleoplasm']].max(axis=1)
    annotationData['nucleoli'] = 0
    annotationData = annotationData.drop(['actin filaments', 'microtubules', 'intermediate filaments',
        'nuclear membrane', 'nucleoplasm'], axis=1)

    annotationData['cytoplasm'] = annotationData[['centrosome', 'cytosol', 'cytoskeleton']].max(axis=1)
    annotationData['cytoskeleton'] = 0
    annotationData = annotationData.drop(['centrosome', 'cytosol'], axis=1)

    annotationData['endoplasmic reticulum'] = annotationData[['endoplasmic reticulum', 'golgi apparatus', 'vesicles']].max(axis=1)
    annotationData['golgi apparatus'] = 0
    annotationData['vesicles'] = 0

    temp = annotationData['IF Organ']
    annotationData = annotationData.drop(labels=['IF Organ'], axis=1)
    annotationData.insert(6, 'IF Organ', temp)


    """ 重置合并数据的'locations'标签 """
    merged = annotationData['locations'].str.count(';')+1 != annotationData.iloc[:,7:].sum(axis=1)
    location_list = list(annotationData)[6:]
    for item in annotationData[merged].iloc[:,7:].itertuples():
        locs = []
        for idx in range(1, len(item)):
            if item[idx] == 1:
                locs.append(location_list[idx])
        locs = ';'.join(locs)
        annotationData.iat[item[0], 5] = locs

    print(list(annotationData))
    annotationData = annotationData.reindex(columns=(list(annotationData)[:-len(locationList)] + locationList))
    print(list(annotationData))

    newAnnotation = pd.merge(normalData, annotationData, how='left')
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
        im = Image.open(im_path)
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
    normalLabeledData = normalLabeledData.drop(['Cell Type', 'Location', 'Sex', 'Age', 'Patient Id'], axis=1)
    normalData = normalData.drop(['Cell Type', 'Location', 'Sex', 'Age', 'Patient Id'], axis=1)
    print("normalLabeledData:", normalLabeledData)
    print("normalData:", normalData)

    # data-6029
    data1 = normalLabeledData[
        (normalLabeledData['IF Verification'].isin(['enhanced'])) &
        (normalLabeledData['Intensity Level'].str.contains('strong')) &
        (normalLabeledData['Quantity'].str.contains(r'>75%'))]

    print(data1)
    print(data1[locationList].sum())
    proteins = data1[locationList].groupby(by=data1['Protein Id']).sum()
    proteins = proteins.replace(0, np.nan)
    trainNum = proteins.count(axis=0).values
    print("totalDataNum:", trainNum)
    print(trainNum.sum() / len(proteins))
    data1.to_csv(dataDir + 'data1.csv', index=True, mode='w')


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



if __name__ == '__main__':
    """ 1. 数据探索性分析 """
    tissueFilePath = dataDir + "tissueUrl.csv"
    tissueSavePath = dataDir + "tissueAnalysis.csv"
    tissueAnalysis(tissueFilePath, tissueSavePath)
    pathologyFilePath = dataDir + "pathologyUrl.csv"
    pathologySavePath = dataDir + "pathologyAnalysis.csv"
    pathologyAnalysis(pathologyFilePath, pathologySavePath)

    """ 2. 匹配图像数据与亚细胞位置标签,按['Protein Name', 'Protein Id', 'Antibody Id', 'IF Verification', 'Organ'] """
    tissueFilePath = dataDir + "tissueUrl.csv"
    locationFilePath = dataDir + "locationOneHot.csv"
    newLocationSavePath = dataDir + "normalWithAnnotation.csv"
    newAnnotation = getAnnotation(tissueFilePath, locationFilePath, newLocationSavePath)

    """ 3. 数据分组，并记录组Idx，按['Protein Name', 'Protein Id', 'Antibody Id', 'Tissue', 'Organ'] """
    newLocationSavePath = dataDir + "normalWithAnnotation.csv"
    pathologyFilePath = dataDir + "pathologyUrl.csv"
    getPairIdx(newLocationSavePath, pathologyFilePath)

    """ 4. 另存有标签数据，并对有标签数据进行数据分析 """
    newLocationSavePath = dataDir + "normalWithAnnotation.csv"
    newAnnotation = pd.read_csv(newLocationSavePath, header=0, index_col=0)
    labeledSavePath = dataDir + "normalLabeled.csv"
    annotationSavePath = dataDir + "annotationAnalysis.csv"
    annotationAnalysis(newAnnotation, labeledSavePath, annotationSavePath)

    """ 5. 数据筛选，按数据可靠性对数据集进行划分 """
    labeledSavePath = dataDir + "normalLabeled.csv"
    newLocationSavePath = dataDir + "normalWithAnnotation.csv"
    pathologyFilePath = dataDir + "pathologyUrl.csv"
    dataScreening(labeledSavePath, newLocationSavePath, pathologyFilePath)

    """ 6. 去除数据集中错误的数据（图片通道数为1） """
    dataPath = dataDir + "data1.csv"
    deletedDataPath = dataDir + "data1_deleted.csv"
    deleteWrongData(dataPath, deletedDataPath)

    """ 7. 数据集划分，在蛋白质水平上对数据进行训练集、验证集划分 """
    deletedDataPath = dataDir + "data1_deleted.csv"
    deletedDataPath = dataDir + "data1.csv"
    datasetPartitioning(deletedDataPath)
    splitData(deletedDataPath)

    splitOrgan(deletedDataPath)

    dataAnalysis()



