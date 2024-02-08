import numpy as np
import pandas as pd


dataDir = "data/"
laceDNNDir = dataDir + "laceDNN/"

annotationPath = dataDir + "v18/tissueUrl.csv"
locationsPath = dataDir + "v18/subcellular_location_v18.tsv"

dataPath = laceDNNDir + "Prostate_ES_5class.csv"
splitOriginalDataPath = laceDNNDir + "splitOriginalData.csv"

laceDNNWithUrlDataPath = laceDNNDir + "laceDNN_url_data.csv"
laceDNNDataPath = laceDNNDir + "laceDNN_data.csv"

laceDNNTrainDataPath = laceDNNDir + "laceDNNTrainData.csv"
laceDNNTestDataPath = laceDNNDir + "laceDNNTestData.csv"

dataNumPath = laceDNNDir + "laceDNN_data_num.csv"

locationList = ['cytoplasm', 'cytoskeleton', 'endoplasmic reticulum', 'golgi apparatus', 'lysosomes', 'mitochondria',
                'nucleoli', 'nucleus', 'plasma membrane', 'vesicles']
locationDict = {'1': 'cytoplasm', '2': 'golgi apparatus', '3': 'mitochondria', '4': 'nucleus', '5': 'plasma membrane'}


def prepareOriginalData():
    originalData = pd.read_csv(dataPath, header=0)
    splitId = originalData['Id'].str.split("_", expand=True)
    originalData['Protein Id'] = splitId[1]
    originalData['antibody'] = splitId[2]
    print(originalData)

    data_test = pd.read_csv(laceDNNDir + "original/resnet34.0.test_val.csv", header=0)[['Id']]
    data_test['split'] = 'test'
    data_val0 = pd.read_csv(laceDNNDir + "original/resnet34.0.val.csv", header=0)[['Id']]
    data_val0['split'] = 'val0'
    data_val1 = pd.read_csv(laceDNNDir + "original/resnet34.1.val.csv", header=0)[['Id']]
    data_val1['split'] = 'val1'
    data_val2 = pd.read_csv(laceDNNDir + "original/resnet34.2.val.csv", header=0)[['Id']]
    data_val2['split'] = 'val2'
    data_val3 = pd.read_csv(laceDNNDir + "original/resnet34.3.val.csv", header=0)[['Id']]
    data_val3['split'] = 'val3'
    data_val4 = pd.read_csv(laceDNNDir + "original/resnet34.4.val.csv", header=0)[['Id']]
    data_val4['split'] = 'val4'

    data = pd.concat([data_test, data_val0, data_val1, data_val2, data_val3, data_val4])
    print(data)

    data = data.rename(columns={'Id': 'Original Id'})
    data['Id'] = data['Original Id'].str.split('/', expand=True)[0].str.split('\'', expand=True)[1]
    data['image'] = data['Original Id'].str.split('/', expand=True)[1].str.rsplit('_', n=2, expand=True)[0]
    data = data.drop_duplicates(subset=['split', 'Id', 'image'])
    print(data)

    data = pd.merge(originalData, data, on=['Id'])
    print(data)

    data.to_csv(splitOriginalDataPath, index=True, mode='w')


# def getLaceDNNData():
#     annotationData = pd.read_csv(annotationPath, header=0)
#     locationsData = pd.read_csv(locationsPath, header=0, sep='\t')

#     data = annotationData[
#         # (annotationData['IF Verification'].isin(['enhanced'])) &
#         (annotationData['Tissue'] == 'prostate') &
#         # (annotationData['IF Verification'].isin(['enhanced'])) &
#         (annotationData['Staining Level'].str.contains('High|Medium'))]

#     print(locationsData)
#     print(data)


def preparelaceDNNData2():
    annotationData = pd.read_csv(annotationPath, header=0)
    originalData = pd.read_csv(dataPath, header=0)
    splitId = originalData['Id'].str.split("_", expand=True)
    originalData['Protein Id'] = splitId[1]
    originalData['antibody'] = splitId[2]
    print(annotationData)
    print(originalData)
    print(originalData['Protein Id'].drop_duplicates())

    annotationData['antibody'] = annotationData['Antibody Id'].str.extract('(\d+)')[0]
    annotationData = annotationData[annotationData['Tissue'].str.lower() == 'prostate']
    annotationData['image'] = annotationData['URL'].str.rsplit('/', n=1, expand=True)[1].str.split('.', expand=True)[0]

    annotationData.insert(loc=3, column='Reliability Verification', value=np.nan)
    annotationData[['IF Verification', 'locations', 'IF Organ']] = np.nan

    laceDNNData = pd.merge(annotationData, originalData, on=['Protein Id', 'antibody'], how='right')
    print(laceDNNData)

    # laceDNNData['locations'] = laceDNNData['Enhanced'] + ";" + laceDNNData['Supported']
    # laceDNNData[locationList] = 0

    # location = laceDNNData['Target'].str.split(" ", expand=True).stack().rename('locations').reset_index(1, drop=True)
    # for i, v in location.items():
    #     v = locationDict[v]
    #     laceDNNData.at[i, v] = 1

    # laceDNNData["Pair Idx"] = "N-" + laceDNNData['antibody'].astype('str')
    # laceDNNData["Staining Level"] = laceDNNData['Dye']

    # print(list(laceDNNData))

    # splitInfo = laceDNNData['split']

    # laceDNNData['URL'] = laceDNNData['Protein Id'] + "/" + laceDNNData['Tissue'] + "/" + laceDNNData['Antibody Id'] + "/" + laceDNNData['URL'].str.rsplit('/', n=1, expand=True)[1]
    # laceDNNData = laceDNNData.drop(['Cell Type', 'Intensity Level', 'Location', 'Sex', 'Age', 'Patient Id'], axis=1)
    # laceDNNData = laceDNNData.drop(['antibody', 'image', 'Id', 'Dye', 'Enhanced', 'Supported', 'Target', 'Original Id', 'split'], axis=1)

    # print(laceDNNData)
    # print(splitInfo)
    # print(laceDNNData[locationList])
    # print(list(laceDNNData))
    # print(laceDNNData[locationList].sum())


def preparelaceDNNData():
    annotationData = pd.read_csv(annotationPath, header=0)
    originalData = pd.read_csv(splitOriginalDataPath, header=0, index_col=0)
    print(annotationData)
    print(originalData)
    print(originalData['Protein Id'].drop_duplicates())

    annotationData['antibody'] = annotationData['Antibody Id'].str.extract('(\d+)')[0].astype(int)
    annotationData = annotationData[annotationData['Tissue'].str.lower() == 'prostate']
    annotationData['image'] = annotationData['URL'].str.rsplit('/', n=1, expand=True)[1].str.split('.', expand=True)[0]

    annotationData.insert(loc=3, column='Reliability Verification', value=np.nan)
    annotationData[['IF Verification', 'locations', 'IF Organ']] = np.nan

    laceDNNData = pd.merge(annotationData, originalData, on=['Protein Id', 'antibody', 'image'], how='right')
    print(laceDNNData)

    laceDNNData['locations'] = laceDNNData['Enhanced'] + ";" + laceDNNData['Supported']
    laceDNNData[locationList] = 0

    location = laceDNNData['Target'].str.split(" ", expand=True).stack().rename('locations').reset_index(1, drop=True)
    for i, v in location.items():
        v = locationDict[v]
        laceDNNData.at[i, v] = 1

    laceDNNData["Pair Idx"] = "N-" + laceDNNData['antibody'].astype('str')
    laceDNNData["Staining Level"] = laceDNNData['Dye']

    print(list(laceDNNData))

    splitInfo = laceDNNData['split']

    laceDNNData['URL'] = laceDNNData['Protein Id'] + "/" + laceDNNData['Tissue'] + "/" + laceDNNData['Antibody Id'] + "/" + laceDNNData['URL'].str.rsplit('/', n=1, expand=True)[1]
    laceDNNData = laceDNNData.drop(['Cell Type', 'Intensity Level', 'Location', 'Sex', 'Age', 'Patient Id'], axis=1)
    laceDNNData = laceDNNData.drop(['antibody', 'image', 'Id', 'Dye', 'Enhanced', 'Supported', 'Target', 'Original Id', 'split'], axis=1)

    print(laceDNNData)
    print(splitInfo)
    print(laceDNNData[locationList])
    print(list(laceDNNData))
    print(laceDNNData[locationList].sum())

    laceDNNData.to_csv(laceDNNDataPath, index=True, mode='w')

    trainData = laceDNNData[splitInfo != 'test']
    testData = laceDNNData[splitInfo == 'test']

    print(trainData)
    print(testData)
    trainData.to_csv(laceDNNTrainDataPath, index=True, mode='w')
    testData.to_csv(laceDNNTestDataPath, index=True, mode='w')

    splitInfo = splitInfo[splitInfo != 'test']
    for i in range(5):
        trainFoldData = trainData[splitInfo != "val{}".format(i)]
        valFoldData = trainData[splitInfo == "val{}".format(i)]
        print(trainFoldData)
        print(valFoldData)
        trainFoldData.to_csv(laceDNNDir + "laceDNN_train_fold%d.csv" % i, index=True, mode='w')
        valFoldData.to_csv(laceDNNDir + "laceDNN_val_fold%d.csv" % i, index=True, mode='w')
        trainFoldData.to_csv(laceDNNDir + "laceDNN_train_split0_fold%d.csv" % i, index=True, mode='w')
        valFoldData.to_csv(laceDNNDir + "laceDNN_val_split0_fold%d.csv" % i, index=True, mode='w')





# def splitData1():
#     trainData = pd.read_csv(laceDNNTrainDataPath, header=0, index_col=0)

#     for k in range(1, 5):
#         trainData['Fold'] = -1

#         print(locationList)
#         for i in range(len(locationList)):
#             unsplitData = trainData[trainData['Fold'] == -1]
#             prot_cnt = unsplitData[locationList].groupby(by=unsplitData['Protein Id']).sum().replace(0, np.nan).count(axis=0).values
#             idx = np.argsort(prot_cnt)
#             loc = locationList[idx[i]]
#             print(prot_cnt, idx, loc)
#             locations = unsplitData[locationList].groupby(by=unsplitData['Protein Id']).sum().replace(0, np.nan).T.stack(dropna=True)
#             if loc in locations:
#                 for j in range(5):
#                     unsplitData = trainData[trainData['Fold'] == -1]
#                     locations = unsplitData[locationList].groupby(by=unsplitData['Protein Id']).sum().replace(0, np.nan).T.stack(dropna=True)
#                     prots = locations[loc].sample(frac=1/(5 - j), random_state=(k * 10 + j)).index
#                     trainData.loc[trainData['Protein Id'].isin(prots), 'Fold'] = j

#         print(trainData)

#         dataNums = []
#         for i in range(5):
#             trainFoldData = trainData[trainData['Fold'] != i].drop(['Fold'], axis=1)
#             valFoldData = trainData[trainData['Fold'] == i].drop(['Fold'], axis=1)
#             print(trainFoldData)
#             print(valFoldData)
#             num = trainFoldData[locationList].sum().tolist()
#             num.append(len(trainFoldData))
#             dataNums.append(num)
#             trainFoldData.to_csv(laceDNNDir + "laceDNN_train_split%d_fold%d.csv" % (k, i), index=True, mode='w')
#             valFoldData.to_csv(laceDNNDir + "laceDNN_val_split%d_fold%d.csv" % (k, i), index=True, mode='w')
#         num = trainData[locationList].sum().tolist()
#         num.append(len(trainData))
#         dataNums.append(num)
#         dataNums = pd.DataFrame(dataNums)
#         dataNums.columns = locationList + ["total"]
#         print(dataNums)

#         dataNums.to_csv(dataNumPath, index=True, mode='w')


def splitData1():
    allData = pd.read_csv(laceDNNDataPath, header=0, index_col=0)

    for k in range(1, 5):
        allData['Fold'] = -1

        print(locationList)
        for i in range(len(locationList)):
            unsplitData = allData[allData['Fold'] == -1]
            prot_cnt = unsplitData[locationList].groupby(by=unsplitData['Protein Id']).sum().replace(0, np.nan).count(axis=0).values
            idx = np.argsort(prot_cnt)
            loc = locationList[idx[i]]
            print(prot_cnt, idx, loc)
            locations = unsplitData[locationList].groupby(by=unsplitData['Protein Id']).sum().replace(0, np.nan).T.stack(dropna=True)
            if loc in locations:
                for j in range(11):
                    unsplitData = allData[allData['Fold'] == -1]
                    locations = unsplitData[locationList].groupby(by=unsplitData['Protein Id']).sum().replace(0, np.nan).T.stack(dropna=True)
                    prots = locations[loc].sample(frac=1/(11 - j), random_state=(k * 11 + j)).index
                    # prots = locations[loc].sample(frac=1/(10 - j), random_state=j).index
                    allData.loc[allData['Protein Id'].isin(prots), 'Fold'] = j


        print(allData)
        trainData = allData[allData['Fold'] != 10]
        testData = allData[allData['Fold'] == 10]
        trainData.to_csv(laceDNNDir + "laceDNN_train_split%d.csv" % (k), index=True, mode='w')
        testData.to_csv(laceDNNDir + "laceDNN_test_split%d.csv" % (k), index=True, mode='w')

        dataNums = []
        for i in range(5):
            trainFoldData = trainData[(trainData['Fold'] != 2 * i) & (trainData['Fold'] != 2 * i + 1)].drop(['Fold'], axis=1)
            valFoldData = trainData[(trainData['Fold'] == 2 * i) | (trainData['Fold'] == 2 * i + 1)].drop(['Fold'], axis=1)
            print(trainFoldData)
            print(valFoldData)
            num = trainFoldData[locationList].sum().tolist()
            num.append(len(trainFoldData))
            dataNums.append(num)
            trainFoldData.to_csv(laceDNNDir + "laceDNN_train_split%d_fold%d.csv" % (k, i), index=True, mode='w')
            valFoldData.to_csv(laceDNNDir + "laceDNN_val_split%d_fold%d.csv" % (k, i), index=True, mode='w')
        num = trainData[locationList].sum().tolist()
        num.append(len(trainData))
        dataNums.append(num)
        dataNums = pd.DataFrame(dataNums)
        dataNums.columns = locationList + ["total"]
        print(dataNums)

        dataNums.to_csv(dataNumPath, index=True, mode='w')




def getAnnotation():
    data = pd.read_csv(laceDNNWithUrlDataPath, header=0, index_col=0)
    data['URL'] = data['Protein Id'] + "/" + data['Tissue'] + "/" + data['Antibody Id'] + "/" + data['URL'].str.rsplit('/', n=1, expand=True)[1]

    data.to_csv(laceDNNDataPath, index=True, mode='w')


def splitData():
    allData = pd.read_csv(laceDNNDataPath, header=0, index_col=0)

    testProteins = []
    proteins = allData[locationList].groupby(by=allData['Protein Id']).sum().replace(0, np.nan)
    locations = proteins.T.stack(dropna=True)
    print(allData[locationList].sum().tolist())

    for loc in locationList:
        if loc in locations:
            prots = locations[loc].sample(frac=1/10, random_state=0).index
            testProteins.extend(prots)

    testProteins = list(set(testProteins))
    testData = allData[allData['Protein Id'].isin(testProteins)]
    trainData = allData[~allData['Protein Id'].isin(testProteins)]

    print(allData)
    print(testData)
    print(trainData)

    testData.to_csv(laceDNNTestDataPath, index=True, mode='w')
    trainData.to_csv(laceDNNTrainDataPath, index=True, mode='w')


    trainData['Fold'] = -1

    for i in range(10):
        unsplitData = trainData[trainData['Fold'] == -1]
        valProteins = []

        proteins = unsplitData[locationList].groupby(by=unsplitData['Protein Id']).sum().replace(0, np.nan)
        locations = proteins.T.stack(dropna=True)

        for loc in locationList:
            if loc in locations:
                prots = locations[loc].sample(frac=1/(10 - i), random_state=i).index
                valProteins.extend(prots)

        valProteins = list(set(valProteins))
        trainData.loc[trainData['Protein Id'].isin(valProteins), 'Fold'] = i

    print(trainData)

    dataNums = []
    for i in range(10):
        trainFoldData = trainData[trainData['Fold'] != i].drop(['Fold'], axis=1)
        valFoldData = trainData[trainData['Fold'] == i].drop(['Fold'], axis=1)
        print(trainFoldData)
        print(valFoldData)
        num = trainFoldData[locationList].sum().tolist()
        num.append(len(trainFoldData))
        dataNums.append(num)
        trainFoldData.to_csv(laceDNNDir + "laceDNN_train_fold%d.csv" % i, index=True, mode='w')
        valFoldData.to_csv(laceDNNDir + "laceDNN_val_fold%d.csv" % i, index=True, mode='w')
    num = trainData[locationList].sum().tolist()
    num.append(len(trainData))
    dataNums.append(num)
    dataNums = pd.DataFrame(dataNums)
    dataNums.columns = locationList + ["total"]
    print(dataNums)

    dataNums.to_csv(dataNumPath, index=True, mode='w')


def dataLabelAnalysis(filePath):
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


def dataAnalysis():
    protein_cols = ['protein_' + loc for loc in ['all'] + locationList]
    image_cols = ['image_' + loc for loc in ['all'] + locationList]
    cols = ['Round', 'Fold', 'Split'] + protein_cols + image_cols + ['MultiLabel Image Num', 'MultiLabel Image Num / Image Num', 'MultiLabel Protein Num', 'MultiLabel Protein Num / Protein Num', 'Label Num', 'Label Num / Protein Num', 'protein_dif Num', 'protein_dif Num / Protein Num']
    cols += ["{} {}".format(i, col) for i in range(1, len(locationList) + 1) for col in ['Labels Image Num', 'Labels Image Ratio', 'Labels Protein Num', 'Labels Protein Ratio']]
    df = pd.DataFrame(columns=cols)

    df.loc[len(df)] = [None, None, 'All'] + dataLabelAnalysis(laceDNNDataPath)
    # df.loc[len(df)] = [None, None, 'Train'] + dataLabelAnalysis(laceDNNTrainDataPath)
    # df.loc[len(df)] = [None, None, 'Test'] + dataLabelAnalysis(laceDNNTestDataPath)

    for k in range(5):
        df.loc[len(df)] = [k, None, 'Train'] + dataLabelAnalysis(laceDNNDir + "laceDNN_train_split%d.csv" % (k))
        df.loc[len(df)] = [k, None, 'Test'] + dataLabelAnalysis(laceDNNDir + "laceDNN_test_split%d.csv" % (k))
        for i in range(5):
            train_path = laceDNNDir + "laceDNN_train_split%d_fold%d.csv" % (k, i)
            val_path = laceDNNDir + "laceDNN_val_split%d_fold%d.csv" % (k, i)
            df.loc[len(df)] = [k, i, 'Train'] + dataLabelAnalysis(train_path)
            df.loc[len(df)] = [k, i, 'Val'] + dataLabelAnalysis(val_path)

    print(df)
    df.to_csv(laceDNNDir + "laceDNN_analysis.csv", index=None, mode='w')


def dataDiff():
    originalData = pd.read_csv(dataPath, header=0)
    splitId = originalData['Id'].str.split("_", expand=True)
    originalData['Protein Id'] = splitId[1]
    originalData['antibody'] = splitId[2]
    # dfA = originalData.drop_duplicates(['Protein Id', 'antibody'])
    dfA = originalData.drop_duplicates(['Protein Id'])
    print(dfA)

    allData = pd.read_csv(laceDNNDataPath, header=0, index_col=0)
    allData['antibody'] = allData['Antibody Id'].str.extract('(\d+)')[0]
    # dfB = allData[['Protein Id', 'antibody']].drop_duplicates(['Protein Id', 'antibody'])
    dfB = allData[['Protein Id']].drop_duplicates(['Protein Id'])
    print(dfB)

    dfC = pd.concat([dfA, dfB])
    # dfC = dfC.drop_duplicates(['Protein Id', 'antibody'], keep=False)
    dfC = dfC.drop_duplicates(['Protein Id'], keep=False)
    print(dfC)

    # dfC.to_csv(laceDNNDir + "diffData.csv", index=None, mode='w')
    dfC.to_csv(laceDNNDir + "diffProtein.csv", index=None, mode='w')



    # originalData = pd.read_csv(splitOriginalDataPath, header=0, index_col=0)
    # originalData = originalData.drop_duplicates(['Protein Id', 'antibody'])
    # print(originalData)
    # originalData = originalData.drop_duplicates(['Protein Id'])
    # print(originalData)




if __name__ == '__main__':
    # prepareOriginalData()
    # preparelaceDNNData()
    preparelaceDNNData2()
    # getAnnotation()
    # splitData()
    # splitData1()

    # dataLabelAnalysis(laceDNNDataPath)
    # dataLabelAnalysis(laceDNNTrainDataPath)
    # dataLabelAnalysis(laceDNNTestDataPath)

    # dataAnalysis()

    # getLaceDNNData()

    # dataDiff()
