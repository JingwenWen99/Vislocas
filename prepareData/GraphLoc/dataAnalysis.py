import numpy as np
import pandas as pd


dataDir = "data/"
GraphLocDir = dataDir + "GraphLoc/"

annotationPath = dataDir + "v18/tissueUrl.csv"

testGenePath = GraphLocDir + "data_split/test_genes.npy"
enhancedLabelPath = GraphLocDir + "enhanced_label.txt"

GraphLocAllDataPath = GraphLocDir + "GraphLoc_all_data.csv"
GraphLocDataPath = GraphLocDir + "GraphLoc_data.csv"
testDataPath = GraphLocDir + "GraphLoc_test.csv"
trainDataPath = GraphLocDir + "GraphLoc_train.csv"

dataNumPath = GraphLocDir + "GraphLoc_data_num.csv"

locationList = ['cytoplasm', 'cytoskeleton', 'endoplasmic reticulum', 'golgi apparatus', 'lysosomes', 'mitochondria',
                'nucleoli', 'nucleus', 'plasma membrane', 'vesicles']
locationDict = {'Endoplasmic Reticulum': 'endoplasmic reticulum', 'Golgi Apparatus': 'golgi apparatus', 'Mitochondria': 'mitochondria',
                'Plasma Membrane': 'cytoplasm', 'Vesicles': 'vesicles', 'Nuclear membrane': 'nucleus', 'Nucleoli': 'nucleus',
                'Nucleoplasm': 'nucleus', 'Cytoplasm': 'cytoplasm'}


def prepareGraphLocData():
    annotationData = pd.read_csv(annotationPath, header=0)
    print(annotationData)

    allGeneList = []
    testGeneList = np.load(testGenePath)
    testGeneList = pd.DataFrame(testGeneList)
    allGeneList.append(testGeneList)
    for i in range(10):
        trainGeneList = np.load("%sdata_split/fold%d_val_genes.npy" % (GraphLocDir, i))
        trainGeneList = pd.DataFrame(trainGeneList)
        allGeneList.append(trainGeneList)
    allGeneList = pd.concat(allGeneList)

    allGeneList.columns=['Protein Id']
    print(allGeneList)

    GraphLocData = pd.merge(annotationData, allGeneList)
    GraphLocData = GraphLocData[(GraphLocData['Intensity Level'].str.split(";", expand=True)[0].isin(['Strong', 'Moderate'])) &
                                (GraphLocData['Quantity'].str.split(";", expand=True)[0] == '>75%')]

    print(GraphLocData)

    liverData = GraphLocData[(GraphLocData['Tissue'] == 'liver')]
    print("liverData:")
    print(liverData)
    liverProtein = liverData['Protein Id'].drop_duplicates()
    print(liverProtein)

    breastData = GraphLocData[(GraphLocData['Tissue'] == 'breast')]
    print("breastData:")
    print(breastData)
    breastProtein = breastData['Protein Id'].drop_duplicates()
    print(breastProtein)

    prostateData = GraphLocData[(GraphLocData['Tissue'] == 'prostate')]
    print("prostateData:")
    print(prostateData)
    prostateProtein = prostateData['Protein Id'].drop_duplicates()
    print(prostateProtein)

    bladderData = GraphLocData[(GraphLocData['Tissue'] == 'urinary bladder')]
    print("bladderData:")
    print(bladderData)
    bladderProtein = bladderData['Protein Id'].drop_duplicates()
    print(bladderProtein)

    allData = pd.concat([liverData, breastData, prostateData, bladderData])
    print(allData)

    image = allData['URL'].drop_duplicates()
    print(image)

    allData.to_csv(GraphLocAllDataPath, index=True, mode='w')


def prepareEnhancedData():
    annotationData = pd.read_csv(annotationPath, header=0)
    print(annotationData)

    enhancedLabel = pd.read_table(enhancedLabelPath, sep = '\t', header = None, names = ['Protein Id', 'label'])
    print(enhancedLabel)

    enhancedData = pd.merge(annotationData, enhancedLabel, on=['Protein Id'], how='right')
    enhancedData = enhancedData[(enhancedData['Intensity Level'].str.split(";", expand=True)[0].isin(['Strong', 'Moderate'])) &
                                (enhancedData['Quantity'].str.split(";", expand=True)[0] == '>75%')]
    print(enhancedData)

    enhancedData = enhancedData[(enhancedData['Tissue'].isin(['liver', 'breast', 'prostate', 'urinary bladder']))]
    print(enhancedData)

    gene = enhancedData['Protein Id'].drop_duplicates()
    print(gene)


def getAnnotation():
    data = pd.read_csv(GraphLocAllDataPath, header=0, index_col=0)
    data['URL'] = data['Protein Id'] + "/" + data['Tissue'] + "/" + data['Antibody Id'] + "/" + data['URL'].str.rsplit('/', n=1, expand=True)[1]
    print("data:")
    print(data)
    enhancedLabel = pd.read_table(enhancedLabelPath, sep = '\t', header = None, names = ['Protein Id', 'label'])
    print("enhancedLabel:")
    print(enhancedLabel)

    data_index = data.index
    data = pd.merge(data, enhancedLabel, on=['Protein Id'])
    data.index = data_index

    data.insert(loc=3, column='Reliability Verification', value=np.nan)

    data[['IF Verification', 'locations', 'IF Organ']] = np.nan
    data['locations'] = data['label']
    data = data.drop(labels=['label'], axis=1)
    data[locationList] = 0

    location = data['locations'].str.split(",", expand=True).stack().rename('locations').reset_index(1, drop=True)
    for i, v in location.items():
        v = locationDict[v]
        data.at[i, v] = 1

    data = data[['Protein Name', 'Protein Id', 'Antibody Id', 'Reliability Verification', 'Tissue', 'Organ', 'Cell Type', 'Staining Level', 'Intensity Level', 'Quantity', 'Location', 'Sex', 'Age', 'Patient Id', 'SnomedParameters', 'URL', 'IF Verification', 'locations', 'IF Organ'] + locationList]
    idx = data['Protein Id'].str.extract('(\d+)')
    idx = pd.to_numeric(idx[0])
    data["Pair Idx"] = "N-" + idx.astype('str')

    # data = data.drop(['Cell Type', 'Intensity Level', 'Quantity', 'Location', 'Sex', 'Age', 'Patient Id'], axis=1)
    data = data.drop(['Cell Type', 'Intensity Level', 'Location', 'Sex', 'Age', 'Patient Id'], axis=1)
    print("data:")
    print(data)

    data.to_csv(GraphLocDataPath, index=True, mode='w')


def splitData():
    allData = pd.read_csv(GraphLocDataPath, header=0, index_col=0)

    testGeneList = np.load(testGenePath)
    testGeneList = pd.DataFrame(testGeneList)
    testGeneList.columns = ['Protein Id']
    testData = pd.merge(allData, testGeneList)
    print("testGeneList:")
    print(testGeneList)
    print("testData:")
    print(testData)
    testData.to_csv(testDataPath, index=True, mode='w')

    valGeneLists = []
    for i in range(10):
        valGeneList = np.load("%sdata_split/fold%d_val_genes.npy" % (GraphLocDir, i))
        valGeneList = pd.DataFrame(valGeneList)
        valGeneList.columns = ['Protein Id']
        valGeneLists.append(valGeneList)
        valData = pd.merge(allData, valGeneList)
        print("valGeneList_%d:" % i)
        print(valGeneList)
        print("valData_%d:" % i)
        print(valData)
        valData.to_csv(GraphLocDir + "GraphLoc_val_fold%d.csv" % i, index=True, mode='w')


    trainGeneList = pd.concat(valGeneLists)
    trainData = pd.merge(allData, trainGeneList)
    print("trainGeneList:")
    print(trainGeneList)
    print("trainData:")
    print(trainData)
    trainData.to_csv(trainDataPath, index=True, mode='w')

    dataNums = []
    for i in range(10):
        trainFoldData = trainData[~trainData['Protein Id'].isin(valGeneLists[i]['Protein Id'])]
        print(trainFoldData)
        trainFoldData.to_csv(GraphLocDir + "GraphLoc_train_fold%d.csv" % i, index=True, mode='w')
        num = trainFoldData[locationList].sum().tolist()
        num.append(len(trainFoldData))
        print(num)
        dataNums.append(num)
    num = trainData[locationList].sum().tolist()
    num.append(len(trainData))
    print(num)
    dataNums.append(num)
    dataNums = pd.DataFrame(dataNums)
    dataNums.columns = locationList + ["total"]
    print(dataNums)

    dataNums.to_csv(dataNumPath, index=True, mode='w')


def splitData1():
    allData = pd.read_csv(GraphLocDataPath, header=0, index_col=0)

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
                    prots = locations[loc].sample(frac=1/(11 - j), random_state=((k - 1) * 11 + j)).index
                    # prots = locations[loc].sample(frac=1/(10 - j), random_state=j).index
                    allData.loc[allData['Protein Id'].isin(prots), 'Fold'] = j


        print(allData)
        trainData = allData[allData['Fold'] != 10]
        testData = allData[allData['Fold'] == 10]
        trainData.to_csv(GraphLocDir + "GraphLoc_train_split%d.csv" % (k), index=True, mode='w')
        testData.to_csv(GraphLocDir + "GraphLoc_test_split%d.csv" % (k), index=True, mode='w')

        dataNums = []
        for i in range(10):
            trainFoldData = trainData[trainData['Fold'] != i].drop(['Fold'], axis=1)
            valFoldData = trainData[trainData['Fold'] == i].drop(['Fold'], axis=1)
            print(trainFoldData)
            print(valFoldData)
            num = trainFoldData[locationList].sum().tolist()
            num.append(len(trainFoldData))
            dataNums.append(num)
            trainFoldData.to_csv(GraphLocDir + "GraphLoc_train_split%d_fold%d.csv" % (k, i), index=True, mode='w')
            valFoldData.to_csv(GraphLocDir + "GraphLoc_val_split%d_fold%d.csv" % (k, i), index=True, mode='w')
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

    df.loc[len(df)] = [None, None, 'All'] + dataLabelAnalysis(GraphLocDataPath)
    # df.loc[len(df)] = [None, None, 'Train'] + dataLabelAnalysis(trainDataPath)
    # df.loc[len(df)] = [None, None, 'Test'] + dataLabelAnalysis(testDataPath)

    for k in range(5):
        df.loc[len(df)] = [k, None, 'Train'] + dataLabelAnalysis(GraphLocDir + "GraphLoc_train_split%d.csv" % (k))
        df.loc[len(df)] = [k, None, 'Test'] + dataLabelAnalysis(GraphLocDir + "GraphLoc_test_split%d.csv" % (k))
        for i in range(10):
            if k == 0:
                train_path = GraphLocDir + "GraphLoc_train_fold%d.csv" % (i)
                val_path = GraphLocDir + "GraphLoc_val_fold%d.csv" % (i)
            else:
                train_path = GraphLocDir + "GraphLoc_train_split%d_fold%d.csv" % (k, i)
                val_path = GraphLocDir + "GraphLoc_val_split%d_fold%d.csv" % (k, i)
            df.loc[len(df)] = [k, i, 'Train'] + dataLabelAnalysis(train_path)
            df.loc[len(df)] = [k, i, 'Val'] + dataLabelAnalysis(val_path)

    print(df)
    df.to_csv(GraphLocDir + "GraphLoc_analysis.csv", index=None, mode='w')




# def splitImPLocData():
#     allData = pd.read_csv(GraphLocDataPath, header=0, index_col=0)

#     # allData = allData[allData['Tissue'] != 'urinary bladder']

#     enhancedLabel = pd.read_table(enhancedLabelPath, sep = '\t', header = None, names = ['Protein Id', 'label'])
#     enhancedData = pd.merge(enhancedLabel, allData)
#     # enhancedData = pd.merge(allData, enhancedLabel)
#     enhancedGene = enhancedData['Protein Id'].drop_duplicates()
#     print(enhancedGene)

#     ratio = [0.7, 0.9]
#     spivot = int(len(enhancedGene) * ratio[0])
#     epivot = int(len(enhancedGene) * ratio[1])
#     trainGene = enhancedGene.iloc[:spivot]
#     valGene = enhancedGene.iloc[spivot:epivot]
#     testGene = enhancedGene.iloc[epivot:]

#     print(trainGene)
#     print(valGene)
#     print(testGene)

#     trainData = pd.merge(allData, trainGene)
#     valData = pd.merge(allData, valGene)
#     testData = pd.merge(allData, testGene)

#     print(trainData)
#     print(valData)
#     print(testData)

#     print(trainData['Protein Id'].drop_duplicates())
#     print(valData['Protein Id'].drop_duplicates())
#     print(testData['Protein Id'].drop_duplicates())


if __name__ == '__main__':
    # prepareGraphLocData()
    # prepareEnhancedData()
    # getAnnotation()
    # splitData()
    # splitData1()
    # splitImPLocData()

    # dataLabelAnalysis(GraphLocDataPath)
    # dataLabelAnalysis(trainDataPath)
    # dataLabelAnalysis(testDataPath)
    # for k in range(1, 5):
    #     for i in range(10):
    #         dataLabelAnalysis(GraphLocDir + "GraphLoc_train_split%d_fold%d.csv" % (k, i))
    #         dataLabelAnalysis(GraphLocDir + "GraphLoc_val_split%d_fold%d.csv" % (k, i))

    dataAnalysis()
