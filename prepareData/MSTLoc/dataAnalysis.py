import numpy as np
import pandas as pd


dataDir = "data/"
# labeledPath = "normalLabeled.csv"
annotationPath = dataDir + "normalWithAnnotation.csv"
MSTLocTrainPath = dataDir + "MSTLoc/train.csv"
MSTLocTestPath = dataDir + "MSTLoc/test.csv"
MSTLocDataPath = dataDir + "MSTLoc/MSTLocData.csv"
newMSTLocDataPath = dataDir + "MSTLoc/newMSTLocData.csv"
undownloadDataPath = dataDir + "MSTLoc/undownloadData.csv"

# MSTLocTrainDataPath = dataDir + "MSTLoc/MSTLocTrain.csv"
# MSTLocTestDataPath = dataDir + "MSTLoc/MSTLocTest.csv"
MSTLocTrainDataPath = dataDir + "MSTLoc/MSTLoc_train.csv"
MSTLocTestDataPath = dataDir + "MSTLoc/MSTLoc_test.csv"

dataNumPath = dataDir + "MSTLoc/MSTLoc_data_num.csv"

duplicatePath = dataDir + "MSTLoc/duplicate.csv"

locationList = ['cytoplasm', 'cytoskeleton', 'endoplasmic reticulum', 'golgi apparatus', 'lysosomes', 'mitochondria',
                'nucleoli', 'nucleus', 'plasma membrane', 'vesicles']
replaceDict = {'Endoplasmic reticulum': 'endoplasmic reticulum', 'Golgi apparatus': 'golgi apparatus', 'Mitochondria': 'mitochondria',
            'Vesicles': 'vesicles', 'Nuclear': 'nucleus', 'Cytoplasm': 'cytoplasm'}





def prepareMSTLocData():
    annotationData = pd.read_csv(annotationPath, header=0, index_col=0)
    MSTLocTrainList = pd.read_csv(MSTLocTrainPath, header=0)
    MSTLocTestList = pd.read_csv(MSTLocTestPath, header=0)
    MSTLocTrainList["split"] = "train"
    MSTLocTestList["split"] = "test"
    MSTLocList = pd.concat([MSTLocTrainList, MSTLocTestList])

    protein_MSTLoc = MSTLocList['gene'].drop_duplicates()
    annotationData = annotationData[annotationData["Protein Id"].isin(protein_MSTLoc)]

    annotationData["imageName"] = annotationData["URL"].str.rsplit("/", n=2, expand=True)[2]
    MSTLocList["imageName"] = MSTLocList["image"].str.rsplit("/", n=2, expand=True)[2]
    MSTLocList.rename(columns={'gene': 'Protein Id'}, inplace=True)
    MSTLocData = pd.merge(annotationData, MSTLocList, on=['Protein Id','imageName'], how='right')

    nullIndex = MSTLocData["URL"].isnull()
    MSTLocData["URL"][nullIndex] = MSTLocData["Protein Id"][nullIndex] + "/" + MSTLocData["imageName"][nullIndex]
    MSTLocData["Pair Idx"][nullIndex] = "N-" + (3000000 + MSTLocData[nullIndex].index).astype(str)
    print(MSTLocData)
    MSTLocData.to_csv(MSTLocDataPath, index=True, mode='w')

    newData = MSTLocData[nullIndex]
    print(newData)
    newData.to_csv(undownloadDataPath, index=True, mode='w')


def getAnnotation():
    MSTLocData = pd.read_csv(MSTLocDataPath, header=0, index_col=0)
    MSTLocData["locations"] = MSTLocData["label"]

    MSTLocData[locationList] = 0
    location = MSTLocData["locations"].str.split(";", expand=True).stack().rename('locations').reset_index(1, drop=True)
    location = location.replace(replaceDict)
    for i, v in location.items():
        MSTLocData.at[i, v] = 1

    MSTLocTrainIndex = MSTLocData["split"] == "train"
    MSTLocTestIndex = MSTLocData["split"] == "test"

    MSTLocData = MSTLocData.drop(['Cell Type', 'Intensity Level', 'Location', 'Sex', 'Age', 'Patient Id'], axis=1)
    MSTLocData = MSTLocData.drop(['imageName', 'image', 'label', 'split'], axis=1)
    # MSTLocTrain = MSTLocData[MSTLocTrainIndex]
    # MSTLocTest = MSTLocData[MSTLocTestIndex]

    # MSTLocTrain.to_csv(MSTLocTrainDataPath, index=True, mode='w')
    # MSTLocTest.to_csv(MSTLocTestDataPath, index=True, mode='w')
    MSTLocData.to_csv(newMSTLocDataPath, index=True, mode='w')


def analysisData():
    MSTLocTrainList = pd.read_csv(MSTLocTrainPath, header=0)
    MSTLocTestList = pd.read_csv(MSTLocTestPath, header=0)
    MSTLocList = pd.concat([MSTLocTrainList, MSTLocTestList])
    print(MSTLocTrainList)
    print(MSTLocTestList)
    print(MSTLocList)

    train_protein_MSTLoc = MSTLocTrainList['gene'].drop_duplicates()
    test_protein_MSTLoc = MSTLocTestList['gene'].drop_duplicates()
    all_protein_MSTLoc = MSTLocList['gene'].drop_duplicates()
    print(train_protein_MSTLoc)
    print(test_protein_MSTLoc)
    print(all_protein_MSTLoc)

    protein_MSTLoc = pd.concat([train_protein_MSTLoc, test_protein_MSTLoc])
    print(protein_MSTLoc[protein_MSTLoc.duplicated()])

    train_url_MSTLoc = MSTLocTrainList['image'].drop_duplicates()
    test_url_MSTLoc = MSTLocTestList['image'].drop_duplicates()
    all_url_MSTLoc = MSTLocList['image'].drop_duplicates()
    print(train_url_MSTLoc)
    print(test_url_MSTLoc)
    print(all_url_MSTLoc)

    print(MSTLocTrainList['image'][MSTLocTrainList['image'].duplicated()])
    url_MSTLoc = pd.concat([train_url_MSTLoc, test_url_MSTLoc])
    print(url_MSTLoc[url_MSTLoc.duplicated()])

    train_MSTLoc = MSTLocTrainList[['gene', 'image']].drop_duplicates()
    test_MSTLoc = MSTLocTestList[['gene', 'image']].drop_duplicates()
    all_MSTLoc = MSTLocList[['gene', 'image']].drop_duplicates()
    print(train_MSTLoc)
    print(test_MSTLoc)
    print(all_MSTLoc)

    item_MSTLoc = pd.concat([train_MSTLoc, test_MSTLoc])
    print(item_MSTLoc[item_MSTLoc.duplicated()])

    MSTLocTrain = MSTLocTrainList.drop_duplicates()
    MSTLocTest = MSTLocTestList.drop_duplicates()
    MSTLoc = MSTLocList.drop_duplicates()
    print(MSTLocTrain)
    print(MSTLocTest)
    print(MSTLoc)

    duplicate = MSTLocList[MSTLocList.duplicated()]
    print(duplicate)
    duplicate.to_csv(duplicatePath, index=True, mode='w')


def labelAnalysis():
    MSTLocTrain = pd.read_csv(MSTLocTrainDataPath, header=0, index_col=0)
    MSTLocTest = pd.read_csv(MSTLocTestDataPath, header=0, index_col=0)

    print(MSTLocTrain.sum())
    print(MSTLocTest.sum())

    MSTLocTrain.drop_duplicates(subset='Protein Id', keep="first", inplace=True)
    MSTLocTest.drop_duplicates(subset='Protein Id', keep="first", inplace=True)

    print(MSTLocTrain.sum())
    print(MSTLocTest.sum())


def checkUrl():
    realUrl = []
    for i in range(1, 4):
        path = dataDir + "MSTLoc/real_url - " + str(i) + ".csv"
        df = pd.read_csv(path, header=None)
        realUrl.append(df)
    realUrl = pd.concat(realUrl)
    realUrl = realUrl.drop_duplicates()

    realUrl['prefix'] = realUrl[1].str.rsplit("/", n=2, expand=True)[0]
    realUrl = realUrl[realUrl['prefix'] == "https://images.proteinatlas.org"]
    realUrl.drop_duplicates(subset=1, keep="first", inplace=True)
    print(realUrl)


def splitData():
    trainData = pd.read_csv(MSTLocTrainDataPath, header=0, index_col=0)
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
        trainFoldData.to_csv(dataDir + "MSTLoc/MSTLoc_train_fold%d.csv" % i, index=True, mode='w')
        valFoldData.to_csv(dataDir + "MSTLoc/MSTLoc_val_fold%d.csv" % i, index=True, mode='w')
    num = trainData[locationList].sum().tolist()
    num.append(len(trainData))
    dataNums.append(num)
    dataNums = pd.DataFrame(dataNums)
    dataNums.columns = locationList + ["total"]
    print(dataNums)

    dataNums.to_csv(dataNumPath, index=True, mode='w')


def splitData1():
    trainData = pd.read_csv(MSTLocTrainDataPath, header=0, index_col=0)

    # for k in range(1, 5):
    for k in range(1):
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
                for j in range(10):
                    unsplitData = trainData[trainData['Fold'] == -1]
                    locations = unsplitData[locationList].groupby(by=unsplitData['Protein Id']).sum().replace(0, np.nan).T.stack(dropna=True)
                    # prots = locations[loc].sample(frac=1/(10 - j), random_state=((k - 1) * 10 + j)).index
                    prots = locations[loc].sample(frac=1/(10 - j), random_state=((k + 100) * 10 + j)).index
                    # prots = locations[loc].sample(frac=1/(10 - j), random_state=j).index
                    trainData.loc[trainData['Protein Id'].isin(prots), 'Fold'] = j


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
            trainFoldData.to_csv(dataDir + "MSTLoc/MSTLoc_train_split%d_fold%d.csv" % (k, i), index=True, mode='w')
            valFoldData.to_csv(dataDir + "MSTLoc/MSTLoc_val_split%d_fold%d.csv" % (k, i), index=True, mode='w')
        num = trainData[locationList].sum().tolist()
        num.append(len(trainData))
        dataNums.append(num)
        dataNums = pd.DataFrame(dataNums)
        dataNums.columns = locationList + ["total"]
        print(dataNums)

        dataNums.to_csv(dataNumPath, index=True, mode='w')



def gen_adj():
    trainData = pd.read_csv(MSTLocTrainDataPath, header=0, index_col=0)
    adj = []
    for row in locationList:
        nums = []
        for col in locationList:
            nums.append(len(trainData[(trainData[row] == 1) & (trainData[col] == 1)]))
        adj.append(nums)
    adj = pd.DataFrame(adj, columns=locationList, index=locationList)
    print(adj)
    adj.to_csv(dataDir + "MSTLoc/MSTLoc_adj_train.csv", index=True, mode='w')

    for i in range(10):
        trainData = pd.read_csv(dataDir + "MSTLoc/MSTLoc_train_fold%d.csv" % i, header=0, index_col=0)
        adj = []
        for row in locationList:
            nums = []
            for col in locationList:
                nums.append(len(trainData[(trainData[row] == 1) & (trainData[col] == 1)]))
            adj.append(nums)
        adj = pd.DataFrame(adj, columns=locationList, index=locationList)
        print(adj)
        adj.to_csv(dataDir + "MSTLoc/MSTLoc_adj_train_fold%d.csv" % i, index=True, mode='w')


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

    df.loc[len(df)] = [None, None, 'All'] + dataLabelAnalysis(newMSTLocDataPath)
    df.loc[len(df)] = [None, None, 'Train'] + dataLabelAnalysis(MSTLocTrainDataPath)
    df.loc[len(df)] = [None, None, 'Test'] + dataLabelAnalysis(MSTLocTestDataPath)

    for k in range(5):
        for i in range(10):
            # if k == 0:
            #     train_path = dataDir + "MSTLoc/MSTLoc_train_fold%d.csv" % (i)
            #     val_path = dataDir + "MSTLoc/MSTLoc_val_fold%d.csv" % (i)
            # else:
            #     train_path = dataDir + "MSTLoc/MSTLoc_train_split%d_fold%d.csv" % (k, i)
            #     val_path = dataDir + "MSTLoc/MSTLoc_val_split%d_fold%d.csv" % (k, i)
            train_path = dataDir + "MSTLoc/MSTLoc_train_split%d_fold%d.csv" % (k, i)
            val_path = dataDir + "MSTLoc/MSTLoc_val_split%d_fold%d.csv" % (k, i)
            df.loc[len(df)] = [k, i, 'Train'] + dataLabelAnalysis(train_path)
            df.loc[len(df)] = [k, i, 'Val'] + dataLabelAnalysis(val_path)

    print(df)
    df.to_csv(dataDir + "MSTLoc/MSTLoc_analysis.csv", index=None, mode='w')


if __name__ == '__main__':
    # prepareMSTLocData()
    # getAnnotation()
    # analysisData()
    # labelAnalysis()
    # checkUrl()
    # splitData()
    # gen_adj()

    # splitData1()

    # dataLabelAnalysis(newMSTLocDataPath)
    # dataLabelAnalysis(MSTLocTrainDataPath)
    # dataLabelAnalysis(MSTLocTestDataPath)

    dataAnalysis()
