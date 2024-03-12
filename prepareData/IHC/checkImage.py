import pandas as pd
import os
import csv

dataDir = "data/"
normalImageDir = "dataset/IHC/normal"
pathologyImageDir = "dataset/IHC/pathology"
# imageDir = normalImageDir
imageDir = pathologyImageDir

normalFile = "tissueUrl.csv"
pathologyFile = "pathologyUrl.csv"
# file = normalFile
file = pathologyFile

normalUndownloadFile = "normal_undownload.csv"
pathologyUndownloadFile = "pathology_undownload.csv"
# undownloadFile = normalUndownloadFile
undownloadFile = pathologyUndownloadFile

# outFile = "normal_undownload0.csv"
outFile = "pathology_undownload0.csv"


def readData(fileName):
    data = pd.read_csv(fileName, header=0)
    # data = data[['Protein Id', 'Antibody Id', 'Tissue', 'Organ', 'URL']][:5500000].values.tolist()
    data = data[['Protein Id', 'Antibody Id', 'Tissue', 'Organ', 'URL']][5500000:6000000].values.tolist()
    return data


def readFromUrl(fileName):
    data = pd.read_csv(fileName, header=None)
    data = data.values.tolist()
    return data


if __name__ == '__main__':
    badImage = "data/bad.jpg"
    sz = os.path.getsize(badImage)

    allData = readData(dataDir + file)

    undownload = 0
    cnt = 0

    f = open(dataDir + outFile, "w", encoding="utf-8", newline="")
    writer = csv.writer(f)

    for item in allData:

        path = '/'.join([imageDir, item[0], item[3], item[2], item[1], item[4].split('/')[-1]])

        if (not os.path.exists(path)) or (os.path.getsize(path) == sz):
            writer.writerow(item)
            undownload += 1
        cnt += 1
        if cnt % 1000 == 0:
            print("DONE:    ", cnt)

    f.close()

    print("ALL DATA: ", len(allData))
    print("UNDOWNLOAD DATA: ", undownload)

