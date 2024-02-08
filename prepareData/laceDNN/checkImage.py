import pandas as pd
import os
import csv

dataDir = "data/laceDNN/"
imageDir = "G:/data/v18/normal/"

file = dataDir + "laceDNN_url_data.csv"
undownloadFile = dataDir + "undownload.csv"

# # outFile = "normal_undownload0.csv"
# outFile = "pathology_undownload0.csv"


def readData(fileName):
    data = pd.read_csv(fileName, header=0, index_col=0)
    print(data)
    # data = data[['Protein Id', 'Antibody Id', 'Tissue', 'Organ', 'URL']][6000000:].sample(frac=1).values.tolist()
    data = data[['Protein Id', 'Antibody Id', 'Tissue', 'URL']].values.tolist()
    return data


def readFromUrl(fileName):
    data = pd.read_csv(fileName, header=None)
    data = data.values.tolist()
    return data


if __name__ == '__main__':
    badImage1 = "D:\\VSCode\\ProteinLocalization\data\\bad.jpg"
    sz1 = os.path.getsize(badImage1)
    badImage2 = "D:\\VSCode\\ProteinLocalization\data\\MSTLoc_bad.jpg"
    sz2 = os.path.getsize(badImage2)
    badImage3 = "D:\\VSCode\\ProteinLocalization\data\\bad2.jpg"
    sz3 = os.path.getsize(badImage3)
    badImage4 = "D:\\VSCode\\ProteinLocalization\data\\bad3.jpg"
    sz4 = os.path.getsize(badImage4)
    szList = [sz2, sz3, sz4]

    allData = readData(file)
    # allData = readFromUrl(dataDir + undownloadFile)
    # allData = allData

    undownload = 0
    cnt = 0

    f = open(undownloadFile, "w", encoding="utf-8", newline="")
    writer = csv.writer(f)

    for item in allData:

        path = '\\'.join([imageDir, item[0], item[2], item[1], item[3].split('/')[-1]])

        if (not os.path.exists(path)) or (os.path.getsize(path) == sz1) or (os.path.getsize(path) < 2 * max(szList)):
            writer.writerow(item)
            undownload += 1
        cnt += 1
        if cnt % 1000 == 0:
            print("DONE:    ", cnt)

    f.close()

    print("ALL DATA: ", len(allData))
    print("UNDOWNLOAD DATA: ", undownload)

