from operator import le
import os
import csv
from re import A, L, S

def readData(fileName):
    data = []
    with open(fileName, "r") as f:
        reader = csv.reader(f, delimiter=",")
        for line in reader:
            data.append(line)
    return data

def readAllData(fileName):
    data = []
    with open(fileName, "r") as f:
        reader = csv.reader(f, delimiter=",")
        next(reader)
        for line in reader:
            urls = line[5].split(';')
            for url in urls:
                data.append([line[1], line[2], line[3], url])
    return data

if __name__ == '__main__':
    badImage = "D:\\VSCode\\ProteinLocalization\data\\bad.jpg"
    sz = os.path.getsize(badImage)

    allData = readAllData("data/location.csv")
    downloadImage = readData("data/downloadImage_3.csv")
    undownload = readData("data/undownload_3.csv")
    badUrl = readData("data/bad_url_1.csv")

    cnt = 0
    newAllData = []
    for data in allData:
        if data not in newAllData:
            newAllData.append(data)
        else:
            print(data)
        cnt += 1
        if cnt % 1000 == 0:
            print(cnt)
    print(len(allData), "    ", len(newAllData))

    cnt = 0
    newDownloadImage = []
    for data in downloadImage:
        if data not in newDownloadImage:
            newDownloadImage.append(data)
        cnt += 1
        if cnt % 1000 == 0:
            print(cnt)
    print(len(downloadImage), "    ", len(newDownloadImage))

    cnt = 0
    newUndownload = []
    for data in undownload:
        if data not in newUndownload:
            newUndownload.append(data)
        cnt += 1
        if cnt % 1000 == 0:
            print(cnt)
    print(len(undownload), "    ", len(newUndownload))

    cnt = 0
    newBadUrl = []
    for data in badUrl:
        if data not in newBadUrl:
            newBadUrl.append(data)
        cnt += 1
        if cnt % 1000 == 0:
            print(cnt)
    print(len(badUrl), "    ", len(newBadUrl))
    # print(len(set(allData)))
    # print(len(set(downloadImage)))
    # print(len(set(undownload)))
    # print(len(set(badUrl)))
    # error = []
    # cnt = 0
    # for item in badUrl:
    #     if item not in undownload:
    #         proteinId, antibodyId, organ, url = item[0], item[1], item[2], item[3]
    #         root = "data/image/" + proteinId + "/" + organ + "/" + antibodyId + "/" + url.split('/')[-1]
    #         if (not os.path.exists(root)) or os.path.getsize(root) == sz:
    #             print(111)
    #             error.append(item)
    #             with open("data/error.csv", "a", encoding="utf-8", newline="") as f:
    #                 writer = csv.writer(f)
    #                 writer.writerow(item)
    #     cnt += 1
    #     if cnt % 10000 == 0:
    #         print("DONE:    ", cnt)
    # print()

    # cnt = 0
    # for item in undownload:
    #     if item not in badUrl:
    #         proteinId, antibodyId, organ, url = item[0], item[1], item[2], item[3]
    #         root = "data/image/" + proteinId + "/" + organ + "/" + antibodyId + "/" + url.split('/')[-1]
    #         if (not os.path.exists(root)) or os.path.getsize(root) == sz:
    #             print(111)
    #             error.append(item)
    #             with open("data/error.csv", "a", encoding="utf-8", newline="") as f:
    #                 writer = csv.writer(f)
    #                 writer.writerow(item)
    #     cnt += 1
    #     if cnt % 10000 == 0:
    #         print("DONE:    ", cnt)
