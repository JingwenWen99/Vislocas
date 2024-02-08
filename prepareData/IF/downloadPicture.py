from multiprocessing import Pool, Lock
import csv
import time

import requests
import os

dataDir = "data/"
imageDir = "D:/data/image/"

def readData(fileName):
    data = []
    with open(fileName, "r") as f:
        reader = csv.reader(f, delimiter=",")
        next(reader)
        for line in reader:
            urls = line[5].split(';')
            for url in urls:
                data.append([line[1], line[2], line[3], url])
    return data

def readFromUrl(fileName):
    data = []
    with open(fileName, "r") as f:
        reader = csv.reader(f, delimiter=",")
        for line in reader:
            data.append(line)
    return data

def saveImage(data):
    T1 = time.time()
    proteinId, antibodyId, organ, url = data[0], data[1], data[2], data[3]
    root = imageDir + proteinId + "/" + organ + "/" + antibodyId + "/"
    path = root + url.split('/')[-1]

    requests.adapters.DEFAULT_RETRIES = 3
    s = requests.session()
    s.keep_alive = False

    cnt = 0
    while True:
        try:
            r = s.get(url, timeout=2)
            if r.status_code != 200:
                raise Exception
        except:
            cnt += 1
            time.sleep(0.5)

            if cnt > 50:
                lock.acquire()
                with open(dataDir + "bad_url.csv", "a", encoding="utf-8", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([proteinId, antibodyId, organ, url])
                    print(url + "保存失败！")
                lock.release()
                break
            continue
        else:
            lock.acquire()
            if not os.path.exists(root):
                os.makedirs(root)
            lock.release()
            with open(path, "wb") as f:
                f.write(r.content)
                # print(path + "保存成功！")
            break
    T2 = time.time()
    print("运行时间:%s秒" % (T2 - T1))

def init_lock(l):
	global lock
	lock = l


if __name__ == '__main__':
    T1 = time.time()
    l = Lock()
    P = Pool(processes = 40, initializer=init_lock, initargs=(l, )) # processes参数可调
    T2 = time.time()
    print("进程池创建!    运行时间:%s秒" % (T2 - T1))
    # data = readData(dataDir + "location.csv") # 下载从xml中提取出的图像数据，此处url为HPA网站默认提供的图像url，为三通道混合图像（绿蓝红）
    data = readFromUrl(dataDir + "url.csv") # 下载的图像包含4通道各自的图像，以及4通道混合图像。若下载图像不完全，后续可修改此处文件名为运行checkImage文件得到的"undownload.csv"

    P.map(func=saveImage, iterable=data)
    P.close()
    T2 = time.time()
    print("All Done!    运行时间:%s秒" % (T2 - T1))
