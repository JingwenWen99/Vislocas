from multiprocessing import Pool, Lock
import pandas as pd
import csv
import time
import random

import aiohttp
import asyncio
import os


dataDir = "data/"
normalImageDir = "dataset/IHC/normal/"
pathologyImageDir = "dataset/IHC/pathology/"
imageDir = normalImageDir
# imageDir = pathologyImageDir

normalFile = "tissueUrl.csv"
pathologyFile = "pathologyUrl.csv"
file = normalFile
# file = pathologyFile

normalBadUrl = "normal_bad_url.csv"
pathologyBadUrl = "pathology_bad_url.csv"
badUrl = normalBadUrl
# badUrl = pathologyBadUrl

normalUndownloadFile = "normal_undownload.csv"
pathologyUndownloadFile = "pathology_undownload.csv"
undownloadFile = normalUndownloadFile


def readData(fileName):
    data = pd.read_csv(fileName, header=0)
    data = data[['Protein Id', 'Antibody Id', 'Tissue', 'Organ', 'URL']].sample(frac=1).values.tolist()
    return data


def readFromUrl(fileName):
    data = pd.read_csv(fileName, header=None)
    data = data.values.tolist()
    return data


def init_lock(l):
	global lock
	lock = l


ip = ['http://120.26.37.240:8000', 'http://114.55.84.12:30001', 'http://112.86.154.242:3128', 'http://218.75.38.154:9091',
        'http://183.129.190.172:9091', 'http://58.246.58.150:9002', 'http://106.14.255.124:80', 'http://112.124.56.162:8080',
        'http://183.249.7.226:9091', 'http://210.5.10.87:53281', 'http://114.67.104.36:18888', 'http://223.68.190.136:9091',
        'http://221.223.25.67:9000', 'http://60.209.97.182:9999', 'http://112.2.34.99:9091', 'http://120.197.179.166:8080',
        'http://47.100.137.173:666', 'http://222.92.207.98:40080', 'http://113.57.84.39:9091', 'http://118.212.152.82:9091',
        'http://114.255.132.60:3128', 'http://221.5.80.66:3128', 'http://39.175.92.35:30001', 'http://218.7.171.91:3128',
        'http://139.155.48.55:1089', 'http://117.160.250.135:8081', 'http://183.247.199.120:30001', 'http://120.194.55.139:6969',
        'http://120.46.152.244:8888', 'http://117.160.250.138:81', 'http://47.101.181.105:3128', 'http://124.71.186.187:3128',
        'http://39.108.101.55:1080', 'http://183.33.192.7:9797', 'http://182.90.224.115:3128', 'http://58.34.41.219:8060',
        'http://39.175.82.253:30001', 'http://115.29.170.58:8118', 'http://183.247.221.119:30001', 'http://183.233.169.226:9091',
        'http://125.65.40.199:12345', 'http://112.6.117.178:8085', 'http://112.5.56.2:9091', 'http://223.84.240.36:9091',
        'http://47.106.105.236:443', 'http://117.160.250.163:8080', 'http://117.160.250.137:9999', 'http://223.96.90.216:8085',
        'http://218.64.84.117:8060', 'http://112.14.40.137:9091', 'http://112.49.34.128:9091', 'http://222.175.22.197:9091',
        'http://39.175.90.51:30001', 'http://120.237.144.57:9091', 'http://183.237.45.34:9091']


async def fetch(session,url):
    for _ in range(random.randint(10, 15)):
        await asyncio.sleep(0.02 + random.random() * _)
        try:
            response = await session.get(url, timeout=aiohttp.ClientTimeout(total=10), proxy=random.choice(ip), ssl=False)
            if response.status == 200:
                return await response.read()
        except:
            continue


async def saveImage(session, data):
    T1 = time.time()

    proteinId, antibodyId, tissue, organ, url = data[0], data[1], data[2], data[3], data[4]

    root = imageDir + proteinId + "/" + organ + "/" + tissue + "/" + antibodyId + "/"
    path = root + url.split('/')[-1]

    r = await fetch(session, url)
    if r is None:
        lock.acquire()
        with open(dataDir + badUrl, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([proteinId, antibodyId, tissue, organ, url])
            print(url + "保存失败！")
        lock.release()
    else:
        lock.acquire()
        if not os.path.exists(root):
            os.makedirs(root)
        lock.release()
        with open(path, "wb") as f:
        # with open(url.split('/')[-1], "wb") as f:
            f.write(r)
            print(path + "保存成功！")

    T2 = time.time()
    print("运行时间:%s秒" % (T2 - T1))


async def task(data):
    conn = aiohttp.TCPConnector(ssl=False, limit=1000)
    async with aiohttp.ClientSession(connector=conn) as session:
        tasks = [asyncio.create_task(saveImage(session, item)) for item in data]
        await asyncio.wait(tasks)
    await asyncio.sleep(1)


def start(data):
    asyncio.run(task(data))


if __name__ == '__main__':
    T1 = time.time()
    l = Lock()

    data = readData(dataDir + file)

    N = 1
    step = 200
    # step = int(len(data) / N) + 1
    data = [data[i: i + step] for i in range(0, len(data), step)]
    P = Pool(processes = N, initializer=init_lock, initargs=(l, ))
    P.map(func=start, iterable=data)
    P.close()

    T2 = time.time()
    print("All Done!    运行时间:%s秒" % (T2 - T1))