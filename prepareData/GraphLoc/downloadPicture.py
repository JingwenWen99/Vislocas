from multiprocessing import Pool, Lock
import pandas as pd
import csv
import time
import random

import aiohttp
import asyncio
import os


dataDir = "data/GraphLoc/"
imageDir = "G:/data/v18/normal/"


# trainFile = dataDir + "train.csv"
# testFile = dataDir + "test.csv"
# files = [trainFile, testFile]
file = dataDir + "GraphLoc_all_data.csv"
undownloadFile = dataDir + "undownload.csv"

badUrl = dataDir + "bad_url.csv"
realUrl = dataDir + "real_url.csv"
differentUrl = dataDir + "different_url.csv"


def readData(fileName):
    data = pd.read_csv(fileName, header=0, index_col=0)
    print(data)
    # data = data[['Protein Id', 'Antibody Id', 'Tissue', 'Organ', 'URL']][6000000:].sample(frac=1).values.tolist()
    data = data[['Protein Id', 'Antibody Id', 'Tissue', 'URL']].sample(frac=1).values.tolist()
    return data


# def readData(fileName_list):
#     data = []
#     for fileName in fileName_list:
#         df = pd.read_csv(fileName, header=0)
#         print(df)
#         data.append(df)
#     data = pd.concat(data)
#     print(data)
#     data = data.values.tolist()
#     return data


def readFromUrl(fileName):
    data = pd.read_csv(fileName, header=None)
    print(data)
    data = data.values.tolist()
    return data


def init_lock(l):
	global lock
	lock = l


# ip = ['https://114.99.3.223:8089', 'https://47.96.16.128:3128', 'https://114.99.6.153:8089', 'https://114.103.89.91:8089',
#         'https://114.106.135.194:8089', 'https://111.224.217.253:8089', 'https://114.106.135.2:8089', 'https://114.106.135.222:8089',
#         'https://27.157.230.191:8089', 'https://114.106.170.231:8089', 'https://121.233.226.115:8089', 'https://114.231.8.112:8089',
#         'https://114.106.134.198:8089', 'https://114.231.106.220:8089', 'https://114.231.41.90:8089', 'https://183.165.247.28:8089',
#         'https://175.10.205.67:4780', 'https://120.26.121.156:3128', 'https://180.105.146.153:8089', 'https://47.116.78.190:7890',
#         'https://180.105.146.153:8089', 'https://47.116.78.190:7890', 'https://183.165.247.253:8089', 'https://114.231.41.92:8089',
#         'https://114.103.89.162:8089', 'https://117.70.48.40:8089', 'https://114.106.170.167:8089', 'https://49.88.158.253:8089',
#         'https://118.31.2.38:8999', 'https://114.106.171.31:8089', 'https://27.184.66.209:7890', 'https://183.165.244.123:8089',
#         'https://114.99.5.177:8089', 'https://111.225.153.181:8089', 'https://114.231.8.248:8089', 'https://123.182.58.51:8089',
#         'https://114.99.13.186:8089', 'https://111.225.152.68:8089', 'https://114.106.137.37:8089', 'https://114.231.45.46:8089']
ip = ['http://8.134.136.224:8080', 'http://183.27.249.123:8085', 'http://39.108.230.16:3128', 'http://61.158.175.38:9002',
        'http://120.194.4.155:5443', 'http://58.57.170.146:9002', 'http://8.219.74.58:8080', 'http://1.117.154.29:2080',
        'http://47.98.219.185:8998	', 'http://139.224.190.222:8083', 'http://117.160.250.133:8081', 'http://114.106.137.12:8089',
        'http://211.138.6.37:9091', 'http://222.64.52.179:7890', 'http://61.175.214.2:9091', 'http://121.226.188.136:8089',
        'http://211.140.132.150:8193', 'http://106.14.47.96:80', 'http://59.58.60.76:4780', 'http://111.21.183.58:9091',
        # 'http://47.116.78.190:7890', 'http://47.98.134.232:7777', 'http://223.247.47.2:8089', 'http://183.166.149.201:41122',

    'http://47.116.78.190:7890', 'http://47.98.134.232:7777', 'http://223.247.47.2:8089', 'http://183.166.149.201:41122',
        'http://139.224.56.162:1234', 'http://117.70.49.229:8089', 'http://122.241.118.233:41122', 'http://121.206.141.62:8089',
        'http://101.132.25.152:50001', 'http://114.231.45.53:8089', 'http://121.226.188.70:8089', 'http://114.231.45.150:8089',
        'http://180.105.117.78:8089', 'http://183.165.245.79:8089', 'http://114.231.82.74:8089', 'http://114.106.173.48:8089',
        'http://114.231.42.117:8089', 'http://49.235.114.158:3128', 'http://124.71.157.181:10443', 'http://114.231.42.117:8089',
        'http://49.235.114.158:3128', 'http://124.71.157.181:10443', 'http://139.129.231.228:8080', 'http://183.165.250.238:8089',
        'http://124.70.205.56:50001', 'http://47.98.219.185:8998', 'http://222.190.223.31:8089', 'http://139.196.151.191:20201',
        'http://117.69.191.92:41122', 'http://114.232.109.163:8089', 'http://114.99.13.16:8089', 'http://117.70.48.97:8089',
        'http://114.232.110.171:8089', 'http://47.96.70.163:8888', 'http://114.231.46.71:8089', 'http://112.17.173.55:9091',
        'http://117.114.149.66:55443', 'http://61.216.156.222:60808', 'http://112.14.47.6:52024', 'http://121.13.252.61:41564',
        'http://117.41.38.19:9000', 'http://222.74.73.202:42055', 'http://202.109.157.65:9000', 'http://121.13.252.62:41564',
        'http://61.164.39.68:53281', 'http://27.42.168.46:55481', 'http://121.13.252.58:41564', 'http://61.216.185.88:60808',
        'http://183.236.232.160:8080', 'http://116.9.163.205:58080', 'http://110.87.202.66:8089', 'http://123.182.59.120:8089',
        'http://101.6.70.93:4780', 'http://114.99.3.44:8089', 'http://111.225.152.152:8089', 'http://111.225.152.4:8089',
        'http://114.104.135.29:41122', 'http://114.99.9.14:8089', 'http://114.103.88.95:8089', 'http://114.99.11.122:8089',
        'http://175.10.205.67:4780', 'http://114.255.132.60:3128', 'http://163.177.106.4:8001', 'http://123.182.59.244:8089',
        'http://111.225.153.170:8089', 'http://59.59.212.24:8089', 'http://27.184.66.209:7890', 'http://123.182.58.56:8089',
        'http://114.99.2.201:8089', 'http://114.99.2.35:8089', 'http://101.6.70.93:4780', 'http://111.224.11.94:8089',
        'http://111.225.153.210:8089', 'http://111.225.152.168:8089', 'http://27.157.228.153:8089', 'http://183.165.246.127:8089',
        'http://166.111.88.44:4780', 'http://114.255.132.60:3128', 'http://49.88.168.101:8089', 'http://163.177.106.4:8001',
        'http://219.131.240.75:9797', 'http://111.225.153.170:8089', 'http://114.102.47.237:8089', 'http://114.102.45.108:8089',
        'http://114.102.47.64:8089', 'http://114.102.45.96:8089', 'http://123.171.1.238:8089', 'http://123.171.42.233:8089',
        'http://114.102.47.153:8089', 'http://114.102.46.101:8089']


async def fetch(session, url):
    # for _ in range(random.randint(10, 15)):
    for i in range(150):
        await asyncio.sleep(0.02 + random.random() * 5)
        # await asyncio.sleep(0.02 + random.random())
        # await asyncio.sleep(0.5)
        try:
            # print(url)
            response = await session.get(url, timeout=aiohttp.ClientTimeout(total=15), proxy=random.choice(ip), ssl=False)
            # response = await session.get(url)

            if response.status == 200:
                return await response.read()
        except:
            continue


async def saveImage(session, data):
    T1 = time.time()

    # url, proteinId, label = data[0], data[1], data[2]
    proteinId, antibodyId, tissue, url = data[0], data[1], data[2], data[3]
    # url = url.rsplit("/", 2)
    # url[0] = "https://v18.proteinatlas.org/images"
    # url = "/".join(url)

    # root = imageDir + proteinId + "/"
    # path = root + url.split('/')[-1]

    root = imageDir + proteinId + "/" + tissue + "/" + antibodyId + "/"
    path = root + url.split('/')[-1]

    # await asyncio.sleep(random.random() * 100)
    r = await fetch(session, url)
    if r is None:
        lock.acquire()
        with open(dataDir + badUrl, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(data)
            # print(url + "保存失败！")
        lock.release()
    else:
        lock.acquire()
        if not os.path.exists(root):
            os.makedirs(root)
        lock.release()
        with open(path, "wb") as f:
        # with open(url.split('/')[-1], "wb") as f:
            f.write(r)

        badImage1 = "D:\\VSCode\\ProteinLocalization\data\\bad.jpg"
        sz1 = os.path.getsize(badImage1)
        badImage2 = "D:\\VSCode\\ProteinLocalization\data\\MSTLoc_bad.jpg"
        sz2 = os.path.getsize(badImage2)

        if os.path.getsize(path) == sz1 or os.path.getsize(path) == sz2:
            lock.acquire()
            with open(dataDir + badUrl, "a", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(data)
                print(url + "图像错误！")
            lock.release()
        else:
            print(path + "保存成功！")

    T2 = time.time()
    # print("运行时间:%s秒" % (T2 - T1))


async def task(data):
    # timeout = aiohttp.ClientTimeout(total=60)
    # conn = aiohttp.TCPConnector(ssl=False, limit=5, limit_per_host=30)
    conn = aiohttp.TCPConnector(ssl=False, limit=20)
    async with aiohttp.ClientSession(connector=conn) as session:
        tasks = [asyncio.create_task(saveImage(session, item)) for item in data]
        await asyncio.wait(tasks)
    await asyncio.sleep(1)


def start(data):
    asyncio.run(task(data))


if __name__ == '__main__':
    T1 = time.time()
    l = Lock()

    # data = readData(file)
    data = readFromUrl(undownloadFile)

    N = 1
    step = int(len(data) / N) + 1
    data = [data[i: i + step] for i in range(0, len(data), step)]
    P = Pool(processes = N, initializer=init_lock, initargs=(l, ))
    P.map(func=start, iterable=data)
    P.close()

    T2 = time.time()
    print("All Done!    运行时间:%s秒" % (T2 - T1))