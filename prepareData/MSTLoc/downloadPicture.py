from multiprocessing import Pool, Lock
import pandas as pd
import csv
import time
import random

import aiohttp
import asyncio
import os


dataDir = "data/MSTLoc/"
imageDir = "G:/data/v20/"


trainFile = "train.csv"
testFile = "test.csv"
files = [trainFile, testFile]

badUrl = "bad_url.csv"
realUrl = "real_url.csv"
differentUrl = "different_url.csv"

def readData(fileName_list):
    data = []
    for fileName in fileName_list:
        df = pd.read_csv(fileName, header=0)
        print(df)
        data.append(df)
    data = pd.concat(data)
    print(data)
    data = data.values.tolist()
    return data


def readFromUrl(fileName):
    data = pd.read_csv(fileName, header=None)
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
ip = ['http://47.116.78.190:7890', 'http://47.98.134.232:7777', 'http://223.247.47.2:8089', 'http://183.166.149.201:41122',
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
# ip = ['http://120.26.37.240:8000', 'http://114.55.84.12:30001', 'http://112.86.154.242:3128', 'http://218.75.38.154:9091',
#         'http://183.129.190.172:9091', 'http://58.246.58.150:9002', 'http://106.14.255.124:80', 'http://112.124.56.162:8080',
#         'http://183.249.7.226:9091', 'http://210.5.10.87:53281', 'http://114.67.104.36:18888', 'http://223.68.190.136:9091',
#         'http://221.223.25.67:9000', 'http://60.209.97.182:9999', 'http://112.2.34.99:9091', 'http://120.197.179.166:8080',
#         'http://47.100.137.173:666', 'http://222.92.207.98:40080', 'http://113.57.84.39:9091', 'http://118.212.152.82:9091',
#         'http://114.255.132.60:3128', 'http://221.5.80.66:3128', 'http://39.175.92.35:30001', 'http://218.7.171.91:3128',
#         'http://139.155.48.55:1089', 'http://117.160.250.135:8081', 'http://183.247.199.120:30001', 'http://120.194.55.139:6969',
#         'http://120.46.152.244:8888', 'http://117.160.250.138:81', 'http://47.101.181.105:3128', 'http://124.71.186.187:3128',
#         'http://39.108.101.55:1080', 'http://183.33.192.7:9797', 'http://182.90.224.115:3128', 'http://58.34.41.219:8060',
#         'http://39.175.82.253:30001', 'http://115.29.170.58:8118', 'http://183.247.221.119:30001', 'http://183.233.169.226:9091',
#         'http://125.65.40.199:12345', 'http://112.6.117.178:8085', 'http://112.5.56.2:9091', 'http://223.84.240.36:9091',
#         'http://47.106.105.236:443', 'http://117.160.250.163:8080', 'http://117.160.250.137:9999', 'http://223.96.90.216:8085',
#         'http://218.64.84.117:8060', 'http://112.14.40.137:9091', 'http://112.49.34.128:9091', 'http://222.175.22.197:9091',
#         'http://39.175.90.51:30001', 'http://120.237.144.57:9091', 'http://183.237.45.34:9091']


async def fetch(session, url):
    # for _ in range(random.randint(10, 15)):
    for i in range(150):
        await asyncio.sleep(0.02 + random.random() * 10)
        # await asyncio.sleep(0.02 + random.random())
        # await asyncio.sleep(0.5)
        try:
            response = await session.get(url, timeout=aiohttp.ClientTimeout(total=15), proxy=random.choice(ip), ssl=False)
            # response = await session.get(url)

            lock.acquire()
            with open(dataDir + realUrl, "a", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([url, response.url])
            # if response.url.startswith("https://images.proteinatlas.org"):
            #     with open(dataDir + realUrl, "a", encoding="utf-8", newline="") as f:
            #         writer = csv.writer(f)
            #         writer.writerow([url, response.url])
            # else:
            #     with open(dataDir + differentUrl, "a", encoding="utf-8", newline="") as f:
            #         writer = csv.writer(f)
            #         writer.writerow([url, response.url])
            lock.release()
            url = response.url
            print("url:", url)
            if response.status == 200:
                print(response.status)
                return await response.read()
            return

            # if response.status == 200:
            #     return await response.read()
        except:
            continue


async def saveImage(session, data):
    T1 = time.time()

    url, proteinId, label = data[0], data[1], data[2]
    url = url.rsplit("/", 2)
    url[0] = "https://v20.proteinatlas.org/images"
    url = "/".join(url)

    root = imageDir + proteinId + "/"
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

    data = readData([dataDir + f for f in files])

    N = 30
    step = int(len(data) / N) + 1
    data = [data[i: i + step] for i in range(0, len(data), step)]
    P = Pool(processes = N, initializer=init_lock, initargs=(l, ))
    P.map(func=start, iterable=data)
    P.close()

    T2 = time.time()
    print("All Done!    运行时间:%s秒" % (T2 - T1))