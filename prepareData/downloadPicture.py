from multiprocessing import Pool, Lock
import csv
import time

import requests
import os

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
    root = "data/image/" + proteinId + "/" + organ + "/" + antibodyId + "/"
    # root = "data/img/"
    path = root + url.split('/')[-1]

    requests.adapters.DEFAULT_RETRIES = 3
    s = requests.session()
    s.keep_alive = False

    cnt = 0
    while True:
        try:
            r = s.get(url, timeout=2)
            # r = s.get(url, timeout=(10, 15))
            # r.raise_for_status()
            if r.status_code != 200:
                raise Exception
        # except requests.RequestException as e:
        except:
            cnt += 1
            time.sleep(0.5)

            if cnt > 50:
                lock.acquire()
                with open("data/bad_url.csv", "a", encoding="utf-8", newline="") as f:
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
    # P = Pool(processes = cpu_count())
    P = Pool(processes = 20, initializer=init_lock, initargs=(l, ))
    T2 = time.time()
    print("进程池创建!    运行时间:%s秒" % (T2 - T1))
    # data = readData("data/location.csv")
    data = readFromUrl("data/url_10.csv")
    # data = readFromBadUrl("data/undownload_9.csv")
    # print(len(data))

    P.map(func=saveImage, iterable=data)
    P.close()
    # P.join()
    T2 = time.time()
    print("All Done!    运行时间:%s秒" % (T2 - T1))
