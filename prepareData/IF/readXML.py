from lxml import etree
import csv
import time

import requests
import os

dataDir = "data/"
imageDir = "E:/data/IF/"

def fast_iter(context, func):
    """
    从原始xml文件中提取数据（ProteinName, ProteinId, AntibodyId, Organ, Locations, URLs, CellLine）并保存入location文件
    """
    T1 = time.time()
    T2 = T1
    cnt = 0
    with open(dataDir + "location.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        header = ["ProteinName", "ProteinId", "AntibodyId", "Organ", "Locations", "URLs", "CellLine"]
        writer.writerow(header)

        for event, elem in context:
            result = func(elem)
            writer.writerows(result)
            cnt += 1
            if cnt % 10 == 0:
                T2 = time.time()
                print("Done:    ", cnt, "   运行时间:%s秒" % (T2 - T1))

            elem.clear()
    T3 = time.time()
    print("All Done:    ", cnt, "   程序运行时间:%s秒" % (T3 - T1))
    del context

def process_element(elem):
    """
    处理element，根据xml文件内格式提取数据：ProteinName, ProteinId, AntibodyId, Organ, Locations, URLs, CellLine
    :params elem: Element
    """
    # T1 = time.time()
    result = []

    proteinName = elem.findall("name")[0].text
    proteinId = elem.findall("identifier")[0].get("id")

    for antibody in elem.findall("antibody"):
        antibodyId = antibody.get("id")
        for subAssay in antibody.findall(".//subAssay[@type='human']"):
            for data in subAssay.iter(tag = "data"):
                organ = data.findall("cellLine")[0].get("organ")
                cellLine = data.findall("cellLine")[0].text

                locations = ""
                for location in data.findall("location"):
                    locations = locations + location.text + ";"
                locations = locations[:-1]

                urls = ""
                for url in data.findall(".//imageUrl"):
                    # saveImage(proteinId, antibodyId, organ, url.text)
                    urls = urls + url.text + ";"
                urls = urls[:-1]

                result.append([proteinName, proteinId, antibodyId, organ, locations, urls, cellLine])
    # T2 = time.time()
    # print(proteinName, "运行时间:%s毫秒" % ((T2 - T1)*1000))
    return result

def saveImage(proteinId, antibodyId, organ, url):
    root = imageDir + proteinId + "/" + organ + "/" + antibodyId + "/"
    path = root + url.split('/')[-1]
    try:
        r = requests.get(url, timeout=20)
        if not os.path.exists(root):
            os.makedirs(root)
        with open(path, "wb") as f:
            f.write(r.content)
            print(path + "保存成功！")
    except:
        with open(dataDir + "bad_url.csv", "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([url])
            print(url + "保存失败！")


if __name__ == '__main__':
    xml_name = dataDir + "proteinatlas.xml"
    context = etree.iterparse(xml_name, tag='entry')
    fast_iter(context, process_element)


