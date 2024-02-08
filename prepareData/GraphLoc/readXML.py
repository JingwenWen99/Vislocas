from __future__ import annotations
from distutils.spawn import spawn
from lxml import etree
import csv
import time


dataDir = "data/v18/"
# imageDir = "E:/data/IHC/"

xmlPath = dataDir + "proteinatlas_v18.xml"
expressionLevelPath = dataDir + "IHC_expressionLevel.csv"
tissuePath = dataDir + "tissueUrl.csv"
pathologyPath = dataDir + "pathologyUrl.csv"
annotationsPath = dataDir + "annotations.csv"

def fast_iter(context, func):
    """
    从原始xml文件中提取数据，并保存入文件
    """
    T1 = time.time()
    T2 = T1
    cnt = 0
    level_cnt = 0
    tissue_cnt = 0
    pathology_cnt = 0
    annotations_cnt = 0

    f1 = open(expressionLevelPath, "w", encoding="utf-8", newline="")
    writer1 = csv.writer(f1)
    levelHeader = ['Protein Name', 'Protein Id', 'Reliability Verification',
        'Adipose tissue', 'Adrenal gland', 'Appendix', 'Bone marrow', 'Breast', 'Bronchus', 'Caudate', 'Cerebellum', 'Cerebral cortex', 'Cervix', 'Colon',
        'Duodenum', 'Endometrium 1', 'Endometrium 2', 'Epididymis', 'Esophagus', 'Fallopian tube', 'Gallbladder', 'Heart muscle', 'Hippocampus', 'Kidney', 'Liver',
        'Lung', 'Lymph node', 'Nasopharynx', 'Oral mucosa', 'Ovary', 'Pancreas', 'Parathyroid gland', 'Placenta', 'Prostate', 'Rectum', 'Salivary gland',
        'Seminal vesicle', 'Skeletal muscle', 'Skin 1', 'Skin 2', 'Small intestine', 'Smooth muscle', 'Soft tissue 1', 'Soft tissue 2', 'Spleen', 'Stomach 1',
        'Stomach 2', 'Testis', 'Thyroid gland', 'Tonsil', 'Urinary bladder', 'Vagina']
    writer1.writerow(levelHeader)

    f2 = open(tissuePath, "w", encoding="utf-8", newline="")
    writer2 = csv.writer(f2)
    # tissueHeader = ['Protein Name', 'Protein Id', 'Antibody Id', 'Reliability Verification',
    #     'Tissue', 'Organ', 'Cell Type', 'Staining Level', 'Intensity Level', 'Quantity', 'Location',
    #     'Sex', 'Age', 'Patient Id', 'SnomedParameters', 'URL']
    tissueHeader = ['Protein Name', 'Protein Id', 'Antibody Id',
        'Tissue', 'Organ', 'Cell Type', 'Staining Level', 'Intensity Level', 'Quantity', 'Location',
        'Sex', 'Age', 'Patient Id', 'SnomedParameters', 'URL']
    writer2.writerow(tissueHeader)

    f3 = open(pathologyPath, "w", encoding="utf-8", newline="")
    writer3 = csv.writer(f3)
    pathologyHeader = ['Protein Name', 'Protein Id', 'Antibody Id', 'Tissue', 'Organ', 'Cell Type',
        'Sex', 'Age', 'Patient Id', 'Staining Level', 'Intensity Level', 'Quantity', 'Location', 'SnomedParameters', 'URL']
    writer3.writerow(pathologyHeader)

    f4 = open(annotationsPath, "w", encoding="utf-8", newline="")
    writer4 = csv.writer(f4)
    annotationHeader = ['proteinName', 'proteinId', 'antibodyId', 'verification', 'organ', 'cellLine', 'locations']
    writer4.writerow(annotationHeader)

    for event, elem in context:
        levelResult, tissueUrl, pathologyUrl, annotation = func(elem)
        writer1.writerows(levelResult)
        writer2.writerows(tissueUrl)
        writer3.writerows(pathologyUrl)
        writer4.writerows(annotation)

        level_cnt += len(levelResult)
        tissue_cnt += len(tissueUrl)
        pathology_cnt += len(pathologyUrl)
        annotations_cnt += len(annotation)

        cnt += 1
        if cnt % 10 == 0:
            T2 = time.time()
            print("Done:    ", cnt, "   运行时间:%s秒" % (T2 - T1))

        elem.clear()

    f1.close()
    f2.close()
    f3.close()
    f4.close()

    T3 = time.time()
    print("All Done:    ", cnt, "   程序运行时间:%s秒" % (T3 - T1))
    print("Expression Level数据量：", level_cnt)
    print("Tissue图片数量：", tissue_cnt)
    print("Pathology图片数量：", pathology_cnt)
    print("Annotation数量：", annotations_cnt)
    del context

def process_element(elem):
    """
    处理element，根据xml文件内格式提取数据
    :params elem: Element
    """
    T1 = time.time()
    levelResult = []
    tissueUrl = []
    pathologyUrl = []
    annotation = []
    tissueList = ['Adipose tissue', 'Adrenal gland', 'Appendix', 'Bone marrow', 'Breast', 'Bronchus', 'Caudate', 'Cerebellum', 'Cerebral cortex', 'Cervix', 'Colon',
        'Duodenum', 'Endometrium 1', 'Endometrium 2', 'Epididymis', 'Esophagus', 'Fallopian tube', 'Gallbladder', 'Heart muscle', 'Hippocampus', 'Kidney', 'Liver',
        'Lung', 'Lymph node', 'Nasopharynx', 'Oral mucosa', 'Ovary', 'Pancreas', 'Parathyroid gland', 'Placenta', 'Prostate', 'Rectum', 'Salivary gland',
        'Seminal vesicle', 'Skeletal muscle', 'Skin 1', 'Skin 2', 'Small intestine', 'Smooth muscle', 'Soft tissue 1', 'Soft tissue 2', 'Spleen', 'Stomach 1',
        'Stomach 2', 'Testis', 'Thyroid gland', 'Tonsil', 'Urinary bladder', 'Vagina']

    proteinName = elem.find("name").text
    proteinId = elem.find("identifier").get("id")

    tissueExpression = elem.find("tissueExpression")
    if tissueExpression:
        reliability_verification = tissueExpression.find("verification").text

        tissueDict = dict.fromkeys(tissueList)
        tissue = tissueExpression.findall("data/tissue")
        level = tissueExpression.findall("data/level")
        tissueDict.update(zip([t.text for t in tissue], [l.text for l in level]))
        levels = list(tissueDict.values())

        levelResult.append([proteinName, proteinId, reliability_verification] + levels)

    for antibody in elem.findall("antibody"):
        antibodyId = antibody.get("id")

        normalTissue = antibody.find("tissueExpression[@assayType='tissue']")
        if normalTissue:
            # reliability_verification = normalTissue.find("verification").text
            for data in normalTissue.findall("data"):
                tissue = data.find("tissue").text
                organ = data.find("tissue").get("organ")

                cellType = []
                stainingLevel = []
                intensityLevel = []
                quantity = []
                location = []
                for tissueCell in data.findall("tissueCell"):
                    cellType.append(tissueCell.find("cellType").text)
                    stainingLevel.append(tissueCell.find("level[@type='staining']").text if tissueCell.find("level[@type='staining']") is not None else "")
                    intensityLevel.append(tissueCell.find("level[@type='intensity']").text if tissueCell.find("level[@type='intensity']") is not None else "")
                    quantity.append(tissueCell.find("quantity").text if tissueCell.find("quantity") is not None else "")
                    location.append(tissueCell.find("location").text)

                cellType = ";".join(cellType)
                stainingLevel = ";".join(stainingLevel)
                intensityLevel = ";".join(intensityLevel)
                quantity = ";".join(quantity)
                location = ";".join(location)

                for patient in data.findall("patient"):
                    sex = patient.find("sex").text if patient.find("sex") is not None else None
                    age = patient.find("age").text if patient.find("age") is not None else None
                    patientId = patient.find("patientId").text
                    snomedParameters = ";".join([snomed.get("tissueDescription") for snomed in patient.findall(".//snomed")])
                    url = patient.find(".//imageUrl").text
                    # tissueUrl.append([proteinName, proteinId, antibodyId, reliability_verification,
                    #     tissue, organ, cellType, stainingLevel, intensityLevel, quantity, location,
                    #     sex, age, patientId, snomedParameters, url])
                    tissueUrl.append([proteinName, proteinId, antibodyId,
                        tissue, organ, cellType, stainingLevel, intensityLevel, quantity, location,
                        sex, age, patientId, snomedParameters, url])

        pathologyTissue = antibody.find("tissueExpression[@assayType='pathology']")
        if pathologyTissue:
            for data in pathologyTissue.findall("data"):
                tissue = data.find("tissue").text
                organ = data.find("tissue").get("organ")
                cellType = data.find("tissueCell/cellType").text

                for patient in data.findall("patient"):
                    sex = patient.find("sex").text if patient.find("sex") is not None else None
                    age = patient.find("age").text if patient.find("age") is not None else None
                    patientId = patient.find("patientId").text
                    stainingLevel = patient.find("level[@type='staining']").text
                    intensityLevel = patient.find("level[@type='intensity']").text
                    quantity = patient.find("quantity").text if patient.find("quantity") is not None else None
                    location = patient.find("location").text
                    for sample in patient.findall("sample"):
                        snomedParameters = ";".join([snomed.get("tissueDescription") for snomed in sample.findall(".//snomed")])
                        url = sample.find(".//imageUrl").text
                        pathologyUrl.append([proteinName, proteinId, antibodyId, tissue, organ, cellType,
                            sex, age, patientId, stainingLevel, intensityLevel, quantity, location,
                            snomedParameters, url])

        for subAssay in antibody.findall(".//subAssay[@type='human']"):
            verification = subAssay.find("verification").text
            for data in subAssay.iter(tag = "data"):
                organ = data.find("cellLine").get("organ")
                cellLine = data.find("cellLine").text
                locations = ";".join([location.text for location in data.findall("location")])
                annotation.append([proteinName, proteinId, antibodyId, verification, organ, cellLine, locations])

    T2 = time.time()
    print(proteinName, "运行时间:%s毫秒" % ((T2 - T1)*1000))
    return levelResult, tissueUrl, pathologyUrl, annotation


if __name__ == '__main__':
    context = etree.iterparse(xmlPath, tag='entry')
    fast_iter(context, process_element)


