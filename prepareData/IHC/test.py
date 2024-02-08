import pandas as pd
import os
import csv

dataDir = "data/"
file = "pathologyUrl.csv"

def readData(fileName):
    data = pd.read_csv(fileName, header=0)
    data = data[['Protein Id', 'Antibody Id', 'Tissue', 'Organ', 'URL']]
    return data

if __name__ == '__main__':
    # badImage = "D:\\VSCode\\ProteinLocalization\data\\bad.jpg"
    # sz = os.path.getsize(badImage)

    # allData = readData(dataDir + file)

    # print(len(allData))
    # print(allData.iloc[5500000])


    # data_file = pd.read_csv("D:/VSCode/ProteinLocalization/data/data_A.csv", header=0)
    # data_file = data_file[['Protein Id', 'Antibody Id', 'Tissue', 'Organ', 'URL']]
    # data_file['URL'] = data_file['URL'].str.rsplit('/', n=1, expand=True)[1]
    # print(data_file)
    # for item in data_file.itertuples(index=False):
    #     proteinId, antibodyId, tissue, organ, url = item
    #     print(item)
    #     print(proteinId, antibodyId, tissue, organ, url)

    # locations = ['actin filaments', 'aggresome', 'cell junctions', 'centriolar satellite', 'centrosome', 'cleavage furrow', 'cytokinetic bridge',
    #     'cytoplasmic bodies', 'cytosol', 'endoplasmic reticulum', 'endosomes', 'focal adhesion sites', 'golgi apparatus', 'intermediate filaments', 'kinetochore',
    #     'lipid droplets', 'lysosomes', 'microtubule ends', 'microtubules', 'midbody', 'midbody ring', 'mitochondria', 'mitotic chromosome', 'mitotic spindle',
    #     'nuclear bodies', 'nuclear membrane', 'nuclear speckles', 'nucleoli', 'nucleoli fibrillar center', 'nucleoli rim', 'nucleoplasm', 'peroxisomes',
    #     'plasma membrane', 'rods & rings', 'vesicles']
    # annotations = dict(zip(locations, range(len(locations))))
    # print(annotations)

    # a = []
    # # locations = locations.split(';')
    # locs = "plasma membrane;cytosol"
    # for l in locs.split(';'):
    #     a.append(annotations[l])
    # print(a)
    # a.sort()
    # print(a)

    # labeledData = pd.read_csv("data/data_val.csv", header=0)
    # locations = ['actin filaments', 'centrosome',
    # 'cytosol', 'endoplasmic reticulum', 'golgi apparatus', 'intermediate filaments',
    # 'microtubules', 'mitochondria', 'nuclear membrane', 'nucleoli', 'nucleoplasm',
    # 'plasma membrane', 'vesicles']
    # locations = [i + '_pred' for i in locations]
    # print(locations)
    # labeledData = pd.concat([labeledData, pd.DataFrame(columns=locations)])
    # print(labeledData)
    # labeledData[locations] =

    predData = pd.read_csv("results/preds/42_data_val.csv", header=0, index_col=0)
    locations = ['actin filaments', 'centrosome',
        'cytosol', 'endoplasmic reticulum', 'golgi apparatus', 'intermediate filaments',
        'microtubules', 'mitochondria', 'nuclear membrane', 'nucleoli', 'nucleoplasm',
        'plasma membrane', 'vesicles']
    locations_pred = [i + '_pred' for i in locations]
    locations_pred_labels = [i + '_pred_labels' for i in locations]
    preds = predData[locations_pred].values
    pred_labels = predData[locations_pred_labels].values
    print(preds)
    print(pred_labels)

    from mvit.models.criterion import t_criterion
    all_pred_labels = t_criterion(preds, 0.5)
    print(pred_labels)
    print(all_pred_labels)

