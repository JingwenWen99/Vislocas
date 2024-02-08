import pandas as pd


dataDir = "data/"
imageDir = "E:/data/IHC/"

# subcellularLocs = ['actin filaments', 'aggresome', 'cell junctions', 'centriolar satellite', 'centrosome', 'cleavage furrow', 'cytokinetic bridge',
#     'cytoplasmic bodies', 'cytosol', 'endoplasmic reticulum', 'endosomes', 'focal adhesion sites', 'golgi apparatus', 'intermediate filaments', 'kinetochore',
#     'lipid droplets', 'lysosomes', 'microtubule ends', 'microtubules', 'midbody', 'midbody ring', 'mitochondria', 'mitotic chromosome', 'mitotic spindle',
#     'nuclear bodies', 'nuclear membrane', 'nuclear speckles', 'nucleoli', 'nucleoli fibrillar center', 'nucleoli rim', 'nucleoplasm', 'peroxisomes',
#     'plasma membrane', 'rods & rings', 'vesicles'] # 共35种亚细胞位置

""" ['proteinName', 'proteinId', 'antibodyId', 'verification', 'organ', 'cellLine', 'locations'] """
def locationAnalysis(filePath1, savePath1, savePath2):
    locationData = pd.read_csv(filePath1, header=0)

    subcellularLocs = locationData['locations'].str.split(";", expand=True).stack().reset_index(drop=True).drop_duplicates().sort_values(ascending=True).tolist() # 亚细胞位置种类
    # print(len(subcellularLocs))
    # print(subcellularLocs)

    newLocationData = locationData[['proteinName', 'proteinId', 'antibodyId', 'verification', 'organ', 'locations']]
    newLocationData = newLocationData.reindex(columns=['proteinName', 'proteinId', 'antibodyId', 'verification', 'organ', 'locations', *subcellularLocs], fill_value=0)

    # 独热编码，每个亚细胞位置类用0、1表示
    location = locationData['locations'].str.split(";", expand=True).stack().rename('locations').reset_index(1, drop=True)
    for i, v in location.items():
        newLocationData.at[i, v] = 1
        print('index: ', i, ' location: ', v)

    print(newLocationData)
    newLocationData.to_csv(savePath1, index=None, mode='w')

    protein_count = locationData['proteinId'].value_counts(sort=False).rename('Protein Id count').rename_axis('Protein Id').reset_index()
    antibody_count = locationData['antibodyId'].value_counts(sort=False).rename('Antibody Id count').rename_axis('Antibody Id').reset_index()
    reliability_count = locationData['verification'].value_counts().rename('Reliability Verification count')
    reliability_count = pd.concat([reliability_count, locationData['verification'].value_counts(normalize=True).rename('Reliability Verification ratio')], axis=1)
    reliability_count = reliability_count.rename_axis('Reliability Verification').reset_index()
    organ_count = locationData['organ'].value_counts().rename('Organ count').rename_axis('Organ').reset_index()
    location_count = location.value_counts().rename('Location count').rename_axis('Location').reset_index()

    analysis = pd.concat([protein_count, antibody_count, reliability_count, organ_count, location_count], axis=1)
    print(analysis)
    analysis.to_csv(savePath2, index=False, mode='w')


if __name__ == '__main__':
    locationFilePath = dataDir + "annotations.csv"
    locationSavePath1 = dataDir + "locationOneHot.csv"
    locationSavePath2 = dataDir + "locationAnalysis.csv"
    locationAnalysis(locationFilePath, locationSavePath1, locationSavePath2)

