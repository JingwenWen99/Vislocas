# import numpy as np
import pandas as pd


dataDir = "data/"


if __name__ == '__main__':
    tissueData = pd.read_csv(dataDir + "tissueUrl_original.csv", header=0)
    print("原始Tissue数据条数：", len(tissueData))
    tissueData = tissueData.drop(tissueData.loc[tissueData['URL'].str.rsplit(".", expand=True, n=1)[1] == 'tif'].index)
    print("去除tif后，Tissue数据条数：", len(tissueData))
    tissueData.to_csv(dataDir + "tissueUrl.csv", header=True, index=None, mode='w')

    PathologyData = pd.read_csv(dataDir + "pathologyUrl_original.csv", header=0)
    print("原始Pathology数据条数：", len(PathologyData))
    PathologyData = PathologyData.drop(PathologyData.loc[PathologyData['URL'].str.rsplit(".", expand=True, n=1)[1] == 'tif'].index)
    print("去除tif后，Pathology数据条数：", len(PathologyData))
    PathologyData.to_csv(dataDir + "pathologyUrl.csv", header=True, index=None, mode='w')



