import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # __file__获取执行文件相对路径，整行为取上一级的上一级目录
sys.path.append(BASE_DIR)

import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
import scipy.stats as stats

import utils.distributed as du
import utils.checkpoint as cu
from datasets.loader import construct_loader
from models.classifier_model import getClassifier
from models.criterion import t_criterion
from models.train_classifier import load_best_classifier_model, cancerTest
from utils.args import parse_args
from utils.config_defaults import get_cfg, labelLists, dir_prefixs
from utils.eval_metrics import cal_metrics


cancerList = ["Glioma", "Melanoma", "SkinCancer"]
subtypesList = {"Glioma": ["normal", "highGrade", "lowGrade"],
                "Melanoma": ["normal", "NOS", "MetastaticSite"],
                "SkinCancer": ["normal", "BCC", "SCC"]}


def getPredictResult(classifier_model, database, cancer_dir):
    args = parse_args()
    cfg = get_cfg()

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set random seed from configs.
    random.seed(cfg.RNG_SEED + args.local_rank)
    np.random.seed(cfg.RNG_SEED + args.local_rank)
    torch.manual_seed(cfg.RNG_SEED + args.local_rank)
    torch.cuda.manual_seed(cfg.RNG_SEED + args.local_rank)
    torch.cuda.manual_seed_all(cfg.RNG_SEED + args.local_rank)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.local_rank != -1:
        torch.distributed.init_process_group(backend="nccl", init_method='env://')
        torch.cuda.set_device(torch.distributed.get_rank())
        device=torch.device("cuda", torch.distributed.get_rank())
    world_size = du.get_world_size()

    if du.is_master_proc():
        print('use {} gpus!'.format(world_size))

    # Classifier
    model = getClassifier(cfg, model_name=classifier_model, pretrain=False, SAE=None)
    model = model.to(device)
    if world_size > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[torch.distributed.get_rank()], output_device=torch.distributed.get_rank())

    result_prefix = "{}/results/{}/independent".format(cfg.DATA.RESULT_DIR, database)
    load_best_classifier_model(cfg, model, classifier_model, device, result_prefix=result_prefix)

    # result_prefix = "{}/results/cancer/{}/independent".format(cfg.DATA.RESULT_DIR, database)
    result_prefix = "{}/results/{}/{}/independent".format(cfg.DATA.RESULT_DIR, cancer_dir, database)

    for cancer in cancerList:
        path_prefix = dir_prefixs[cancer_dir]
        normal_file_path = "%snormal%s.csv" % (path_prefix, cancer)
        pathology_file_path = "%spathology%s.csv" % (path_prefix, cancer)

        normal_loader = construct_loader(cfg, normal_file_path, condition="normal", database="IHC", shuffle=False, drop_last=False)
        pathology_loader = construct_loader(cfg, pathology_file_path, condition="pathology", database="IHC", shuffle=False, drop_last=False)

        cancerTest(cfg, device, normal_loader, normal_file_path, None, model, model_name=classifier_model, multilabel=True, result_prefix=result_prefix)
        cancerTest(cfg, device, pathology_loader, pathology_file_path, None, model, model_name=classifier_model, multilabel=True, result_prefix=result_prefix)


def getProteinLevelLabel(resultData, labels, func="integrate", getGroundTruth=False, threshold=0.5):
    locations_pred = [i + '_pred' for i in labels]
    locations_pred_labels = [i + '_pred_labels' for i in labels]

    if getGroundTruth:
        proteinLabel = np.array(resultData[labels])
        predProteinLabel = np.array(resultData[locations_pred_labels])
        cal_metrics(None, proteinLabel, predProteinLabel, None, -1, locations=labels, csvWriter=None, fold=None, thresh="", split=None)

    resultData['Label Num'] = resultData[locations_pred_labels].sum(axis=1)
    print(resultData['Label Num'])
    groupData = resultData.groupby(['Protein Name', 'Protein Id'])
    print(groupData)

    proteinLabel = None
    if getGroundTruth:
        proteinLabel = groupData[labels].max()

    if func == "max":
        predProteinLabel = groupData[locations_pred_labels].max()
        predProtein = predProteinLabel
        predProtein.columns = locations_pred
    elif func == "avg":
        predProteinLabel = groupData[locations_pred_labels].mean()
        predProtein = predProteinLabel
        predProtein.columns = locations_pred
        predProteinLabel[predProteinLabel < 0.5] = 0
        predProteinLabel[predProteinLabel >= 0.5] = 1
    elif func == "pred_avg":
        predProteinLabel = groupData[locations_pred_labels].mean()
        predProteinResult = groupData[locations_pred].mean()
        predProteinResult.columns = locations_pred_labels
        predProtein = predProteinResult
        predProtein.columns = locations_pred
        predProteinLabel[predProteinResult < 0.5] = 0
        predProteinLabel[predProteinResult >= 0.5] = 1
    elif func == "integrate":
        ratio = 0.5
        predProteinLabel = groupData[locations_pred_labels].mean()
        predProteinResult = groupData[locations_pred].mean()
        predProteinResult.columns = locations_pred_labels
        predProteinLabel0 = predProteinLabel
        predProtein = ratio * predProteinLabel0 + (1 - ratio) * predProteinResult
        predProteinLabel[predProtein < 0.5] = 0
        predProteinLabel[predProtein >= 0.5] = 1
        predProtein.columns = locations_pred
    elif func == "integrate025":
        ratio = 0.25
        predProteinLabel = groupData[locations_pred_labels].mean()
        predProteinResult = groupData[locations_pred].mean()
        predProteinResult.columns = locations_pred_labels
        predProteinLabel0 = predProteinLabel
        predProtein = ratio * predProteinLabel0 + (1 - ratio) * predProteinResult
        predProteinLabel[predProtein < 0.5] = 0
        predProteinLabel[predProtein >= 0.5] = 1
        predProtein.columns = locations_pred
    elif func == "integrate075":
        ratio = 0.75
        predProteinLabel = groupData[locations_pred_labels].mean()
        predProteinResult = groupData[locations_pred].mean()
        predProteinResult.columns = locations_pred_labels
        predProteinLabel0 = predProteinLabel
        predProtein = ratio * predProteinLabel0 + (1 - ratio) * predProteinResult
        predProteinLabel[predProtein < 0.5] = 0
        predProteinLabel[predProtein >= 0.5] = 1
        predProtein.columns = locations_pred
    elif func == "rank":
        predProtein = groupData[locations_pred].mean()
        predProteinLabel = predProtein.copy()
        predProteinLabel.columns = locations_pred_labels
        predProteinLabelNum = groupData['Label Num'].max()
        ranked = predProteinLabel.rank(axis=1, ascending=False)
        for loc in locations_pred_labels:
            predProteinLabel.loc[ranked[loc] <= predProteinLabelNum, loc] = 1
            predProteinLabel.loc[ranked[loc] > predProteinLabelNum, loc] = 0
    elif func == "average":
        predProtein = groupData[locations_pred].mean()
        predProteinLabel = predProtein.copy()
        predProteinLabel.columns = locations_pred_labels
        all_pred_labels = t_criterion(np.array(predProteinLabel), threshold)
        predProteinLabel[locations_pred_labels] = all_pred_labels
    else:
        predProteinLabel = groupData[locations_pred_labels].max()
        predProtein = predProteinLabel.rename(locations_pred)

    all_pred_labels = t_criterion(np.array(predProtein[locations_pred]), threshold)
    predProteinLabel[locations_pred_labels] = all_pred_labels

    if getGroundTruth:
        cal_metrics(None, np.array(proteinLabel), np.array(predProteinLabel), None, -1, locations=labels, csvWriter=None, fold=None, thresh="", split=None)

    predData = pd.merge(predProtein, predProteinLabel, how='left', left_index=True, right_index=True)
    if proteinLabel is not None:
        predData = pd.merge(proteinLabel, predData, how='left', left_index=True, right_index=True)
    print(predData)

    return predData


def getGliomaResult(result_prefix=None, model_name=None, func=None, labels=None, threshold=0.5):
    normalData = pd.read_csv("{}/{}/preds/test_t=0.5_aug0_normalGlioma.csv".format(result_prefix, model_name), header=0, index_col=0)
    predData = getProteinLevelLabel(normalData, labels, func=func, getGroundTruth=True, threshold=threshold)

    proteins = normalData[['Protein Name', 'Protein Id']].drop_duplicates().reset_index(drop=True).reset_index()
    print(proteins)

    predData = pd.merge(proteins, predData, on='Protein Id', how='right').set_index('index')
    predData.columns = predData.columns.tolist()
    print(predData)
    predData.to_csv("{}/{}/preds/{}_protNormalGlioma.csv".format(result_prefix, model_name, func), index=True, mode='w')


    pathologyData = pd.read_csv("{}/{}/preds/test_t=0.5_aug0_pathologyGlioma.csv".format(result_prefix, model_name), header=0, index_col=0)

    highGradeData = pathologyData[pathologyData[['Glioma, malignant, High grade', 'Glioblastoma, NOS']].sum(axis=1) > 0]
    predData = getProteinLevelLabel(highGradeData, labels, func=func, getGroundTruth=False, threshold=threshold)

    predData = pd.merge(proteins, predData, on='Protein Id', how='right').set_index('index')
    predData.columns = predData.columns.tolist()
    print(predData)
    predData.to_csv("{}/{}/preds/{}_protHighGradeGlioma.csv".format(result_prefix, model_name, func), index=True, mode='w')


    lowGradeData = pathologyData[pathologyData['Glioma, malignant, Low grade'] == 1]
    predData = getProteinLevelLabel(lowGradeData, labels, func=func, getGroundTruth=False, threshold=threshold)

    predData = pd.merge(proteins, predData, on='Protein Id', how='right').set_index('index')
    predData.columns = predData.columns.tolist()
    print(predData)
    predData.to_csv("{}/{}/preds/{}_protLowGradeGlioma.csv".format(result_prefix, model_name, func), index=True, mode='w')



def getMelanomaResult(result_prefix=None, model_name=None, func=None, labels=None, threshold=0.5):
    normalData = pd.read_csv("{}/{}/preds/test_t=0.5_aug0_normalMelanoma.csv".format(result_prefix, model_name), header=0, index_col=0)
    predData = getProteinLevelLabel(normalData, labels, func=func, getGroundTruth=True, threshold=threshold)

    proteins = normalData[['Protein Name', 'Protein Id']].drop_duplicates().reset_index(drop=True).reset_index()
    print(proteins)


    predData = pd.merge(proteins, predData, on='Protein Id', how='right').set_index('index')
    predData.columns = predData.columns.tolist()
    print(predData)
    predData.to_csv("{}/{}/preds/{}_protNormalMelanoma.csv".format(result_prefix, model_name, func), index=True, mode='w')

    pathologyData = pd.read_csv("{}/{}/preds/test_t=0.5_aug0_pathologyMelanoma.csv".format(result_prefix, model_name), header=0, index_col=0)

    NOSData = pathologyData[pathologyData['Malignant melanoma, NOS'] == 1]
    predData = getProteinLevelLabel(NOSData, labels, func=func, getGroundTruth=False, threshold=threshold)

    predData = pd.merge(proteins, predData, on='Protein Id', how='right').set_index('index')
    predData.columns = predData.columns.tolist()
    print(predData)
    predData.to_csv("{}/{}/preds/{}_protNOSMelanoma.csv".format(result_prefix, model_name, func), index=True, mode='w')


    MetastaticSiteData = pathologyData[pathologyData['Malignant melanoma, Metastatic site'] == 1]
    predData = getProteinLevelLabel(MetastaticSiteData, labels, func=func, getGroundTruth=False, threshold=threshold)

    predData = pd.merge(proteins, predData, on='Protein Id', how='right').set_index('index')
    predData.columns = predData.columns.tolist()
    print(predData)
    predData.to_csv("{}/{}/preds/{}_protMetastaticSiteMelanoma.csv".format(result_prefix, model_name, func), index=True, mode='w')


def getSkinCancerResult(result_prefix=None, model_name=None, func=None, labels=None, threshold=0.5):
    normalData = pd.read_csv("{}/{}/preds/test_t=0.5_aug0_normalSkinCancer.csv".format(result_prefix, model_name), header=0, index_col=0)
    predData = getProteinLevelLabel(normalData, labels, func=func, getGroundTruth=True, threshold=threshold)

    proteins = normalData[['Protein Name', 'Protein Id']].drop_duplicates().reset_index(drop=True).reset_index()
    print(proteins)

    predData = pd.merge(proteins, predData, on='Protein Id', how='right').set_index('index')
    predData.columns = predData.columns.tolist()
    print(predData)
    predData.to_csv("{}/{}/preds/{}_protNormalSkinCancer.csv".format(result_prefix, model_name, func), index=True, mode='w')

    pathologyData = pd.read_csv("{}/{}/preds/test_t=0.5_aug0_pathologySkinCancer.csv".format(result_prefix, model_name), header=0, index_col=0)

    BCCData = pathologyData[pathologyData[['Basal cell carcinoma', 'BCC, low aggressive', 'BCC, high aggressive']].sum(axis=1) > 0]
    predData = getProteinLevelLabel(BCCData, labels, func=func, getGroundTruth=False, threshold=threshold)

    predData = pd.merge(proteins, predData, on='Protein Id', how='right').set_index('index')
    predData.columns = predData.columns.tolist()
    print(predData)
    predData.to_csv("{}/{}/preds/{}_protBCCSkinCancer.csv".format(result_prefix, model_name, func), index=True, mode='w')


    SCCData = pathologyData[pathologyData[['Squamous cell carcinoma, NOS', 'Squamous cell carcinoma in situ, NOS', 'Squamous cell carcinoma, metastatic, NOS']].sum(axis=1) > 0]
    predData = getProteinLevelLabel(SCCData, labels, func=func, getGroundTruth=False, threshold=threshold)

    predData = pd.merge(proteins, predData, on='Protein Id', how='right').set_index('index')
    predData.columns = predData.columns.tolist()
    print(predData)
    predData.to_csv("{}/{}/preds/{}_protSCCSkinCancer.csv".format(result_prefix, model_name, func), index=True, mode='w')


def getProteinLevelResult(classifier_model, database, cancer_dir, func=None, threshold=0.5):
    args = parse_args()
    cfg = get_cfg()

    result_prefix = "{}/results/{}/{}/independent".format(cfg.DATA.RESULT_DIR, cancer_dir, database)
    labels = labelLists.get(database, cfg.CLASSIFIER.LOCATIONS)

    getGliomaResult(result_prefix=result_prefix, model_name=classifier_model, func=func, labels=labels, threshold=threshold)
    getMelanomaResult(result_prefix=result_prefix, model_name=classifier_model, func=func, labels=labels, threshold=threshold)
    getSkinCancerResult(result_prefix=result_prefix, model_name=classifier_model, func=func, labels=labels, threshold=threshold)


def getTransProt(proteins, df_A, df_B, labels, col_name):
    locations_pred_labels = [i + '_pred_labels' for i in labels]

    intersected_proteins = pd.merge(df_A[['Protein Name', 'Protein Id']], df_B[['Protein Name', 'Protein Id']], how='inner')
    data_A = pd.merge(df_A, intersected_proteins, how='right')
    data_B = pd.merge(df_B, intersected_proteins, how='right')
    intersected_proteins[col_name] = np.all(np.equal(data_A[locations_pred_labels], data_B[locations_pred_labels]), axis=1)
    intersected_proteins[[col_name + '_' + loc for loc in labels]] = np.equal(data_A[locations_pred_labels], data_B[locations_pred_labels])
    print(intersected_proteins)
    proteins = pd.merge(proteins, intersected_proteins, on=['Protein Name', 'Protein Id'], how='left')
    return proteins


def filterGliomaTransProt(result_prefix=None, model_name=None, func=None, labels=None):
    normalData = pd.read_csv("{}/{}/preds/{}_protNormalGlioma.csv".format(result_prefix, model_name, func), header=0, index_col=0)
    highGradeData = pd.read_csv("{}/{}/preds/{}_protHighGradeGlioma.csv".format(result_prefix, model_name, func), header=0, index_col=0)
    lowGradeData = pd.read_csv("{}/{}/preds/{}_protLowGradeGlioma.csv".format(result_prefix, model_name, func), header=0, index_col=0)

    proteins = normalData[['Protein Name', 'Protein Id']].reset_index()

    proteins = getTransProt(proteins, normalData, highGradeData, labels, 'normal_highGrade')
    proteins = getTransProt(proteins, normalData, lowGradeData, labels, 'normal_lowGrade')
    proteins = getTransProt(proteins, highGradeData, lowGradeData, labels, 'highGrade_lowGrade')
    proteins = proteins.set_index('index')

    print(proteins)
    proteins.to_csv("{}/{}/preds/GliomaTransProt.csv".format(result_prefix, model_name), index=True, mode='w')


def filterMelanomaTransProt(result_prefix=None, model_name=None, func=None, labels=None):
    normalData = pd.read_csv("{}/{}/preds/{}_protNormalMelanoma.csv".format(result_prefix, model_name, func), header=0, index_col=0)
    NOSData = pd.read_csv("{}/{}/preds/{}_protNOSMelanoma.csv".format(result_prefix, model_name, func), header=0, index_col=0)
    MetastaticSiteData = pd.read_csv("{}/{}/preds/{}_protMetastaticSiteMelanoma.csv".format(result_prefix, model_name, func), header=0, index_col=0)

    proteins = normalData[['Protein Name', 'Protein Id']].reset_index()

    proteins = getTransProt(proteins, normalData, NOSData, labels, 'normal_NOS')
    proteins = getTransProt(proteins, normalData, MetastaticSiteData, labels, 'normal_MetastaticSite')
    proteins = getTransProt(proteins, NOSData, MetastaticSiteData, labels, 'NOS_MetastaticSite')
    proteins = proteins.set_index('index')

    print(proteins)
    proteins.to_csv("{}/{}/preds/MelanomaTransProt.csv".format(result_prefix, model_name), index=True, mode='w')


def filterSkinCancerTransProt(result_prefix=None, model_name=None, func=None, labels=None):
    normalData = pd.read_csv("{}/{}/preds/{}_protNormalSkinCancer.csv".format(result_prefix, model_name, func), header=0, index_col=0)
    BCCData = pd.read_csv("{}/{}/preds/{}_protBCCSkinCancer.csv".format(result_prefix, model_name, func), header=0, index_col=0)
    SCCData = pd.read_csv("{}/{}/preds/{}_protSCCSkinCancer.csv".format(result_prefix, model_name, func), header=0, index_col=0)

    proteins = normalData[['Protein Name', 'Protein Id']].reset_index()

    proteins = getTransProt(proteins, normalData, BCCData, labels, 'normal_BCC')
    proteins = getTransProt(proteins, normalData, SCCData, labels, 'normal_SCC')
    proteins = getTransProt(proteins, BCCData, SCCData, labels, 'BCC_SCC')
    proteins = proteins.set_index('index')

    print(proteins)
    proteins.to_csv("{}/{}/preds/SkinCancerTransProt.csv".format(result_prefix, model_name), index=True, mode='w')


def filterTranslocationProtein(classifier_model, database, cancer_dir, func=None):
    args = parse_args()
    cfg = get_cfg()

    result_prefix = "{}/results/{}/{}/independent".format(cfg.DATA.RESULT_DIR, cancer_dir, database)
    labels = labelLists.get(database, cfg.CLASSIFIER.LOCATIONS)

    filterGliomaTransProt(result_prefix=result_prefix, model_name=classifier_model, func=func, labels=labels)
    filterMelanomaTransProt(result_prefix=result_prefix, model_name=classifier_model, func=func, labels=labels)
    filterSkinCancerTransProt(result_prefix=result_prefix, model_name=classifier_model, func=func, labels=labels)


def Ttest(proteins, df_A, df_B, labels, col_name):
    proteinList = proteins[proteins[col_name] == False]['Protein Id']
    print(col_name)
    print(proteinList)
    base_ttest_col = ['T_statistic', 'T_pvalue']
    ttest_col = ['_'.join([col_name, loc, col]) for loc in labels for col in base_ttest_col]
    proteins[ttest_col] = np.nan

    for prot in proteinList:
        data_A = df_A[df_A['Protein Id'] == prot]
        data_B = df_B[df_B['Protein Id'] == prot]

        for loc in labels:
            A = data_A[loc + '_pred']
            B = data_B[loc + '_pred']

            T_statistic, T_pvalue = stats.ttest_ind(B, A)

            ttest_col = ['_'.join([col_name, loc, col]) for col in base_ttest_col]
            proteins.loc[proteins['Protein Id'] == prot, ttest_col] = T_statistic, T_pvalue
    return proteins


def calProbStatistic(proteins, df, labels, data_name):
    locations_pred = [i + '_pred' for i in labels]
    num = df.groupby(['Protein Name'])['Protein Id'].count().rename(data_name + 'num').reset_index()
    proteins = pd.merge(proteins, num, on=['Protein Name'], how='left')
    statisticData = df.groupby(['Protein Name', 'Protein Id'])[locations_pred].agg([np.mean, np.std])
    statisticData.columns = [data_name + '_'.join(col) for col in statisticData.columns]
    statisticData = statisticData.reset_index()
    proteins = pd.merge(proteins, statisticData, on=['Protein Name', 'Protein Id'], how='left')
    return proteins


def GliomaTransTtest(result_prefix=None, model_name=None, labels=None):
    proteinData = pd.read_csv("{}/{}/preds/GliomaTransProt.csv".format(result_prefix, model_name), header=0, index_col=0)
    normalData = pd.read_csv("{}/{}/preds/test_t=0.5_aug0_normalGlioma.csv".format(result_prefix, model_name), header=0, index_col=0)
    pathologyData = pd.read_csv("{}/{}/preds/test_t=0.5_aug0_pathologyGlioma.csv".format(result_prefix, model_name), header=0, index_col=0)
    highGradeData = pathologyData[pathologyData[['Glioma, malignant, High grade', 'Glioblastoma, NOS']].sum(axis=1) > 0]
    lowGradeData = pathologyData[pathologyData['Glioma, malignant, Low grade'] == 1]
    proteinData = proteinData.reset_index()

    proteinData = calProbStatistic(proteinData, normalData, labels, 'normal_')
    proteinData = calProbStatistic(proteinData, highGradeData, labels, 'highGrade_')
    proteinData = calProbStatistic(proteinData, lowGradeData, labels, 'lowGrade_')

    proteinData = Ttest(proteinData, normalData, highGradeData, labels, 'normal_highGrade')
    proteinData = Ttest(proteinData, normalData, lowGradeData, labels, 'normal_lowGrade')
    proteinData = Ttest(proteinData, highGradeData, lowGradeData, labels, 'highGrade_lowGrade')

    proteinData = proteinData.set_index('index')
    print(proteinData)
    proteinData.to_csv("{}/{}/preds/GliomaTtest.csv".format(result_prefix, model_name), index=True, mode='w')


def MelanomaTransTtest(result_prefix=None, model_name=None, labels=None):
    proteinData = pd.read_csv("{}/{}/preds/MelanomaTransProt.csv".format(result_prefix, model_name), header=0, index_col=0)
    normalData = pd.read_csv("{}/{}/preds/test_t=0.5_aug0_normalMelanoma.csv".format(result_prefix, model_name), header=0, index_col=0)
    pathologyData = pd.read_csv("{}/{}/preds/test_t=0.5_aug0_pathologyMelanoma.csv".format(result_prefix, model_name), header=0, index_col=0)
    NOSData = pathologyData[pathologyData['Malignant melanoma, NOS'] == 1]
    MetastaticSiteData = pathologyData[pathologyData['Malignant melanoma, Metastatic site'] == 1]
    proteinData = proteinData.reset_index()

    proteinData = calProbStatistic(proteinData, normalData, labels, 'normal_')
    proteinData = calProbStatistic(proteinData, NOSData, labels, 'NOS_')
    proteinData = calProbStatistic(proteinData, MetastaticSiteData, labels, 'MetastaticSite_')

    proteinData = Ttest(proteinData, normalData, NOSData, labels, 'normal_NOS')
    proteinData = Ttest(proteinData, normalData, MetastaticSiteData, labels, 'normal_MetastaticSite')
    proteinData = Ttest(proteinData, NOSData, MetastaticSiteData, labels, 'NOS_MetastaticSite')


    proteinData = proteinData.set_index('index')
    print(proteinData)
    proteinData.to_csv("{}/{}/preds/MelanomaTtest.csv".format(result_prefix, model_name), index=True, mode='w')


def SkinCancerTransTtest(result_prefix=None, model_name=None, labels=None):
    proteinData = pd.read_csv("{}/{}/preds/SkinCancerTransProt.csv".format(result_prefix, model_name), header=0, index_col=0)
    normalData = pd.read_csv("{}/{}/preds/test_t=0.5_aug0_normalSkinCancer.csv".format(result_prefix, model_name), header=0, index_col=0)
    pathologyData = pd.read_csv("{}/{}/preds/test_t=0.5_aug0_pathologySkinCancer.csv".format(result_prefix, model_name), header=0, index_col=0)
    BCCData = pathologyData[pathologyData[['Basal cell carcinoma', 'BCC, low aggressive', 'BCC, high aggressive']].sum(axis=1) > 0]
    SCCData = pathologyData[pathologyData[['Squamous cell carcinoma, NOS', 'Squamous cell carcinoma in situ, NOS', 'Squamous cell carcinoma, metastatic, NOS']].sum(axis=1) > 0]
    proteinData = proteinData.reset_index()

    proteinData = calProbStatistic(proteinData, normalData, labels, 'normal_')
    proteinData = calProbStatistic(proteinData, BCCData, labels, 'BCC_')
    proteinData = calProbStatistic(proteinData, SCCData, labels, 'SCC_')

    proteinData = Ttest(proteinData, normalData, BCCData, labels, 'normal_BCC')
    proteinData = Ttest(proteinData, normalData, SCCData, labels, 'normal_SCC')
    proteinData = Ttest(proteinData, BCCData, SCCData, labels, 'BCC_SCC')


    proteinData = proteinData.set_index('index')
    print(proteinData)
    proteinData.to_csv("{}/{}/preds/SkinCancerTtest.csv".format(result_prefix, model_name), index=True, mode='w')


def transTtest(classifier_model, database, cancer_dir):
    args = parse_args()
    cfg = get_cfg()

    result_prefix = "{}/results/{}/{}/independent".format(cfg.DATA.RESULT_DIR, cancer_dir, database)
    labels = labelLists.get(database, cfg.CLASSIFIER.LOCATIONS)

    GliomaTransTtest(result_prefix=result_prefix, model_name=classifier_model, labels=labels)
    MelanomaTransTtest(result_prefix=result_prefix, model_name=classifier_model, labels=labels)
    SkinCancerTransTtest(result_prefix=result_prefix, model_name=classifier_model, labels=labels)


def filterCandidate(proteinData, df_A, df_B, labels, col1, col2, p_threshold=0.01):
    col_name = '_'.join([col1, col2])
    print(col_name)
    significance = proteinData[['_'.join([col_name, loc, 'T_pvalue']) for loc in labels]] < p_threshold
    trans = proteinData[['_'.join([col_name, loc]) for loc in labels]] == False
    significance.columns = trans.columns
    data = proteinData[np.any(significance & trans, axis=1)]

    data = data.reset_index()
    data = pd.merge(data, df_A, on=['Protein Name', 'Protein Id'], how='left')
    data = pd.merge(data, df_B, on=['Protein Name', 'Protein Id'], how='left')
    data = data.set_index('index')

    new_cols = ['Protein Name', 'Protein Id']
    new_cols += [col_name]
    new_cols += ['_'.join([col_name, loc]) for loc in labels]
    new_cols += ['_'.join([col1, 'num'])]
    new_cols += ['_'.join([col1, loc, 'pred', statistic]) for loc in labels for statistic in ['mean', 'std']]
    new_cols += ['_'.join([col2, 'num'])]
    new_cols += ['_'.join([col2, loc, 'pred', statistic]) for loc in labels for statistic in ['mean', 'std']]
    new_cols += ['_'.join([col_name, loc, col]) for loc in labels for col in ['T_statistic', 'T_pvalue']]
    new_cols += ['_'.join([col1, loc, 'pred']) for loc in labels]
    new_cols += ['_'.join([col1, loc, 'pred_labels']) for loc in labels]
    new_cols += ['_'.join([col2, loc, 'pred']) for loc in labels]
    new_cols += ['_'.join([col2, loc, 'pred_labels']) for loc in labels]
    # print(new_cols)
    data = data[new_cols]

    for loc in labels:
        data['_'.join([col_name, loc])] = data['_'.join([col2, loc, 'pred_labels'])] - data['_'.join([col1, loc, 'pred_labels'])]

    data['min_T_pvalue'] = data[['_'.join([col_name, loc, 'T_pvalue']) for loc in labels]].min(axis=1)
    data = data.sort_values('min_T_pvalue')

    print(data)
    return data


def getGliomaCandidate(result_prefix=None, model_name=None, func=None, labels=None, p_threshold=0.01):
    proteinData = pd.read_csv("{}/{}/preds/GliomaTtest.csv".format(result_prefix, model_name), header=0, index_col=0)
    normalData = pd.read_csv("{}/{}/preds/{}_protNormalGlioma.csv".format(result_prefix, model_name, func), header=0, index_col=0)
    highGradeData = pd.read_csv("{}/{}/preds/{}_protHighGradeGlioma.csv".format(result_prefix, model_name, func), header=0, index_col=0)
    lowGradeData = pd.read_csv("{}/{}/preds/{}_protLowGradeGlioma.csv".format(result_prefix, model_name, func), header=0, index_col=0)
    normalData.columns = normalData.columns.tolist()[:2] + ['_'.join(['normal', col]) for col in normalData.columns.tolist()[2:]]
    highGradeData.columns = highGradeData.columns.tolist()[:2] + ['_'.join(['highGrade', col]) for col in highGradeData.columns.tolist()[2:]]
    lowGradeData.columns = lowGradeData.columns.tolist()[:2] + ['_'.join(['lowGrade', col]) for col in lowGradeData.columns.tolist()[2:]]

    normal_highGrade_candidate = filterCandidate(proteinData, normalData, highGradeData, labels, 'normal', 'highGrade', p_threshold=p_threshold)
    normal_lowGrade_candidate = filterCandidate(proteinData, normalData, lowGradeData, labels, 'normal', 'lowGrade', p_threshold=p_threshold)
    highGrade_lowGrade_candidate = filterCandidate(proteinData, highGradeData, lowGradeData, labels, 'highGrade', 'lowGrade', p_threshold=p_threshold)

    normal_highGrade_candidate.to_csv("{}/{}/preds/{}_Glioma_normal_highGrade_Candidate.csv".format(result_prefix, model_name, p_threshold), index=True, mode='w')
    normal_lowGrade_candidate.to_csv("{}/{}/preds/{}_Glioma_normal_lowGrade_Candidate.csv".format(result_prefix, model_name, p_threshold), index=True, mode='w')
    highGrade_lowGrade_candidate.to_csv("{}/{}/preds/{}_Glioma_highGrade_lowGrade_Candidate.csv".format(result_prefix, model_name, p_threshold), index=True, mode='w')


def getMelanomaCandidate(result_prefix=None, model_name=None, func=None, labels=None, p_threshold=0.01):
    proteinData = pd.read_csv("{}/{}/preds/MelanomaTtest.csv".format(result_prefix, model_name), header=0, index_col=0)
    normalData = pd.read_csv("{}/{}/preds/{}_protNormalMelanoma.csv".format(result_prefix, model_name, func), header=0, index_col=0)
    NOSData = pd.read_csv("{}/{}/preds/{}_protNOSMelanoma.csv".format(result_prefix, model_name, func), header=0, index_col=0)
    MetastaticSiteData = pd.read_csv("{}/{}/preds/{}_protMetastaticSiteMelanoma.csv".format(result_prefix, model_name, func), header=0, index_col=0)
    normalData.columns = normalData.columns.tolist()[:2] + ['_'.join(['normal', col]) for col in normalData.columns.tolist()[2:]]
    NOSData.columns = NOSData.columns.tolist()[:2] + ['_'.join(['NOS', col]) for col in NOSData.columns.tolist()[2:]]
    MetastaticSiteData.columns = MetastaticSiteData.columns.tolist()[:2] + ['_'.join(['MetastaticSite', col]) for col in MetastaticSiteData.columns.tolist()[2:]]

    normal_NOS_candidate = filterCandidate(proteinData, normalData, NOSData, labels, 'normal', 'NOS', p_threshold=p_threshold)
    normal_MetastaticSite_candidate = filterCandidate(proteinData, normalData, MetastaticSiteData, labels, 'normal', 'MetastaticSite', p_threshold=p_threshold)
    NOS_MetastaticSite_candidate = filterCandidate(proteinData, NOSData, MetastaticSiteData, labels, 'NOS', 'MetastaticSite', p_threshold=p_threshold)

    normal_NOS_candidate.to_csv("{}/{}/preds/{}_Melanoma_normal_NOS_Candidate.csv".format(result_prefix, model_name, p_threshold), index=True, mode='w')
    normal_MetastaticSite_candidate.to_csv("{}/{}/preds/{}_Melanoma_normal_MetastaticSite_Candidate.csv".format(result_prefix, model_name, p_threshold), index=True, mode='w')
    NOS_MetastaticSite_candidate.to_csv("{}/{}/preds/{}_Melanoma_NOS_MetastaticSite_Candidate.csv".format(result_prefix, model_name, p_threshold), index=True, mode='w')


def getSkinCancerCandidate(result_prefix=None, model_name=None, func=None, labels=None, p_threshold=0.01):
    proteinData = pd.read_csv("{}/{}/preds/SkinCancerTtest.csv".format(result_prefix, model_name), header=0, index_col=0)
    normalData = pd.read_csv("{}/{}/preds/{}_protNormalSkinCancer.csv".format(result_prefix, model_name, func), header=0, index_col=0)
    BCCData = pd.read_csv("{}/{}/preds/{}_protBCCSkinCancer.csv".format(result_prefix, model_name, func), header=0, index_col=0)
    SCCData = pd.read_csv("{}/{}/preds/{}_protSCCSkinCancer.csv".format(result_prefix, model_name, func), header=0, index_col=0)
    normalData.columns = normalData.columns.tolist()[:2] + ['_'.join(['normal', col]) for col in normalData.columns.tolist()[2:]]
    BCCData.columns = BCCData.columns.tolist()[:2] + ['_'.join(['BCC', col]) for col in BCCData.columns.tolist()[2:]]
    SCCData.columns = SCCData.columns.tolist()[:2] + ['_'.join(['SCC', col]) for col in SCCData.columns.tolist()[2:]]

    normal_BCC_candidate = filterCandidate(proteinData, normalData, BCCData, labels, 'normal', 'BCC', p_threshold=p_threshold)
    normal_SCC_candidate = filterCandidate(proteinData, normalData, SCCData, labels, 'normal', 'SCC', p_threshold=p_threshold)
    BCC_SCC_candidate = filterCandidate(proteinData, BCCData, SCCData, labels, 'BCC', 'SCC', p_threshold=p_threshold)

    normal_BCC_candidate.to_csv("{}/{}/preds/{}_SkinCancer_normal_BCC_Candidate.csv".format(result_prefix, model_name, p_threshold), index=True, mode='w')
    normal_SCC_candidate.to_csv("{}/{}/preds/{}_SkinCancer_normal_SCC_Candidate.csv".format(result_prefix, model_name, p_threshold), index=True, mode='w')
    BCC_SCC_candidate.to_csv("{}/{}/preds/{}_SkinCancer_BCC_SCC_Candidate.csv".format(result_prefix, model_name, p_threshold), index=True, mode='w')


def getCandidateBiomarkers(classifier_model, database, cancer_dir, func=None, p_threshold=0.01):
    args = parse_args()
    cfg = get_cfg()

    result_prefix = "{}/results/{}/{}/independent".format(cfg.DATA.RESULT_DIR, cancer_dir, database)
    labels = labelLists.get(database, cfg.CLASSIFIER.LOCATIONS)

    getGliomaCandidate(result_prefix=result_prefix, model_name=classifier_model, func=func, labels=labels, p_threshold=p_threshold)
    getMelanomaCandidate(result_prefix=result_prefix, model_name=classifier_model, func=func, labels=labels, p_threshold=p_threshold)
    getSkinCancerCandidate(result_prefix=result_prefix, model_name=classifier_model, func=func, labels=labels, p_threshold=p_threshold)


def candidateAnalysis(classifier_model, database, cancer_dir, p_threshold=0.01):
    args = parse_args()
    cfg = get_cfg()

    result_prefix = "{}/results/{}/{}/independent".format(cfg.DATA.RESULT_DIR, cancer_dir, database)
    labels = labelLists.get(database, cfg.CLASSIFIER.LOCATIONS)

    df = []
    for cancer in cancerList:
        subtypes = subtypesList.get(cancer)
        for i in range(len(subtypes) - 1):
            for j in range(i + 1, len(subtypes)):
                data = pd.read_csv("{}/{}/preds/{}_{}_{}_{}_Candidate.csv".format(result_prefix, classifier_model, p_threshold, cancer, subtypes[i], subtypes[j]), header=0, index_col=0)
                df.append([cancer, subtypes[i], subtypes[j], len(data)])
    df = pd.DataFrame(df, columns=['Cancer', 'Subtype1', 'Subtype2', 'Candidate num'])
    print(df)
    df.to_csv("{}/{}/preds/{}_CandidateAnalysis.csv".format(result_prefix, classifier_model, p_threshold), index=True, mode='w')




def main():
    """
    Main function to spawn the test process.
    """
    classifier_model = "Vislocas_mlce_lr-000005_bn_drop-01_attn-drop-01_drop-path-01_batch12_seed6293_wd-005_aug_no-normalized"
    database = "IHC"
    cancer_dir = "cancer"
    p_threshold = 1e-4

    getPredictResult(classifier_model=classifier_model, database=database, cancer_dir=cancer_dir)
    if du.get_world_size() > 1:
        dist.barrier()

    getProteinLevelResult(classifier_model=classifier_model, database=database, cancer_dir=cancer_dir, func="average", threshold=0.5)
    filterTranslocationProtein(classifier_model=classifier_model, database=database, cancer_dir=cancer_dir, func="average")
    transTtest(classifier_model=classifier_model, database=database, cancer_dir=cancer_dir)
    getCandidateBiomarkers(classifier_model=classifier_model, database=database, cancer_dir=cancer_dir, func="average", p_threshold=p_threshold)
    candidateAnalysis(classifier_model=classifier_model, database=database, cancer_dir=cancer_dir, p_threshold=p_threshold)


if __name__ == "__main__":
    main()