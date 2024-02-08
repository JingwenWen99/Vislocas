import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # __file__获取执行文件相对路径，整行为取上一级的上一级目录
sys.path.append(BASE_DIR)

import math
import random
import csv

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist

import utils.distributed as du
import utils.checkpoint as cu
from datasets.loader import construct_loader
from models.autoencoder import AutoEncoderLayer, StackedAutoEncoder, SAEmodel
from models.classifier_model import getClassifier
from models.criterion import t_criterion
from models.losses import get_loss_func
from models.train_autoencoder import load_best_model, test_layer
from models.train_classifier import load_best_classifier_model, test, val_epoch
from utils.args import parse_args
from utils.config_defaults import get_cfg, labelLists, dir_prefixs
from utils.eval_metrics import cal_metrics, get_curve
from utils.optimizer import construct_optimizer, get_optimizer_func
from utils.scheduler import construct_scheduler, get_scheduler_func
from tools.cal_metrics import get_metrics, cal_mean_std


organList = ["Brain", "Connective & soft tissue", "Female tissues", "Gastrointestinal tract", "Kidney & urinary bladder", "Skin"]
brainList = ["Caudate", "Cerebellum", "Cerebral cortex", "Hippocampus"]


def get_balanced_result(resultPath, labels, location_num, func="balanced1", threshold=0.5, get_threshold=False, beta=1, csvWriter=None, split_num=-1, fold=0, thresh='', split=''):
    resultData = pd.read_csv(resultPath, header=0, index_col=0)
    locations_pred = [i + '_pred' for i in labels]
    locations_pred_labels = [i + '_pred_labels' for i in labels]
    original_locations_pred = ['original_' + i + '_pred' for i in labels]
    original_locations_pred_labels = ['original_' + i + '_pred_labels' for i in labels]

    resultData[original_locations_pred] = resultData[locations_pred]
    resultData[original_locations_pred_labels] = resultData[locations_pred_labels]

    if func == "balanced1":
        max_num = max(location_num)
        for i in range(len(labels)):
            resultData[locations_pred[i]] = resultData[locations_pred[i]] * max_num / location_num[i]

        all_pred_labels = t_criterion(np.array(resultData[locations_pred]), threshold)
        resultData[locations_pred_labels] = all_pred_labels
    elif func == "balanced2":
        max_num = max(location_num)
        for i in range(len(labels)):
            ratio = np.log2(max_num / location_num[i])
            if max_num == location_num[i]:
                ratio = 1
            resultData[locations_pred[i]] = resultData[locations_pred[i]] * ratio

        all_pred_labels = t_criterion(np.array(resultData[locations_pred]), threshold)
        resultData[locations_pred_labels] = all_pred_labels
    elif func == "balanced3":
        max_num = max(location_num)
        for i in range(len(labels)):
            ratio = np.log(max_num / location_num[i])
            if max_num == location_num[i]:
                ratio = 1
            resultData[locations_pred[i]] = resultData[locations_pred[i]] * ratio

        all_pred_labels = t_criterion(np.array(resultData[locations_pred]), threshold)
        resultData[locations_pred_labels] = all_pred_labels
    elif func == "rank":
        max_p = resultData[locations_pred].max(axis=1)
        min_p = resultData[locations_pred].min(axis=1)
        cutoff = max_p - (max_p - min_p) * threshold
        resultData[locations_pred_labels] = resultData[locations_pred]
        print(resultData[locations_pred_labels])
        for l in locations_pred_labels:
            resultData.loc[resultData[l] >= cutoff, l] = 1
            resultData.loc[resultData[l] < cutoff, l] = 0
        print(resultData[locations_pred_labels])
    elif func == "balanced4":
        mean_p = resultData[locations_pred].mean()
        std_p = resultData[locations_pred].std()
        for l in locations_pred:
            resultData[l] = (resultData[l] - mean_p[l]) / std_p[l]

        all_pred_labels = t_criterion(np.array(resultData[locations_pred]), threshold)
        resultData[locations_pred_labels] = all_pred_labels
    elif func == "balanced5":
        max_idx = np.argmax(np.array(resultData[locations_pred]), axis=1)
        t_k = [location_num[i] for i in max_idx]
        for i in range(len(labels)):
            resultData[locations_pred[i]] = resultData[locations_pred[i]] * t_k / location_num[i]

        all_pred_labels = t_criterion(np.array(resultData[locations_pred]), threshold)
        resultData[locations_pred_labels] = all_pred_labels
    elif func == "balanced6":
        max_idx = np.argmax(np.array(resultData[locations_pred]), axis=1)
        t_k = [location_num[i] for i in max_idx]
        for i in range(len(labels)):
            ratio = [np.log2(t / location_num[i]) if t != location_num[i] else 1 for t in t_k]
            resultData[locations_pred[i]] = resultData[locations_pred[i]] * ratio

        all_pred_labels = t_criterion(np.array(resultData[locations_pred]), threshold)
        resultData[locations_pred_labels] = all_pred_labels
    elif func == "balanced7":
        max_idx = np.argmax(np.array(resultData[locations_pred]), axis=1)
        t_k = [location_num[i] for i in max_idx]
        for i in range(len(labels)):
            ratio = [np.log(t / location_num[i]) if t != location_num[i] else 1 for t in t_k]
            resultData[locations_pred[i]] = resultData[locations_pred[i]] * ratio

        all_pred_labels = t_criterion(np.array(resultData[locations_pred]), threshold)
        resultData[locations_pred_labels] = all_pred_labels
    elif func == "threshold":
        if get_threshold:
            threshold = get_curve(None, np.array(resultData[locations_pred_labels]), np.array(resultData[locations_pred]), beta=beta, writer=None, locations=labels)
        all_pred_labels = t_criterion(np.array(resultData[locations_pred]), threshold)
        resultData[locations_pred_labels] = all_pred_labels
        thresh_locations = ['thresh_' + i for i in labels]
        resultData[thresh_locations] = threshold
    elif func == "balanced8":
        sum_num = sum(location_num)
        for i in range(len(labels)):
            ratio = np.log(sum_num / location_num[i]) + 1
            resultData[locations_pred[i]] = resultData[locations_pred[i]] * ratio

        all_pred_labels = t_criterion(np.array(resultData[locations_pred]), threshold)
        resultData[locations_pred_labels] = all_pred_labels
    elif func == "balanced9":
        max_num = max(location_num)
        for i in range(len(labels)):
            ratio = np.log(max_num / location_num[i]) + 1
            resultData[locations_pred[i]] = resultData[locations_pred[i]] * ratio

        all_pred_labels = t_criterion(np.array(resultData[locations_pred]), threshold)
        resultData[locations_pred_labels] = all_pred_labels
    elif func == "balanced10":
        max_num = max(location_num)
        for i in range(len(labels)):
            ratio = np.log10(max_num / location_num[i]) + 1
            resultData[locations_pred[i]] = resultData[locations_pred[i]] * ratio

        all_pred_labels = t_criterion(np.array(resultData[locations_pred]), threshold)
        resultData[locations_pred_labels] = all_pred_labels
    elif func == "balanced11":
        sum_num = sum(location_num)
        for i in range(len(labels)):
            ratio = np.log10(sum_num / location_num[i]) + 1
            resultData[locations_pred[i]] = resultData[locations_pred[i]] * ratio

        all_pred_labels = t_criterion(np.array(resultData[locations_pred]), threshold)
        resultData[locations_pred_labels] = all_pred_labels
    elif func == "balanced12":
        sum_num = sum(location_num)
        for i in range(len(labels)):
            ratio = np.log(sum_num) / np.log(location_num[i])
            resultData[locations_pred[i]] = resultData[locations_pred[i]] * ratio

        all_pred_labels = t_criterion(np.array(resultData[locations_pred]), threshold)
        resultData[locations_pred_labels] = all_pred_labels
    elif func == "balanced13":
        sum_num = sum(location_num)
        for i in range(len(labels)):
            ratio = np.log(sum_num / location_num[i])
            resultData[locations_pred[i]] = resultData[locations_pred[i]] * ratio

        all_pred_labels = t_criterion(np.array(resultData[locations_pred]), threshold)
        resultData[locations_pred_labels] = all_pred_labels
    elif func == "balanced14":
        sum_num = sum(location_num)
        for i in range(len(labels)):
            ratio = np.log(sum_num / location_num[i] + 1)
            resultData[locations_pred[i]] = resultData[locations_pred[i]] * ratio

        all_pred_labels = t_criterion(np.array(resultData[locations_pred]), threshold)
        resultData[locations_pred_labels] = all_pred_labels
    elif func == "balanced15":
        max_num = max(location_num)
        for i in range(len(labels)):
            ratio = np.log(max_num / location_num[i] + 1)
            resultData[locations_pred[i]] = resultData[locations_pred[i]] * ratio

        all_pred_labels = t_criterion(np.array(resultData[locations_pred]), threshold)
        resultData[locations_pred_labels] = all_pred_labels
    elif func == "original":
        all_pred_labels = t_criterion(np.array(resultData[locations_pred]), threshold)
        resultData[locations_pred_labels] = all_pred_labels
    elif func == "balanced9_rank":
        max_num = max(location_num)
        for i in range(len(labels)):
            ratio = np.log(max_num / location_num[i]) + 1
            resultData[locations_pred[i]] = resultData[locations_pred[i]] * ratio

        LabelNum = resultData[locations_pred_labels].sum(axis=1)
        resultData[locations_pred_labels] = resultData[locations_pred].copy()
        resultData[locations_pred_labels].columns = locations_pred_labels
        ranked = resultData[locations_pred_labels].rank(axis=1, ascending=False)
        for loc in locations_pred_labels:
            resultData.loc[ranked[loc] <= LabelNum, loc] = 1
            resultData.loc[ranked[loc] > LabelNum, loc] = 0
    elif func == "balanced16":
        max_num = max(location_num)
        for i in range(len(labels)):
            ratio = np.log2(max_num / location_num[i] + 1)
            resultData[locations_pred[i]] = resultData[locations_pred[i]] * ratio

        all_pred_labels = t_criterion(np.array(resultData[locations_pred]), threshold)
        resultData[locations_pred_labels] = all_pred_labels
    elif func == "balanced17":
        max_num = max(location_num)
        for i in range(len(labels)):
            ratio = np.log2(max_num / location_num[i]) + 1
            resultData[locations_pred[i]] = resultData[locations_pred[i]] * ratio

        all_pred_labels = t_criterion(np.array(resultData[locations_pred]), threshold)
        resultData[locations_pred_labels] = all_pred_labels


    proteinLabel = np.array(resultData[labels])
    predProteinLabel = np.array(resultData[locations_pred_labels])
    cal_metrics(None, proteinLabel, predProteinLabel, None, -1, locations=labels, csvWriter=csvWriter, randomSplit=split_num, fold=fold, thresh="{}_{}".format(thresh, func), split=split)

    return resultData, threshold





def main():
    """
    Main function to spawn the test process.
    """
    args = parse_args()
    cfg = get_cfg()

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.local_rank != -1:
        torch.distributed.init_process_group(backend="nccl", init_method='env://')
        torch.cuda.set_device(torch.distributed.get_rank())
        device=torch.device("cuda", torch.distributed.get_rank())
    world_size = du.get_world_size()

    if du.is_master_proc():
        print('use {} gpus!'.format(world_size))


    # for classifier_model in ["cct_modified56_mlce_lr-000005_bn_drop-01_attn-drop-01_drop-path-01_batch12_seed6293_wd-005_aug_no-normalized",
    #         "cct_modified68_mlce_lr-000005_bn_drop-01_attn-drop-01_drop-path-01_batch12_seed6293_wd-005_aug_no-normalized",
    #         "cct_modified69_mlce_lr-000005_bn_drop-01_attn-drop-01_drop-path-01_batch12_seed6293_wd-005_aug_no-normalized",
    #         "cct_modified70_mlce_lr-000005_bn_drop-01_attn-drop-01_drop-path-01_batch12_seed6293_wd-005_aug_no-normalized",
    #         "cct_modified71_mlce_lr-000005_bn_drop-01_attn-drop-01_drop-path-01_batch12_seed6293_wd-005_aug_no-normalized",
    #         # "cct_modified72_mlce_lr-000005_bn_drop-01_attn-drop-01_drop-path-01_batch12_seed6293_wd-005_aug_no-normalized",
    #         "cct_modified73_mlce_lr-000005_bn_drop-01_attn-drop-01_drop-path-01_batch12_seed6293_wd-005_aug_no-normalized",
    #         "cct_modified74_mlce_lr-000005_bn_drop-01_attn-drop-01_drop-path-01_batch12_seed6293_wd-005_aug_no-normalized",
    #         "cct_modified75_mlce_lr-000005_bn_drop-01_attn-drop-01_drop-path-01_batch12_seed6293_wd-005_aug_no-normalized",
    #         "cct_modified76_mlce_lr-000005_bn_drop-01_attn-drop-01_drop-path-01_batch12_seed6293_wd-005_aug_no-normalized",
    #         "cct_modified77_mlce_lr-000005_bn_drop-01_attn-drop-01_drop-path-01_batch12_seed6293_wd-005_aug_no-normalized",
    #         "cct_modified78_mlce_lr-000005_bn_drop-01_attn-drop-01_drop-path-01_batch12_seed6293_wd-005_aug_no-normalized"]:
    for classifier_model in ["cct_modified72_mlce_lr-000005_bn_drop-01_attn-drop-01_drop-path-01_batch12_seed6293_wd-005_aug_no-normalized"]:
    # classifier_model = "cct_modified72_mlce_lr-000005_bn_drop-01_attn-drop-01_drop-path-01_batch12_seed6293_wd-005_aug_no-normalized"

        for database in cfg.DATA.DATASET_NAME:
            split_list = range(5)
            if database in ["IHC"]:
                split_list = [-2] + list(range(5))
                # split_list = [-2] + list(range(17))
            elif database in ["MSTLoc"]:
                split_list = [-2] + list(range(5))
            # elif database in ["laceDNN"]:
            #     split_list = [0, 1, 3]
            for split_num in split_list:
                proteinLevel = False
                if database in ["GraphLoc", "laceDNN"]:
                    proteinLevel = True
                # if database in ["IHC"]:
                #     proteinLevel = True

                multilabel = True

                getSPE = False
                getAuc = False
                getMcc = False

                if database in ["PScL-HDeep", "PScL-2LSAESM", "PScL-DDCFPred", 'su']:
                    multilabel = False
                    getSPE = True
                    getAuc = True
                    getMcc = True

                if database in ["su"]:
                    getSPE = True
                    getAuc = True

                csvWriter = None

                total_fold = 10
                if database in ["SIN-Locator", "laceDNN", "IHC"]:
                    total_fold = 5
                if split_num == -2:
                    total_fold = 1
                elif split_num == 5:
                    total_fold = len(organList)
                elif split_num == 6:
                    total_fold = len(brainList)
                for fold in range(total_fold):

                    path_prefix = dir_prefixs[database]
                    if split_num == -2:
                        result_prefix = "{}/results/{}/independent".format(cfg.DATA.RESULT_DIR, database)
                        train_file_path = "%s_train.csv" % (path_prefix)
                        val_file_path = "%s_test.csv" % (path_prefix)
                    elif split_num == -1:
                        result_prefix = "{}/results/{}/fold{}".format(cfg.DATA.RESULT_DIR, database, fold)
                        train_file_path = "%s_train_fold%d.csv" % (path_prefix, fold)
                        val_file_path = "%s_val_fold%d.csv" % (path_prefix, fold)
                        if not database in ["SIN-Locator"]:
                            test_file_path = "%s_test.csv" % (path_prefix)
                    else:
                        result_prefix = "{}/results/{}/split{}/fold{}".format(cfg.DATA.RESULT_DIR, database, split_num, fold)
                        train_file_path = "%s_train_split%d_fold%d.csv" % (path_prefix, split_num, fold)
                        val_file_path = "%s_val_split%d_fold%d.csv" % (path_prefix, split_num, fold)
                        if not database in ["SIN-Locator"]:
                            test_file_path = "%s_test.csv" % (path_prefix)
                            if database in ["GraphLoc", "laceDNN"]:
                                test_file_path = "%s_test_split%d.csv" % (path_prefix, split_num)
                    analysis_file_path = "%s_analysis.csv" % (path_prefix)
                    if database in ["IHC"]:
                        analysis_file_path = "%s_IHC_analysis.csv" % (path_prefix)
                    train_result_path = "{}/{}/preds/{}test_{}_aug0_{}".format(result_prefix, classifier_model, "proteinLevel/rank_" if proteinLevel else "", 't=0.5', train_file_path.split('/')[-1])
                    val_result_path = "{}/{}/preds/{}test_{}_aug0_{}".format(result_prefix, classifier_model, "proteinLevel/rank_" if proteinLevel else "", 't=0.5', val_file_path.split('/')[-1])
                    if not database in ["SIN-Locator"] and split_num != -2:
                        test_result_path = "{}/{}/preds/{}test_{}_aug0_{}".format(result_prefix, classifier_model, "proteinLevel/rank_" if proteinLevel else "", 't=0.5', test_file_path.split('/')[-1])

                    labels = labelLists.get(database, cfg.CLASSIFIER.LOCATIONS)

                    analysisData = pd.read_csv(analysis_file_path, header=0)
                    if split_num < 0:
                        analysisData = analysisData[(analysisData['Round'].isna()) & (analysisData['Split'] == 'Train')]
                    else:
                        analysisData = analysisData[(analysisData['Round'] == split_num) & (analysisData['Fold'] == fold) & (analysisData['Split'] == 'Train')]
                    location_num = analysisData[['image_' + i for i in labels]].values.flatten()
                    # if proteinLevel:
                    location_num = analysisData[['protein_' + i for i in labels]].values.flatten()
                    location_num = location_num.tolist()

                    # # for func in ["balanced9"]:
                    # # for func in ["balanced8", "balanced9", "balanced10", "balanced11", "balanced14", "balanced15"]:
                    # # for func in ["balanced16", "balanced17"]:
                    # for func in ["original"]:
                    #     # for thresh in [0.5]:
                    #     # for thresh in [0, 0.1, 0.2, 0.3 ,0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                    #     # for thresh in [0.9]:
                    #     # for thresh in [0.1, 0.2, 0.3 ,0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                    #     # for thresh in [0.5, 0.6, 0.7, 0.8]:
                    #     for thresh in [0.5, 0.6, 0.7, 0.8, 0.9]:
                    #     # for thresh in [0.5, 0.6, 0.7, 0.8, 0.9, 1]:
                    #     # for thresh in [1.2]:
                    #     # for thresh in [1.1, 1.2]:
                    #     # for thresh in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]:
                    #     # for thresh in [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]:
                    #         # train_resultData, thresh = get_balanced_result(train_result_path, labels, location_num, func=func, threshold=thresh, csvWriter=csvWriter, split_num=split_num, fold=fold, thresh='t=0.5', split='train')
                    #         # train_resultData.to_csv("{}/{}/preds/{}test_{}_{}_aug0_{}".format(result_prefix, classifier_model, "", func, "t={}".format(thresh), train_file_path.split('/')[-1]), index=True, mode='w')
                    #         # val_resultData, thresh = get_balanced_result(val_result_path, labels, location_num, func=func, threshold=thresh, csvWriter=csvWriter, split_num=split_num, fold=fold, thresh='t=0.5', split='val')
                    #         # val_resultData.to_csv("{}/{}/preds/{}test_{}_{}_aug0_{}".format(result_prefix, classifier_model, "", func, "t={}".format(thresh), val_file_path.split('/')[-1]), index=True, mode='w')
                    #         # if not database in ["SIN-Locator"] and split_num != -2:
                    #         #     test_resultData, thresh = get_balanced_result(test_result_path, labels, location_num, func=func, threshold=thresh, csvWriter=csvWriter, split_num=split_num, fold=fold, thresh='t=0.5', split='test')
                    #         #     test_resultData.to_csv("{}/{}/preds/{}test_{}_{}_aug0_{}".format(result_prefix, classifier_model, "", func, "t={}".format(thresh), test_file_path.split('/')[-1]), index=True, mode='w')
                    #         train_resultData, thresh = get_balanced_result(train_result_path, labels, location_num, func=func, threshold=thresh, csvWriter=csvWriter, split_num=split_num, fold=fold, thresh='t=0.5', split='train')
                    #         train_resultData.to_csv("{}/{}/preds/{}test_{}_{}_aug0_{}".format(result_prefix, classifier_model, "proteinLevel/rank_" if proteinLevel else "", func, "t={}".format(thresh), train_file_path.split('/')[-1]), index=True, mode='w')
                    #         val_resultData, thresh = get_balanced_result(val_result_path, labels, location_num, func=func, threshold=thresh, csvWriter=csvWriter, split_num=split_num, fold=fold, thresh='t=0.5', split='val')
                    #         val_resultData.to_csv("{}/{}/preds/{}test_{}_{}_aug0_{}".format(result_prefix, classifier_model, "proteinLevel/rank_" if proteinLevel else "", func, "t={}".format(thresh), val_file_path.split('/')[-1]), index=True, mode='w')
                    #         if not database in ["SIN-Locator"] and split_num != -2:
                    #             test_resultData, thresh = get_balanced_result(test_result_path, labels, location_num, func=func, threshold=thresh, csvWriter=csvWriter, split_num=split_num, fold=fold, thresh='t=0.5', split='test')
                    #             test_resultData.to_csv("{}/{}/preds/{}test_{}_{}_aug0_{}".format(result_prefix, classifier_model, "proteinLevel/rank_" if proteinLevel else "", func, "t={}".format(thresh), test_file_path.split('/')[-1]), index=True, mode='w')

                    for func in ["threshold"]:
                        for beta_n in [2, 1, 0.5, 0.25]:
                            beta = [beta_n for i in range(10)]
                            thresh = 0.5
                            train_resultData, thresh = get_balanced_result(train_result_path, labels, location_num, func=func, threshold=thresh, get_threshold=True, beta=beta, csvWriter=csvWriter, split_num=split_num, fold=fold, thresh='t=0.5', split='train')
                            train_resultData.to_csv("{}/{}/preds/{}test_{}_{}_aug0_{}".format(result_prefix, classifier_model, "proteinLevel/rank_" if proteinLevel else "", func, "f{}".format(beta_n), train_file_path.split('/')[-1]), index=True, mode='w')
                            val_resultData, threshold = get_balanced_result(val_result_path, labels, location_num, func=func, threshold=thresh, get_threshold=False, beta=beta, csvWriter=csvWriter, split_num=split_num, fold=fold, thresh='t=0.5', split='val')
                            val_resultData.to_csv("{}/{}/preds/{}test_{}_{}_aug0_{}".format(result_prefix, classifier_model, "proteinLevel/rank_" if proteinLevel else "", func, "f{}".format(beta_n), val_file_path.split('/')[-1]), index=True, mode='w')
                            if not database in ["SIN-Locator"] and split_num != -2:
                                test_resultData, threshold = get_balanced_result(test_result_path, labels, location_num, func=func, threshold=thresh, get_threshold=False, beta=beta, csvWriter=csvWriter, split_num=split_num, fold=fold, thresh='t=0.5', split='test')
                                test_resultData.to_csv("{}/{}/preds/{}test_{}_{}_aug0_{}".format(result_prefix, classifier_model, "proteinLevel/rank_" if proteinLevel else "", func, "f{}".format(beta_n), test_file_path.split('/')[-1]), index=True, mode='w')


        # # for func in ["balanced9"]:
        # # for func in ["balanced16", "balanced17"]:
        # # for func in ["balanced8", "balanced9", "balanced10", "balanced11", "balanced14", "balanced15"]:
        # # for func in ["balanced9_rank"]:
        # for func in ["original"]:
        #     # for thresh in [0.5]:
        #     # for thresh in [0, 0.1, 0.2, 0.3 ,0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        #     # for thresh in [0.9]:
        #     # for thresh in [0.1, 0.2, 0.3 ,0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        #     # for thresh in [0.5, 0.6, 0.7, 0.8]:
        #     # for thresh in [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]:
        #     # for thresh in [1.2]:
        #     # for thresh in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]:
        #     for thresh in [0.5, 0.6, 0.7, 0.8, 0.9]:
        #     # for thresh in [1.1, 1.2]:
        #     # for thresh in [0.5, 0.6, 0.7, 0.8, 0.9, 1]:
        #         thresh_func = "{}_t={}".format(func, thresh)
        #         thresh_func_prefix = "{}_".format(thresh_func)
        #         get_metrics(classifier_model=classifier_model, thresh_func=thresh_func, thresh_func_prefix=thresh_func_prefix)
        #         cal_mean_std(classifier_model=classifier_model, thresh_func=thresh_func, thresh_func_prefix=thresh_func_prefix)



        for func in ["threshold"]:
            for beta_n in [2, 1, 0.5, 0.25]:
                thresh_func = "{}_f{}".format(func, beta_n)
                thresh_func_prefix = "{}_".format(thresh_func)
                get_metrics(classifier_model=classifier_model, thresh_func=thresh_func, thresh_func_prefix=thresh_func_prefix)
                cal_mean_std(classifier_model=classifier_model, thresh_func=thresh_func, thresh_func_prefix=thresh_func_prefix)

        get_metrics(classifier_model=classifier_model)
        cal_mean_std(classifier_model=classifier_model)



if __name__ == "__main__":
    main()