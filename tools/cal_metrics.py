import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # __file__获取执行文件相对路径，整行为取上一级的上一级目录
sys.path.append(BASE_DIR)
# print(sys.path)

import csv

import numpy as np
import pandas as pd

from utils.args import parse_args
from utils.config_defaults import get_cfg, labelLists, dir_prefixs
from utils.eval_metrics import cal_metrics


organList = ["Brain", "Connective & soft tissue", "Female tissues", "Gastrointestinal tract", "Kidney & urinary bladder", "Skin"]
brainList = ["Caudate", "Cerebellum", "Cerebral cortex", "Hippocampus"]


# classifier_model = "cct_modified72_weightedbce_lr-000005_bn_drop-01_attn-drop-01_drop-path-01_batch12_seed6293_wd-005_aug_no-normalized"
# classifier_model = "cct_modified72_focalloss_lr-000005_bn_drop-01_attn-drop-01_drop-path-01_batch12_seed6293_wd-005_aug_no-normalized"
# classifier_model = "cct_modified72_mlce_lr-000005_bn_drop-01_attn-drop-01_drop-path-01_batch12_seed6293_wd-005_aug_no-normalized"
# classifier_model = "cct_modified70_mlce_lr-000005_bn_drop-01_attn-drop-01_drop-path-01_batch12_seed6293_wd-005_aug_no-normalized"


def get_metrics(classifier_model="cct_modified72_mlce_lr-000005_bn_drop-01_attn-drop-01_drop-path-01_batch12_seed6293_wd-005_aug_no-normalized", thresh_func="t=0.5", thresh_func_prefix=""):
    args = parse_args()
    cfg = get_cfg()


    # for database in ["GraphLoc", "laceDNN"]:
    for database in cfg.DATA.DATASET_NAME:
        proteinLevel = False
        if database in ["GraphLoc", "laceDNN"]:
            proteinLevel = True
        # if database in ["IHC"]:
        #     proteinLevel = True

        multilabel = True
        getSPE = False
        getAuc = False
        getMcc = False

        if database in ["PScL-HDeep", "PScL-2LSAESM", "PScL-DDCFPred"]:
            # multilabel = False
            getSPE = True
            getAuc = True
            getMcc = True

        if database in ["su"]:
            getSPE = True
            getAuc = True

        f_path = "{}/results/{}/{}crossValidation_{}_metrics.csv".format(cfg.DATA.RESULT_DIR, database, thresh_func_prefix, classifier_model)
        if proteinLevel:
            f_path = "{}/results/{}/{}proteinLevel_crossValidation_{}_metrics.csv".format(cfg.DATA.RESULT_DIR, database, thresh_func_prefix, classifier_model)
        f = open(f_path, "w", encoding="utf-8", newline="")
        csvWriter = csv.writer(f)
        annotationHeader = ['randomSplit', 'fold', 'threshold', 'split']
        annotationHeader += ['total_TP', 'total_FP', 'total_TN', 'total_FN']
        labels = labelLists.get(database, cfg.CLASSIFIER.LOCATIONS)
        for loc in labels:
            annotationHeader += [loc + _ for _ in ['_TP', '_FP', '_TN', '_FN']]
        annotationHeader += ['example_subset_accuracy', 'example_accuracy', 'example_precision', 'example_recall', 'example_f1', 'example_hamming_loss',
                'label_accuracy_macro', 'label_precision_macro', 'label_recall_macro', 'label_f1_macro', 'label_jaccard_macro',
                'label_accuracy_micro', 'label_precision_micro', 'label_recall_micro', 'label_f1_micro', 'label_jaccard_micro']
        if getSPE:
            annotationHeader += ['label_specificity_macro', 'label_specificity_micro']
        if getAuc:
            annotationHeader += ['auc', 'meanAUC', 'stdAUC']
        if getMcc:
            annotationHeader += ['MCC']
        for loc in labels:
            annotationHeader += [loc + _ for _ in ['_accuracy', '_precision', '_recall', '_f1', '_jaccard']]
        csvWriter.writerow(annotationHeader)

        locations_pred = [i + '_pred' for i in labels]
        locations_pred_labels = [i + '_pred_labels' for i in labels]

        split_list = range(5)
        if database in ["IHC"]:
            split_list = [-2] + list(range(5))
            # split_list = [-2] + list(range(17))
        elif database in ["MSTLoc"]:
            split_list = [-2] + list(range(5))
        for split_num in split_list:

            all_res = []

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

                if split_num == -2:
                    path_prefix = dir_prefixs[database]
                    result_prefix = "{}/results/{}/independent".format(cfg.DATA.RESULT_DIR, database)
                    log_prefix = "{}/independent".format(database)
                    print(log_prefix)

                    train_file_path = "%s_train.csv" % (path_prefix)
                    val_file_path = "%s_test.csv" % (path_prefix)
                elif split_num == -1:
                    path_prefix = dir_prefixs[database]
                    result_prefix = "{}/results/{}/fold{}".format(cfg.DATA.RESULT_DIR, database, fold)
                    log_prefix = "{}/fold{}".format(database, fold)
                    print(log_prefix)

                    train_file_path = "%s_train_fold%d.csv" % (path_prefix, fold)
                    val_file_path = "%s_val_fold%d.csv" % (path_prefix, fold)
                    file_list = [train_file_path, val_file_path]
                    if not database in ["SIN-Locator"]:
                        test_file_path = "%s_test.csv" % (path_prefix)
                        file_list.append(test_file_path)
                else:
                    path_prefix = dir_prefixs[database]
                    result_prefix = "{}/results/{}/split{}/fold{}".format(cfg.DATA.RESULT_DIR, database, split_num, fold)
                    log_prefix = "{}/split{}/fold{}".format(database, split_num, fold)
                    print(log_prefix)

                    train_file_path = "%s_train_split%d_fold%d.csv" % (path_prefix, split_num, fold)
                    val_file_path = "%s_val_split%d_fold%d.csv" % (path_prefix, split_num, fold)
                    file_list = [train_file_path, val_file_path]
                    if not database in ["SIN-Locator"]:
                        test_file_path = "%s_test.csv" % (path_prefix)
                        if database in ["GraphLoc", "laceDNN"]:
                            test_file_path = "%s_test_split%d.csv" % (path_prefix, split_num)
                        file_list.append(test_file_path)

                val_path = "{}/{}/preds/test_{}_aug0_{}".format(result_prefix, classifier_model, thresh_func, val_file_path.split('/')[-1])
                if proteinLevel:
                    val_path = "{}/{}/preds/proteinLevel/rank_test_{}_aug0_{}".format(result_prefix, classifier_model, thresh_func, val_file_path.split('/')[-1])
                val_res = pd.read_csv(val_path, header=0, index_col=0)
                all_res.append(val_res)
                if not database in ["SIN-Locator"] and split_num != -2:
                    test_path = "{}/{}/preds/test_{}_aug0_{}".format(result_prefix, classifier_model, thresh_func, test_file_path.split('/')[-1])
                    if proteinLevel:
                        test_path = "{}/{}/preds/proteinLevel/rank_test_{}_aug0_{}".format(result_prefix, classifier_model, thresh_func, test_file_path.split('/')[-1])
                    test_res = pd.read_csv(test_path, header=0, index_col=0)
                    all_labels = test_res[labels]
                    all_pred_labels = test_res[locations_pred_labels]
                    all_labels = np.array(all_labels)
                    all_pred_labels = np.array(all_pred_labels)
                    split_str = 'test'
                    if split_num in range(7, 12):
                        split_str = 'reSampled1_test'
                    elif split_num in range(12, 17):
                        split_str = 'reSampled2_test'
                    cal_metrics(cfg, all_labels, all_pred_labels, None, -1, locations=labels, csvWriter=csvWriter, randomSplit=split_num, fold=fold, thresh='t=0.5', split=split_str, getQuantity=True, getSPE=getSPE, getAuc=getAuc, getMcc=getMcc)


                if split_num in [5, 6]:
                    val_labels = val_res[labels]
                    val_pred_labels = val_res[locations_pred_labels]
                    val_labels = np.array(val_labels)
                    val_pred_labels = np.array(val_pred_labels)
                    split_str = organList[fold]
                    if split_num == 6:
                        split_str = brainList[fold]
                    cal_metrics(cfg, val_labels, val_pred_labels, None, -1, locations=labels, csvWriter=csvWriter, randomSplit=split_num, fold=fold, thresh='t=0.5', split=split_str, getQuantity=True, getSPE=getSPE, getAuc=getAuc, getMcc=getMcc)


            all_res = pd.concat(all_res, axis=0)
            if split_num != -2:
                result_prefix = "{}/results/{}/split{}".format(cfg.DATA.RESULT_DIR, database, split_num)
            save_path = "{}/crossValidation_{}_{}_split{}.csv".format(result_prefix, classifier_model, thresh_func, split_num)
            if proteinLevel:
                save_path = "{}/proteinLevel_crossValidation_{}_split{}.csv".format(result_prefix, thresh_func, split_num)
            all_res.to_csv(save_path, index=True, mode='w')
            all_labels = all_res[labels]
            all_pred_labels = all_res[locations_pred_labels]
            all_labels = np.array(all_labels)
            all_pred_labels = np.array(all_pred_labels)
            split_str = 'crossValidation'
            if split_num == -2:
                split_str = 'independentTest'
            elif split_num == 5:
                split_str = 'organ_crossValidation'
            elif split_num == 6:
                split_str = 'brain_crossValidation'
            elif split_num in range(7, 12):
                split_str = 'reSampled1_crossValidation'
            elif split_num in range(12, 17):
                split_str = 'reSampled2_crossValidation'
            cal_metrics(cfg, all_labels, all_pred_labels, None, -1, locations=labels, csvWriter=csvWriter, randomSplit=split_num, fold=None, thresh='t=0.5', split=split_str, getQuantity=True, getSPE=getSPE, getAuc=getAuc, getMcc=getMcc)

        f.close()


def cal_mean_std(classifier_model="cct_modified72_mlce_lr-000005_bn_drop-01_attn-drop-01_drop-path-01_batch12_seed6293_wd-005_aug_no-normalized", thresh_func="t=0.5", thresh_func_prefix=""):
    args = parse_args()
    cfg = get_cfg()


    # for database in ["GraphLoc", "laceDNN"]:
    for database in cfg.DATA.DATASET_NAME:
        proteinLevel = False
        if database in ["GraphLoc", "laceDNN"]:
            proteinLevel = True
        # if database in ["IHC"]:
        #     proteinLevel = True
        metricsPath = "{}/results/{}/{}crossValidation_{}_metrics.csv".format(cfg.DATA.RESULT_DIR, database, thresh_func_prefix, classifier_model)
        if proteinLevel:
            metricsPath = "{}/results/{}/{}proteinLevel_crossValidation_{}_metrics.csv".format(cfg.DATA.RESULT_DIR, database, thresh_func_prefix, classifier_model)
        metricsData = pd.read_csv(metricsPath, header=0)
        metricsData.insert(loc=4, column='statistic', value=np.nan)
        print(metricsData)

        statisticData = metricsData.groupby(['split']).agg([np.mean, np.std]).stack().reset_index()
        statisticData['statistic'] = statisticData['level_1']
        statisticData = statisticData.drop(['level_1'], axis=1)
        statisticData[['randomSplit', 'fold']] = np.nan
        print(statisticData)
        metricsData = metricsData.append(statisticData, ignore_index = True)
        print(metricsData)

        metricsSavePath = "{}/results/{}/statistic_{}crossValidation_{}_metrics.csv".format(cfg.DATA.RESULT_DIR, database, thresh_func_prefix, classifier_model)
        if proteinLevel:
            metricsSavePath = "{}/results/{}/statistic_{}proteinLevel_crossValidation_{}_metrics.csv".format(cfg.DATA.RESULT_DIR, database, thresh_func_prefix, classifier_model)
        metricsData.to_csv(metricsSavePath, index=None, mode='w')


if __name__ == '__main__':
    """
    Main function to spawn the test process.
    """
    classifier_model="cct_modified78_mlce_lr-000005_bn_drop-01_attn-drop-01_drop-path-01_batch12_seed6293_wd-005_aug_no-normalized"
    get_metrics(classifier_model=classifier_model)
    cal_mean_std(classifier_model=classifier_model)