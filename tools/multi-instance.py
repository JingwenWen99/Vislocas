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


def get_protein_level_result(resultPath, labels, func="max", csvWriter=None, split_num=-1, fold=0, thresh='', split=''):
    resultData = pd.read_csv(resultPath, header=0, index_col=0)
    locations_pred = [i + '_pred' for i in labels]
    locations_pred_labels = [i + '_pred_labels' for i in labels]


    resultData['Label Num'] = resultData[locations_pred_labels].sum(axis=1)
    groupData = resultData.groupby(['Protein Name', 'Protein Id'])

    print(groupData)

    proteinLabel = groupData[labels].max()

    if func == "max":
        predProteinResult = groupData[locations_pred].mean()
        predProteinLabel = groupData[locations_pred_labels].max()
    elif func == "avg":
        predProteinResult = groupData[locations_pred].mean()
        predProteinLabel = groupData[locations_pred_labels].mean()
        predProteinLabel[predProteinLabel < 0.5] = 0
        predProteinLabel[predProteinLabel >= 0.5] = 1
    elif func == "pred_avg":
        predProteinLabel = groupData[locations_pred_labels].mean()
        predProteinResult = groupData[locations_pred].mean()
        predProteinResult.columns = locations_pred_labels
        predProteinLabel[predProteinResult < 0.5] = 0
        predProteinLabel[predProteinResult >= 0.5] = 1
    elif func == "integrate":
        ratio = 0.5
        predProteinLabel = groupData[locations_pred_labels].mean()
        predProteinResult = groupData[locations_pred].mean()
        predProteinResult.columns = locations_pred_labels
        predProteinLabel0 = predProteinLabel
        predProteinLabel[ratio * predProteinLabel0 + (1 - ratio) * predProteinResult < 0.5] = 0
        predProteinLabel[ratio * predProteinLabel0 + (1 - ratio) * predProteinResult >= 0.5] = 1
    elif func == "integrate025":
        ratio = 0.25
        predProteinLabel = groupData[locations_pred_labels].mean()
        predProteinResult = groupData[locations_pred].mean()
        predProteinResult.columns = locations_pred_labels
        predProteinLabel0 = predProteinLabel
        predProteinLabel[ratio * predProteinLabel0 + (1 - ratio) * predProteinResult < 0.5] = 0
        predProteinLabel[ratio * predProteinLabel0 + (1 - ratio) * predProteinResult >= 0.5] = 1
    elif func == "integrate075":
        ratio = 0.75
        predProteinLabel = groupData[locations_pred_labels].mean()
        predProteinResult = groupData[locations_pred].mean()
        predProteinResult.columns = locations_pred_labels
        predProteinLabel0 = predProteinLabel
        predProteinLabel[ratio * predProteinLabel0 + (1 - ratio) * predProteinResult < 0.5] = 0
        predProteinLabel[ratio * predProteinLabel0 + (1 - ratio) * predProteinResult >= 0.5] = 1
    elif func == "rank":
        predProteinResult = groupData[locations_pred].mean()
        predProteinLabel = predProteinResult.copy()
        predProteinLabel.columns = locations_pred_labels
        predProteinLabelNum = groupData['Label Num'].max()
        ranked = predProteinLabel.rank(axis=1, ascending=False)
        for loc in locations_pred_labels:
            predProteinLabel.loc[ranked[loc] <= predProteinLabelNum, loc] = 1
            predProteinLabel.loc[ranked[loc] > predProteinLabelNum, loc] = 0
    else:
        predProteinLabel = groupData[locations_pred_labels].max()

    predData = pd.merge(proteinLabel, predProteinResult, how='left', left_index=True, right_index=True)
    predData = pd.merge(predData, predProteinLabel, how='left', left_index=True, right_index=True)
    predData.columns = predData.columns.tolist()
    predData.reset_index(inplace=True)

    splitPath = resultPath.rsplit('/', 1)
    if not os.path.exists("{}/proteinLevel".format(splitPath[0])):
        os.makedirs("{}/proteinLevel".format(splitPath[0]))
    predData.to_csv("{}/proteinLevel/{}_{}".format(splitPath[0], func, splitPath[1]), index=True, mode='w')


    proteinLabel = np.array(proteinLabel)
    predProteinLabel = np.array(predProteinLabel)

    cal_metrics(None, proteinLabel, predProteinLabel, None, -1, locations=labels, csvWriter=csvWriter, randomSplit=split_num, fold=fold, thresh="{}_{}".format(thresh, func), split=split)


if __name__ == '__main__':
    """
    Main function to spawn the test process.
    """
    args = parse_args()
    cfg = get_cfg()

    # classifier_model = "cct_modified72_mlce_lr-000005_bn_drop-01_attn-drop-01_drop-path-01_batch12_seed6293_wd-005_aug_no-normalized"
    classifier_model = "cct_modified56_mlce_lr-000005_bn_drop-01_attn-drop-01_drop-path-01_batch12_seed6293_wd-005_aug_no-normalized"

    # for database in ["laceDNN", "GraphLoc"]:
    for database in cfg.DATA.DATASET_NAME:
        split_list = range(5)
        if database in ["IHC"]:
            split_list = [-2] + list(range(5))
            # split_list = [-2] + list(range(17))
        for split_num in split_list:

            if split_num == -2:
                file_path = "{}/results/{}/independent/proteinLevel_{}_metrics.csv".format(cfg.DATA.RESULT_DIR, database, classifier_model)
            else:
                file_path = "{}/results/{}/split{}/proteinLevel_{}_metrics.csv".format(cfg.DATA.RESULT_DIR, database, split_num, classifier_model)
            f = open(file_path, "w", encoding="utf-8", newline="")
            csvWriter = csv.writer(f)
            annotationHeader = ['fold', 'threshold', 'split',
                    'example_subset_accuracy', 'example_accuracy', 'example_precision', 'example_recall', 'example_f1', 'example_hamming_loss',
                    'label_accuracy_macro', 'label_precision_macro', 'label_recall_macro', 'label_f1_macro', 'label_jaccard_macro',
                    'label_accuracy_micro', 'label_precision_micro', 'label_recall_micro', 'label_f1_micro', 'label_jaccard_micro']
            labels = labelLists.get(database, cfg.CLASSIFIER.LOCATIONS)
            for loc in labels:
                annotationHeader += [loc + _ for _ in ['_accuracy', '_precision', '_recall', '_f1', '_jaccard']]
            csvWriter.writerow(annotationHeader)

            total_fold = 10
            if database in ["laceDNN", "IHC"]:
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
                    if not database in ["SIN-Locator"]:
                        test_file_path = "%s_test.csv" % (path_prefix)
                else:
                    path_prefix = dir_prefixs[database]
                    result_prefix = "{}/results/{}/split{}/fold{}".format(cfg.DATA.RESULT_DIR, database, split_num, fold)
                    log_prefix = "{}/split{}/fold{}".format(database, split_num, fold)
                    print(log_prefix)

                    train_file_path = "%s_train_split%d_fold%d.csv" % (path_prefix, split_num, fold)
                    val_file_path = "%s_val_split%d_fold%d.csv" % (path_prefix, split_num, fold)
                    if not database in ["SIN-Locator"]:
                        test_file_path = "%s_test.csv" % (path_prefix)
                        if database in ["GraphLoc", "laceDNN"]:
                            test_file_path = "%s_test_split%d.csv" % (path_prefix, split_num)


                # for func in ["rank", "integrate"]:
                for func in ["rank"]:
                    resultPath = "{}/{}/preds/{}test_{}_aug0_{}".format(result_prefix, classifier_model, "", 't=0.5', train_file_path.split('/')[-1])
                    get_protein_level_result(resultPath, labels, func=func, csvWriter=csvWriter, split_num=split_num, fold=fold, thresh='t=0.5', split='train')
                    resultPath = "{}/{}/preds/{}test_{}_aug0_{}".format(result_prefix, classifier_model, "", 't=0.5', val_file_path.split('/')[-1])
                    get_protein_level_result(resultPath, labels, func=func, csvWriter=csvWriter, split_num=split_num, fold=fold, thresh='t=0.5', split='val')
                    if not database in ["SIN-Locator"] and split_num != -2:
                        resultPath = "{}/{}/preds/{}test_{}_aug0_{}".format(result_prefix, classifier_model, "", 't=0.5', test_file_path.split('/')[-1])
                        get_protein_level_result(resultPath, labels, func=func, csvWriter=csvWriter, split_num=split_num, fold=fold, thresh='t=0.5', split='test')
                    # resultPath = "{}/{}/preds/{}test_{}_aug0_{}".format(result_prefix, classifier_model, "", 'balanced9_t=0.9', train_file_path.split('/')[-1])
                    # get_protein_level_result(resultPath, labels, func=func, csvWriter=csvWriter, split_num=split_num, fold=fold, thresh='t=0.9', split='train')
                    # resultPath = "{}/{}/preds/{}test_{}_aug0_{}".format(result_prefix, classifier_model, "", 'balanced9_t=0.9', val_file_path.split('/')[-1])
                    # get_protein_level_result(resultPath, labels, func=func, csvWriter=csvWriter, split_num=split_num, fold=fold, thresh='t=0.9', split='val')
                    # resultPath = "{}/{}/preds/{}test_{}_aug0_{}".format(result_prefix, classifier_model, "", 'balanced9_t=0.9', test_file_path.split('/')[-1])
                    # get_protein_level_result(resultPath, labels, func=func, csvWriter=csvWriter, split_num=split_num, fold=fold, thresh='t=0.9', split='test')


                # # for func in ["rank", "integrate"]:
                # for func in ["rank"]:
                #     for t in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]:
                #     # resultPath = "{}/{}/preds/{}test_{}_aug0_{}".format(result_prefix, classifier_model, "", 't=0.5', train_file_path.split('/')[-1])
                #     # get_protein_level_result(resultPath, labels, func=func, csvWriter=csvWriter, split_num=split_num, fold=fold, thresh='t=0.5', split='train')
                #     # resultPath = "{}/{}/preds/{}test_{}_aug0_{}".format(result_prefix, classifier_model, "", 't=0.5', val_file_path.split('/')[-1])
                #     # get_protein_level_result(resultPath, labels, func=func, csvWriter=csvWriter, split_num=split_num, fold=fold, thresh='t=0.5', split='val')
                #     # resultPath = "{}/{}/preds/{}test_{}_aug0_{}".format(result_prefix, classifier_model, "", 't=0.5', test_file_path.split('/')[-1])
                #     # get_protein_level_result(resultPath, labels, func=func, csvWriter=csvWriter, split_num=split_num, fold=fold, thresh='t=0.5', split='test')
                #         resultPath = "{}/{}/preds/{}test_{}_aug0_{}".format(result_prefix, classifier_model, "", "balanced8_t={}".format(t), train_file_path.split('/')[-1])
                #         get_protein_level_result(resultPath, labels, func=func, csvWriter=csvWriter, split_num=split_num, fold=fold, thresh="t={}".format(t), split='train')
                #         resultPath = "{}/{}/preds/{}test_{}_aug0_{}".format(result_prefix, classifier_model, "", "balanced8_t={}".format(t), val_file_path.split('/')[-1])
                #         get_protein_level_result(resultPath, labels, func=func, csvWriter=csvWriter, split_num=split_num, fold=fold, thresh="t={}".format(t), split='val')
                #         resultPath = "{}/{}/preds/{}test_{}_aug0_{}".format(result_prefix, classifier_model, "", "balanced8_t={}".format(t), test_file_path.split('/')[-1])
                #         get_protein_level_result(resultPath, labels, func=func, csvWriter=csvWriter, split_num=split_num, fold=fold, thresh="t={}".format(t), split='test')




            f.close()
