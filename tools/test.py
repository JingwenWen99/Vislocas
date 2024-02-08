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
from models.classifier_model import getClassifier
from models.losses import get_loss_func
from models.train_classifier import load_best_classifier_model, test, val_epoch
from utils.args import parse_args
from utils.config_defaults import get_cfg, labelLists, dir_prefixs
from utils.optimizer import construct_optimizer, get_optimizer_func
from utils.scheduler import construct_scheduler, get_scheduler_func


organList = ["Brain", "Connective & soft tissue", "Female tissues", "Gastrointestinal tract", "Kidney & urinary bladder", "Skin"]
brainList = ["Caudate", "Cerebellum", "Cerebral cortex", "Hippocampus"]


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


    classifier_model = "cct_modified72_mlce_lr-000005_bn_drop-01_attn-drop-01_drop-path-01_batch12_seed6293_wd-005_aug_no-normalized"

    for database in cfg.DATA.DATASET_NAME:
        split_list = range(5)
        if database in ["IHC"]:
            split_list = [-2, 0, 1, 2, 3, 4, 5]
        elif database in ["MSTLoc"]:
            split_list = [-2]
        for split_num in split_list:

            # Set random seed from configs.
            random.seed(cfg.RNG_SEED + args.local_rank)
            np.random.seed(cfg.RNG_SEED + args.local_rank)
            torch.manual_seed(cfg.RNG_SEED + args.local_rank)
            torch.cuda.manual_seed(cfg.RNG_SEED + args.local_rank)
            torch.cuda.manual_seed_all(cfg.RNG_SEED + args.local_rank)

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
            if du.is_master_proc():
                if split_num == -2:
                    file_name = "{}/results/{}/independent/{}_metrics.csv".format(cfg.DATA.RESULT_DIR, database, classifier_model)
                elif split_num == -1:
                    file_name = "{}/results/{}/{}_metrics.csv".format(cfg.DATA.RESULT_DIR, database, classifier_model)
                else:
                    file_name = "{}/results/{}/split{}/{}_metrics.csv".format(cfg.DATA.RESULT_DIR, database, split_num, classifier_model)
                f = open(file_name, "w", encoding="utf-8", newline="")
                csvWriter = csv.writer(f)
                annotationHeader = ['fold', 'threshold', 'split',
                        'example_subset_accuracy', 'example_accuracy', 'example_precision', 'example_recall', 'example_f1', 'example_hamming_loss',
                        'label_accuracy_macro', 'label_precision_macro', 'label_recall_macro', 'label_f1_macro', 'label_jaccard_macro',
                        'label_accuracy_micro', 'label_precision_micro', 'label_recall_micro', 'label_f1_micro', 'label_jaccard_micro']
                if getSPE:
                    annotationHeader += ['label_specificity_macro', 'label_specificity_micro']
                if getAuc:
                    annotationHeader += ['auc', 'meanAUC', 'stdAUC']
                if getMcc:
                    annotationHeader += ['MCC']
                labels = labelLists.get(database, cfg.CLASSIFIER.LOCATIONS)
                for loc in labels:
                    annotationHeader += [loc + _ for _ in ['_accuracy', '_precision', '_recall', '_f1', '_jaccard']]
                csvWriter.writerow(annotationHeader)


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
                    if du.is_master_proc():
                        print(log_prefix)

                    train_file_path = "%s_train.csv" % (path_prefix)
                    val_file_path = "%s_test.csv" % (path_prefix)
                elif split_num == -1:
                    path_prefix = dir_prefixs[database]
                    result_prefix = "{}/results/{}/fold{}".format(cfg.DATA.RESULT_DIR, database, fold)
                    log_prefix = "{}/fold{}".format(database, fold)
                    if du.is_master_proc():
                        print(log_prefix)

                    train_file_path = "%s_train_fold%d.csv" % (path_prefix, fold)
                    val_file_path = "%s_val_fold%d.csv" % (path_prefix, fold)
                    if not database in ["SIN-Locator"]:
                        test_file_path = "%s_test.csv" % (path_prefix)
                else:
                    path_prefix = dir_prefixs[database]
                    result_prefix = "{}/results/{}/split{}/fold{}".format(cfg.DATA.RESULT_DIR, database, split_num, fold)
                    log_prefix = "{}/split{}/fold{}".format(database, split_num, fold)
                    if du.is_master_proc():
                        print(log_prefix)

                    train_file_path = "%s_train_split%d_fold%d.csv" % (path_prefix, split_num, fold)
                    val_file_path = "%s_val_split%d_fold%d.csv" % (path_prefix, split_num, fold)
                    if not database in ["SIN-Locator"]:
                        test_file_path = "%s_test.csv" % (path_prefix)
                        if database in ["GraphLoc", "laceDNN"]:
                            test_file_path = "%s_test_split%d.csv" % (path_prefix, split_num)
                analysis_file_path = "%s_analysis.csv" % (path_prefix)
                if database in ["IHC"]:
                    analysis_file_path = "%s_IHC_analysis.csv" % (path_prefix)


                """ 读取数据 """
                train_loader = construct_loader(cfg, train_file_path, condition="normal", database=database, shuffle=True, drop_last=False)
                val_loader = construct_loader(cfg, val_file_path, condition="normal", database=database, shuffle=False, drop_last=False)
                if not database in ["SIN-Locator"] and split_num != -2:
                    test_loader = construct_loader(cfg, test_file_path, condition="normal", database=database, shuffle=False, drop_last=False)



                """ 构建分类器模型 """
                # Classifier
                model = getClassifier(cfg, model_name=classifier_model, pretrain=False)


                model = model.to(device)
                if world_size > 1:
                    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
                    model = nn.parallel.DistributedDataParallel(model, device_ids=[torch.distributed.get_rank()], output_device=torch.distributed.get_rank())

                load_best_classifier_model(cfg, model, classifier_model, device, result_prefix=result_prefix)
                prefix=""
                if world_size > 1:
                    dist.barrier()


                nums = None
                totalNums = 0
                analysisData = pd.read_csv(analysis_file_path, header=0)
                if split_num < 0:
                    analysisData = analysisData[(analysisData['Round'].isna()) & (analysisData['Split'] == 'Train')]
                else:
                    analysisData = analysisData[(analysisData['Round'] == split_num) & (analysisData['Fold'] == fold) & (analysisData['Split'] == 'Train')]


                if "mlce" in classifier_model:
                    criterion = get_loss_func("multilabel_balanced_cross_entropy")(reduction="none", nums=nums, total_nums=totalNums).to(device)
                elif "focalloss" in classifier_model:
                    criterion = get_loss_func("focal_loss")(reduction="none").to(device)
                elif "weightedbce" in classifier_model:
                    location_num = analysisData[['image_' + i for i in cfg.CLASSIFIER.LOCATIONS]].values.flatten()
                    location_num = location_num.tolist()
                    weight = torch.tensor([math.floor(max(location_num) / x) if x != 0 else 0 for x in location_num]).to(device)
                    criterion = get_loss_func("bce_logit")(reduction="none", weight=weight).to(device)
                elif "bce" in classifier_model:
                    criterion = get_loss_func("bce_logit")(reduction="none").to(device)
                else:
                    criterion = get_loss_func("multilabel_balanced_cross_entropy")(reduction="none", nums=nums, total_nums=totalNums).to(device)


                _, optimal_thres = test(cfg, device, train_loader, train_file_path, None, model, criterion=criterion, model_name=classifier_model, multilabel=multilabel, threshold=None, get_threshold=False,
                    result_prefix=result_prefix, log_prefix=log_prefix, prefix=prefix, csvWriter=csvWriter, randomSplit=split_num, fold=fold, thresh='t=0.5', split='train', getSPE=getSPE, getAuc=getAuc, getMcc=getMcc)
                if du.is_master_proc():
                    print("Optimal Threshold: ", optimal_thres)
                test(cfg, device, val_loader, val_file_path, None, model, criterion=criterion, model_name=classifier_model, multilabel=multilabel, threshold=optimal_thres, get_threshold=False,
                    result_prefix=result_prefix, log_prefix=log_prefix, prefix=prefix, csvWriter=csvWriter, randomSplit=split_num, fold=fold, thresh='t=0.5', split='val', getSPE=getSPE, getAuc=getAuc, getMcc=getMcc)
                if not database in ["SIN-Locator"] and split_num != -2:
                    test(cfg, device, test_loader, test_file_path, None, model, criterion=criterion, model_name=classifier_model, multilabel=multilabel, threshold=optimal_thres, get_threshold=False,
                        result_prefix=result_prefix, log_prefix=log_prefix, prefix=prefix, csvWriter=csvWriter, randomSplit=split_num, fold=fold, thresh='t=0.5', split='test', getSPE=getSPE, getAuc=getAuc, getMcc=getMcc)


            if du.is_master_proc():
                f.close()


if __name__ == "__main__":
    main()