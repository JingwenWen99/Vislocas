import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # __file__获取执行文件相对路径，整行为取上一级的上一级目录
sys.path.append(BASE_DIR)

import math
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
# from models.lightvit import lightvit_small, lightvit_tiny_modified, lightvit_modified
from models.classifier_model import getClassifier
from models.losses import get_loss_func
# from models.tClassifier1 import Classifier1
from models.train_autoencoder import load_best_model, test_layer
from models.train_classifier import load_best_classifier_model, test, val_epoch
from utils.args import parse_args
from utils.config_defaults import get_cfg, labelLists, dir_prefixs
from utils.optimizer import construct_optimizer, get_optimizer_func
from utils.scheduler import construct_scheduler, get_scheduler_func


# dir_prefixs = {"GraphLoc": "data/GraphLoc/GraphLoc",
#                 "MSTLoc": "data/MSTLoc/MSTLoc",
#                 "laceDNN": "data/laceDNN/laceDNN",
#                 "PScL-HDeep": "data/PScL-HDeep/PScL_HDeep",
#                 "PScL-2LSAESM": "data/PScL-2LSAESM/PScL_2LSAESM",
#                 "PScL-DDCFPred": "data/PScL-DDCFPred/PScL_DDCFPred",
#                 "su": "data/su/Su",
#                 "SIN-Locator": "data/SIN-Locator/SIN_Locator"}


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

    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # classifier_model = "cct_modified56_mlce_lr-000005_bn_drop-01_attn-drop-01_drop-path-01_batch12_seed6293_wd-005_aug_no-normalized"
    classifier_model = "cct_modified56_mlce_lr-000015_bn_drop-01_attn-drop-01_drop-path-01_batch12*5_seed6293_wd-005_aug_no-normalized"
    # classifier_model = "cct_modified56_mlce_lr-000005_bn_drop-03_attn-drop-0_drop-path-02_batch12_seed6293_wd-005_aug_no-normalized"

    for database in cfg.DATA.DATASET_NAME:
        if database in ["IHC"]:
            classifier_model = "cct_modified56_mlce_lr-000015_bn_drop-01_attn-drop-01_drop-path-01_batch12*5_seed6293_wd-005_aug_no-normalized"
        else:
            classifier_model = "cct_modified56_mlce_lr-000005_bn_drop-01_attn-drop-01_drop-path-01_batch12_seed6293_wd-005_aug_no-normalized"

        # for split_num in range(3, 5):
        # for split_num in [-1]:
        # split_num = 2
        split_list = [-1]
        if database in ["GraphLoc"]:
            split_list = [-1, 1, 2, 3, 4]
        elif database in ["laceDNN"]:
            split_list = range(5)
        for split_num in split_list:

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
                if split_num == -1:
                    file_name = "{}/results/{}/aug_{}_metrics.csv".format(cfg.DATA.RESULT_DIR, database, classifier_model)
                else:
                    file_name = "{}/results/{}/split{}/aug_{}_metrics.csv".format(cfg.DATA.RESULT_DIR, database, split_num, classifier_model)
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
            if database in ["SIN-Locator", "laceDNN"]:
                total_fold = 5
            for fold in range(total_fold):
            # for fold in range(10):

                if split_num == -1:
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


                # dataNums = pd.read_csv("%s_data_num.csv" % (path_prefix), header=0, index_col=0)
                # nums = dataNums.loc[fold].tolist()
                # location_num = nums[:-1]
                # data_num = nums[-1]

                # classifier_model = "cct_modified34_mlce"


                """ 读取数据 """
                train_loader = construct_loader(cfg, train_file_path, condition="normal", database=database, shuffle=False, aug=True, drop_last=False)
                val_loader = construct_loader(cfg, val_file_path, condition="normal", database=database, shuffle=False, aug=True, drop_last=False)
                if not database in ["SIN-Locator"]:
                    test_loader = construct_loader(cfg, test_file_path, condition="normal", database=database, shuffle=False, aug=True, drop_last=False)


                """ 构建SAE模型 """
                SAE = None
                if cfg.SAE.CONSTRUCT:
                    # SAE
                    layers_list = SAEmodel(cfg.SAE.MODEL_NAME)

                    for layer in range(len(layers_list)):
                        layers_list[layer].to(device)
                        if world_size > 1:
                            layers_list[layer] = nn.SyncBatchNorm.convert_sync_batchnorm(layers_list[layer])
                            layers_list[layer] = nn.parallel.DistributedDataParallel(layers_list[layer], device_ids=[torch.distributed.get_rank()], output_device=torch.distributed.get_rank())
                            layers_list[layer].module.is_training_layer = True
                        # else:
                        #     layers_list[layer].is_training_layer = False

                    load_best_model(cfg, layers_list, cfg.SAE.MODEL_NAME, device, result_prefix)
                    if world_size > 1:
                        dist.barrier()

                # for layer in range(len(layers_list)):
                #     test_layer(cfg, device, train_loader, criterion=get_loss_func("huber")(beta=cfg.SAE.BETA).to(device), layers_list=layers_list, layer=layer)
                #     test_layer(cfg, device, val_loader, criterion=get_loss_func("huber")(beta=cfg.SAE.BETA).to(device), layers_list=layers_list, layer=layer)
                #     test_layer(cfg, device, test_loader, criterion=get_loss_func("huber")(beta=cfg.SAE.BETA).to(device), layers_list=layers_list, layer=layer)

                    if world_size > 1:
                        for layer in range(len(layers_list)):
                            layers_list[layer] = layers_list[layer].cpu().module
                    torch.cuda.empty_cache()
                    SAE = StackedAutoEncoder(layers_list, False)

                """ 构建分类器模型 """
                # Classifier
                model = getClassifier(cfg, model_name=classifier_model, pretrain=False, SAE=SAE)

                if cfg.SAE.CONSTRUCT:
                    SAE = SAE.to(device)
                model = model.to(device)
                if world_size > 1:
                    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
                    model = nn.parallel.DistributedDataParallel(model, device_ids=[torch.distributed.get_rank()], output_device=torch.distributed.get_rank())

                # # # # best_path = "{}/{}/{}best_model.pth".format(result_prefix, model_name, prefix)
                # checkpoint_path = "{}/{}/latest_model.pth".format(result_prefix, classifier_model)
                # # # checkpoint_path = "{}/{}/best_model.pth".format(result_prefix, cfg.CLASSIFIER.MODEL_NAME)
                # prefix=""
                # # checkpoint_path = "{}/{}/finetune_latest_model.pth".format(result_prefix, cfg.CLASSIFIER.MODEL_NAME)
                # # # checkpoint_path = "{}/{}/finetune_best_model.pth".format(result_prefix, cfg.CLASSIFIER.MODEL_NAME)
                # # prefix="finetune_"
                # # print(checkpoint_path)
                # cu.load_checkpoint_test(checkpoint_path, model)
                load_best_classifier_model(cfg, model, classifier_model, device, result_prefix=result_prefix)
                prefix=""
                # load_best_classifier_model(cfg, model, classifier_model, device, prefix="finetune_", result_prefix=result_prefix)
                # prefix="finetune_"
                if world_size > 1:
                    dist.barrier()


                for aug in range(10):

                    if du.is_master_proc():
                        print("aug:", aug)

                    # weight = torch.tensor([math.sqrt((cfg.CLASSIFIER.DATA_NUM / x - 1) / 55) for x in cfg.CLASSIFIER.LOCATIONS_NUM]).to(device)
                    # criterion = get_loss_func("multilabel_categorical_cross_entropy")(reduction="none", weight=weight).to(device)
                    # criterion = get_loss_func("multilabel_categorical_cross_entropy")(reduction="none").to(device)

                    # weight = torch.tensor([pow(max(location_num) / x, 1 / 2) if x != 0 else 1e-8 for x in location_num]).to(device)
                    # pos_weight = torch.tensor([pow(data_num / x - 1, 1 / 2) if x != 0 else 1 for x in location_num]).to(device)
                    # criterion = get_loss_func("multilabel_categorical_cross_entropy")(reduction="none", weight=weight, pos_weight=pos_weight).to(device)
                    criterion = get_loss_func("multilabel_balanced_cross_entropy")(reduction="none", scale=0).to(device)

                    # val_epoch(cfg, device, train_loader, train_file_path, SAE, model, criterion=criterion, l1_alpha=cfg.CLASSIFIER.L1_ALPHA, l2_alpha=cfg.CLASSIFIER.L2_ALPHA, cur_epoch=-1, epoch=-1, model_name="lightViT_small", writer=None, metricsWriter=None, result_prefix=result_prefix, prefix=prefix)
                    # _, optimal_thres = test(cfg, device, train_loader, cfg.DATA.TRAIN_FILE, SAE, model, criterion=criterion, model_name=cfg.CLASSIFIER.MODEL_NAME, threshold=None, get_threshold=True)
                    _, optimal_thres = test(cfg, device, train_loader, train_file_path, SAE, model, criterion=criterion, model_name=classifier_model, multilabel=multilabel, threshold=None, get_threshold=False,
                        result_prefix=result_prefix, log_prefix=log_prefix, prefix=prefix, csvWriter=csvWriter, fold=fold, aug=aug, thresh='t=0.5', split='train', getSPE=getSPE, getAuc=getAuc, getMcc=getMcc)
                    if du.is_master_proc():
                        print("Optimal Threshold: ", optimal_thres)
                    # test(cfg, device, val_loader, cfg.DATA.VAL_FILE, SAE, model, criterion=criterion, model_name=cfg.CLASSIFIER.MODEL_NAME, threshold=optimal_thres, get_threshold=False)
                    test(cfg, device, val_loader, val_file_path, SAE, model, criterion=criterion, model_name=classifier_model, multilabel=multilabel, threshold=optimal_thres, get_threshold=False,
                        result_prefix=result_prefix, log_prefix=log_prefix, prefix=prefix, csvWriter=csvWriter, fold=fold, aug=aug, thresh='t=0.5', split='val', getSPE=getSPE, getAuc=getAuc, getMcc=getMcc)
                    if not database in ["SIN-Locator"]:
                        test(cfg, device, test_loader, test_file_path, SAE, model, criterion=criterion, model_name=classifier_model, multilabel=multilabel, threshold=optimal_thres, get_threshold=False,
                            result_prefix=result_prefix, log_prefix=log_prefix, prefix=prefix, csvWriter=csvWriter, fold=fold, aug=aug, thresh='t=0.5', split='test', getSPE=getSPE, getAuc=getAuc, getMcc=getMcc)


                    # _, optimal_thres = test(cfg, device, train_loader, train_file_path, SAE, model, criterion=criterion, model_name=classifier_model, multilabel=multilabel, threshold=None, get_threshold=True,
                    #     result_prefix=result_prefix, log_prefix=log_prefix, prefix=prefix, beta=[1 for i in range(10)], csvWriter=csvWriter, fold=fold, thresh='threshold(f1)', split='train', getSPE=getSPE, getAuc=getAuc, getMcc=getMcc)
                    # if du.is_master_proc():
                    #     print("Optimal Threshold: ", optimal_thres)
                    # # test(cfg, device, val_loader, cfg.DATA.VAL_FILE, SAE, model, criterion=criterion, model_name=cfg.CLASSIFIER.MODEL_NAME, threshold=optimal_thres, get_threshold=False)
                    # test(cfg, device, val_loader, val_file_path, SAE, model, criterion=criterion, model_name=classifier_model, multilabel=multilabel, threshold=optimal_thres, get_threshold=False,
                    #     result_prefix=result_prefix, log_prefix=log_prefix, prefix=prefix, beta=[1 for i in range(10)], csvWriter=csvWriter, fold=fold, thresh='threshold(f1)', split='val', getSPE=getSPE, getAuc=getAuc, getMcc=getMcc)
                    # if not database in ["SIN-Locator"]:
                    #     test(cfg, device, test_loader, test_file_path, SAE, model, criterion=criterion, model_name=classifier_model, multilabel=multilabel, threshold=optimal_thres, get_threshold=False,
                    #         result_prefix=result_prefix, log_prefix=log_prefix, prefix=prefix, beta=[1 for i in range(10)], csvWriter=csvWriter, fold=fold, thresh='threshold(f1)', split='test', getSPE=getSPE, getAuc=getAuc, getMcc=getMcc)


                    # _, optimal_thres = test(cfg, device, train_loader, train_file_path, SAE, model, criterion=criterion, model_name=classifier_model, multilabel=multilabel, threshold=None, get_threshold=True,
                    #     result_prefix=result_prefix, log_prefix=log_prefix, prefix=prefix, beta=[0.5 for i in range(10)], csvWriter=csvWriter, fold=fold, thresh='threshold(f0.5)', split='train', getSPE=getSPE, getAuc=getAuc, getMcc=getMcc)
                    # if du.is_master_proc():
                    #     print("Optimal Threshold: ", optimal_thres)
                    # # test(cfg, device, val_loader, cfg.DATA.VAL_FILE, SAE, model, criterion=criterion, model_name=cfg.CLASSIFIER.MODEL_NAME, threshold=optimal_thres, get_threshold=False)
                    # test(cfg, device, val_loader, val_file_path, SAE, model, criterion=criterion, model_name=classifier_model, multilabel=multilabel, threshold=optimal_thres, get_threshold=False,
                    #     result_prefix=result_prefix, log_prefix=log_prefix, prefix=prefix, beta=[0.5 for i in range(10)], csvWriter=csvWriter, fold=fold, thresh='threshold(f0.5)', split='val', getSPE=getSPE, getAuc=getAuc, getMcc=getMcc)
                    # if not database in ["SIN-Locator"]:
                    #     test(cfg, device, test_loader, test_file_path, SAE, model, criterion=criterion, model_name=classifier_model, multilabel=multilabel, threshold=optimal_thres, get_threshold=False,
                    #         result_prefix=result_prefix, log_prefix=log_prefix, prefix=prefix, beta=[0.5 for i in range(10)], csvWriter=csvWriter, fold=fold, thresh='threshold(f0.5)', split='test', getSPE=getSPE, getAuc=getAuc, getMcc=getMcc)


                    # _, optimal_thres = test(cfg, device, train_loader, train_file_path, SAE, model, criterion=criterion, model_name=classifier_model, multilabel=multilabel, threshold=None, get_threshold=True,
                    #     result_prefix=result_prefix, log_prefix=log_prefix, prefix=prefix, beta=[0.25 for i in range(10)], csvWriter=csvWriter, fold=fold, thresh='threshold(f0.25)', split='train', getSPE=getSPE, getAuc=getAuc, getMcc=getMcc)
                    # if du.is_master_proc():
                    #     print("Optimal Threshold: ", optimal_thres)
                    # # test(cfg, device, val_loader, cfg.DATA.VAL_FILE, SAE, model, criterion=criterion, model_name=cfg.CLASSIFIER.MODEL_NAME, threshold=optimal_thres, get_threshold=False)
                    # test(cfg, device, val_loader, val_file_path, SAE, model, criterion=criterion, model_name=classifier_model, multilabel=multilabel, threshold=optimal_thres, get_threshold=False,
                    #     result_prefix=result_prefix, log_prefix=log_prefix, prefix=prefix, beta=[0.25 for i in range(10)], csvWriter=csvWriter, fold=fold, thresh='threshold(f0.25)', split='val', getSPE=getSPE, getAuc=getAuc, getMcc=getMcc)
                    # if not database in ["SIN-Locator"]:
                    #     test(cfg, device, test_loader, test_file_path, SAE, model, criterion=criterion, model_name=classifier_model, multilabel=multilabel, threshold=optimal_thres, get_threshold=False,
                    #         result_prefix=result_prefix, log_prefix=log_prefix, prefix=prefix, beta=[0.25 for i in range(10)], csvWriter=csvWriter, fold=fold, thresh='threshold(f0.25)', split='test', getSPE=getSPE, getAuc=getAuc, getMcc=getMcc)



                    # _, optimal_thres = test(cfg, device, train_loader, cfg.DATA.TRAIN_FILE, SAE, model, criterion=criterion, model_name=cfg.CLASSIFIER.MODEL_NAME, threshold=None, get_threshold=False)
                    # print("Optimal Threshold: ", optimal_thres)
                    # test(cfg, device, val_loader, cfg.DATA.VAL_FILE, SAE, model, criterion=criterion, model_name=cfg.CLASSIFIER.MODEL_NAME, threshold=None, get_threshold=False)
            if du.is_master_proc():
                f.close()

if __name__ == "__main__":
    main()