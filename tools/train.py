import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # __file__ a get the relative path of the executable file, the whole line is taken to the previous level of the previous directory
sys.path.append(BASE_DIR)
# print(sys.path)

import math
import random

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
from models.train_classifier import load_best_classifier_model, train, pretrain, load_best_pretrain_classifier_model
from utils.args import parse_args
from utils.config_defaults import get_cfg, dir_prefixs
from utils.optimizer import construct_optimizer, get_optimizer_func
from utils.scheduler import construct_scheduler, get_scheduler_func


organList = ["Brain", "Connective & soft tissue", "Female tissues", "Gastrointestinal tract", "Kidney & urinary bladder", "Skin"]
brainList = ["Caudate", "Cerebellum", "Cerebral cortex", "Hippocampus"]


def main():
    """
    Main function to spawn the train and test process.
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


    for classifier_model in ["Vislocas_mlce_lr-000005_bn_drop-01_attn-drop-01_drop-path-01_batch12_seed6293_wd-005_aug_no-normalized"]:
        head_layer='classifier.fc'

        for database in cfg.DATA.DATASET_NAME:
            multilabel = True

            split_list = range(5)
            if database in ["IHC"]:
                split_list = [-2, 0, 1, 2, 3, 4]
            elif database in ["MSTLoc"]:
                split_list = [-2]
            for split_num in split_list:

                # Set random seed from configs.
                random.seed(cfg.RNG_SEED + args.local_rank)
                np.random.seed(cfg.RNG_SEED + args.local_rank)
                torch.manual_seed(cfg.RNG_SEED + args.local_rank)
                torch.cuda.manual_seed(cfg.RNG_SEED + args.local_rank)
                torch.cuda.manual_seed_all(cfg.RNG_SEED + args.local_rank)

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
                        if du.is_master_proc():
                            print(log_prefix)

                        train_file_path = "%s_train.csv" % (path_prefix)
                        val_file_path = "%s_test.csv" % (path_prefix)
                        adj_file_path = "%s_adj_train_fold%d.csv" % (path_prefix, fold)
                    elif split_num == -1:
                        path_prefix = dir_prefixs[database]
                        result_prefix = "{}/results/{}/fold{}".format(cfg.DATA.RESULT_DIR, database, fold)
                        log_prefix = "{}/fold{}".format(database, fold)
                        if du.is_master_proc():
                            print(log_prefix)

                        train_file_path = "%s_train_fold%d.csv" % (path_prefix, fold)
                        val_file_path = "%s_val_fold%d.csv" % (path_prefix, fold)
                        adj_file_path = "%s_adj_train_fold%d.csv" % (path_prefix, fold)
                    else:
                        path_prefix = dir_prefixs[database]
                        result_prefix = "{}/results/{}/split{}/fold{}".format(cfg.DATA.RESULT_DIR, database, split_num, fold)
                        log_prefix = "{}/split{}/fold{}".format(database, split_num, fold)
                        if du.is_master_proc():
                            print(log_prefix)

                        train_file_path = "%s_train_split%d_fold%d.csv" % (path_prefix, split_num, fold)
                        val_file_path = "%s_val_split%d_fold%d.csv" % (path_prefix, split_num, fold)
                        adj_file_path = "%s_adj_train_fold%d.csv" % (path_prefix, fold)
                    analysis_file_path = "%s_analysis.csv" % (path_prefix)
                    if database in ["IHC"]:
                        analysis_file_path = "%s_IHC_analysis.csv" % (path_prefix)


                    """ load data """
                    train_loader = construct_loader(cfg, train_file_path, condition="normal", database=database, aug=True, shuffle=True, drop_last=False)
                    val_loader = construct_loader(cfg, val_file_path, condition="normal", database=database, shuffle=False, drop_last=False)


                    """ Constructing classifier model """
                    # Classifier
                    model = getClassifier(cfg, model_name=classifier_model, pretrain=False, adj_file=adj_file_path)

                    model = model.to(device)
                    if world_size > 1:
                        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
                        model = nn.parallel.DistributedDataParallel(model, device_ids=[torch.distributed.get_rank()], output_device=torch.distributed.get_rank())



                    """ Training the classifier model """
                    if cfg.CLASSIFIER.TRAIN:
                        if du.is_master_proc():
                            print("Train classifier {}".format(classifier_model))

                        """ Constructing optimizier """
                        if "wd-01" in classifier_model:
                            optimizer = get_optimizer_func("adamw")(model.parameters(), lr=cfg.CLASSIFIER.HEAD_BASE_LR, weight_decay=0.1, amsgrad=False)
                        elif "wd-005" in classifier_model:
                            optimizer = get_optimizer_func("adamw")(model.parameters(), lr=cfg.CLASSIFIER.HEAD_BASE_LR, weight_decay=0.05, amsgrad=False)
                        elif "wd-002" in classifier_model:
                            optimizer = get_optimizer_func("adamw")(model.parameters(), lr=cfg.CLASSIFIER.HEAD_BASE_LR, weight_decay=0.02, amsgrad=False)
                        elif "wd-0025" in classifier_model:
                            optimizer = get_optimizer_func("adamw")(model.parameters(), lr=cfg.CLASSIFIER.HEAD_BASE_LR, weight_decay=0.025, amsgrad=False)
                        elif "wd-001" in classifier_model:
                            optimizer = get_optimizer_func("adamw")(model.parameters(), lr=cfg.CLASSIFIER.HEAD_BASE_LR, weight_decay=0.01, amsgrad=False)
                        elif "wd-0005" in classifier_model:
                            optimizer = get_optimizer_func("adamw")(model.parameters(), lr=cfg.CLASSIFIER.HEAD_BASE_LR, weight_decay=0.005, amsgrad=False)
                        elif "wd-0001" in classifier_model:
                            optimizer = get_optimizer_func("adamw")(model.parameters(), lr=cfg.CLASSIFIER.HEAD_BASE_LR, weight_decay=0.001, amsgrad=False)
                        elif "wd-00001" in classifier_model:
                            optimizer = get_optimizer_func("adamw")(model.parameters(), lr=cfg.CLASSIFIER.HEAD_BASE_LR, weight_decay=0.0001, amsgrad=False)
                        else:
                            optimizer = get_optimizer_func("adamw")(model.parameters(), lr=cfg.CLASSIFIER.HEAD_BASE_LR, weight_decay=0, amsgrad=False)

                        """ Constructing scheduler """
                        scheduler = construct_scheduler(cfg.CLASSIFIER, optimizer, "warmupCosine")
                        scaler = torch.cuda.amp.GradScaler(enabled=True)


                        """ Loss function """
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


                        start_epoch = 0
                        min_loss = float("inf")
                        if cfg.CLASSIFIER.CKP:
                            checkpoint_path = "{}/{}/latest_model.pth".format(result_prefix, classifier_model)
                            start_epoch, min_loss = cu.load_checkpoint(checkpoint_path, model, optimizer, scheduler, scaler)

                        dist.barrier()

                        train(cfg, None, model, loader=[train_loader, val_loader],
                            optimizer=optimizer, scheduler=scheduler, scaler=scaler, criterion=criterion, l1_alpha=cfg.CLASSIFIER.L1_ALPHA, l2_alpha=cfg.CLASSIFIER.L2_ALPHA,
                            epoch=cfg.CLASSIFIER.EPOCH_NUM, start_epoch=start_epoch, min_loss=min_loss, model_name=classifier_model, patch_size=cfg.SAE.MODEL_NAME, multilabel=multilabel,
                            result_prefix=result_prefix, log_prefix=log_prefix, train_file_path=train_file_path, val_file_path=val_file_path,
                            device=device
                        )

                    load_best_classifier_model(cfg, model, classifier_model, device, result_prefix=result_prefix)

                    if du.is_master_proc():
                        print(model)


if __name__ == "__main__":
    main()