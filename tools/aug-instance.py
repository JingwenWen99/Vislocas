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


# dir_prefixs = {"GraphLoc": "data/GraphLoc/GraphLoc",
#                 "MSTLoc": "data/MSTLoc/MSTLoc",
#                 "laceDNN": "data/laceDNN/laceDNN",
#                 "PScL-HDeep": "data/PScL-HDeep/PScL_HDeep",
#                 "PScL-2LSAESM": "data/PScL-2LSAESM/PScL_2LSAESM",
#                 "PScL-DDCFPred": "data/PScL-DDCFPred/PScL_DDCFPred",
#                 "su": "data/su/Su",
#                 "SIN-Locator": "data/SIN-Locator/SIN_Locator"}


def get_augment_result(resultData, savePath, labels, func="max", csvWriter=None, fold=0, thresh='', split=''):
    resultData = pd.read_csv(resultPath, header=0, index_col=0)
    locations_pred = [i + '_pred' for i in labels]
    locations_pred_labels = [i + '_pred_labels' for i in labels]

    groupData = resultData.groupby(['index'])

    print(groupData)

    imageLabel = groupData[labels].max()

    if func == "max":
        predLabel = groupData[locations_pred_labels].max()
    elif func == "avg":
        predLabel = groupData[locations_pred_labels].mean()
        predLabel[predLabel < 0.5] = 0
        predLabel[predLabel >= 0.5] = 1
    elif func == "pred_avg":
        predLabel = groupData[locations_pred_labels].mean()
        predResult = groupData[locations_pred].mean()
        predResult.columns = locations_pred_labels
        predLabel[predResult < 0.5] = 0
        predLabel[predResult >= 0.5] = 1
    elif func == "integrate":
        ratio = 0.5
        predLabel = groupData[locations_pred_labels].mean()
        predResult = groupData[locations_pred].mean()
        predResult.columns = locations_pred_labels
        predLabel0 = predLabel
        predLabel[ratio * predLabel0 + (1 - ratio) * predResult < 0.5] = 0
        predLabel[ratio * predLabel0 + (1 - ratio) * predResult >= 0.5] = 1
    elif func == "integrate025":
        ratio = 0.25
        predLabel = groupData[locations_pred_labels].mean()
        predResult = groupData[locations_pred].mean()
        predResult.columns = locations_pred_labels
        predLabel0 = predLabel
        predLabel[ratio * predLabel0 + (1 - ratio) * predResult < 0.5] = 0
        predLabel[ratio * predLabel0 + (1 - ratio) * predResult >= 0.5] = 1
    elif func == "integrate075":
        ratio = 0.75
        predLabel = groupData[locations_pred_labels].mean()
        predResult = groupData[locations_pred].mean()
        predResult.columns = locations_pred_labels
        predLabel0 = predLabel
        predLabel[ratio * predLabel0 + (1 - ratio) * predResult < 0.5] = 0
        predLabel[ratio * predLabel0 + (1 - ratio) * predResult >= 0.5] = 1
    else:
        predLabel = groupData[locations_pred_labels].max()

    predData = pd.merge(imageLabel, predLabel, how='left', left_index=True, right_index=True)
    predData.columns = predData.columns.tolist()
    predData.reset_index(inplace=True)

    splitPath = resultPath.rsplit('/', 1)
    if not os.path.exists("{}/augment".format(splitPath[0])):
        os.makedirs("{}/augment".format(splitPath[0]))
    predData.to_csv("{}/augment/{}_{}".format(splitPath[0], func, splitPath[1]), index=True, mode='w')


    proteinLabel = np.array(proteinLabel)
    predProteinLabel = np.array(predProteinLabel)

    cal_metrics(None, proteinLabel, predProteinLabel, None, -1, locations=labels, csvWriter=csvWriter, fold=fold, thresh="{}_{}".format(thresh, func), split=split)


if __name__ == '__main__':
    """
    Main function to spawn the test process.
    """
    args = parse_args()
    cfg = get_cfg()

    classifier_model = "cct_modified56_mlce_lr-000005_bn_drop-01_attn-drop-01_drop-path-01_batch12_seed6293_wd-005_aug_no-normalized"

    for database in ["IHC"]:
    # for database in cfg.DATA.DATASET_NAME:
        if database in ["IHC"]:
            classifier_model = "cct_modified56_mlce_lr-000015_bn_drop-01_attn-drop-01_drop-path-01_batch12*5_seed6293_wd-005_aug_no-normalized"
        else:
            classifier_model = "cct_modified56_mlce_lr-000005_bn_drop-01_attn-drop-01_drop-path-01_batch12_seed6293_wd-005_aug_no-normalized"

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

        split_list = [-1]
        if database in ["GraphLoc"]:
            split_list = [-1, 1, 2, 3, 4]
        elif database in ["laceDNN"]:
            split_list = range(5)
        for split_num in split_list:

            if split_num == -1:
                file_name = "{}/results/{}/augment_{}_metrics.csv".format(cfg.DATA.RESULT_DIR, database, classifier_model)
            else:
                file_name = "{}/results/{}/split{}/augment_{}_metrics.csv".format(cfg.DATA.RESULT_DIR, database, split_num, classifier_model)

            f = open("{}/results/{}/split{}/augment_{}_metrics.csv".format(cfg.DATA.RESULT_DIR, database, split_num, classifier_model), "w", encoding="utf-8", newline="")
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

            for fold in range(10):
                if split_num == -1:
                    path_prefix = dir_prefixs[database]
                    result_prefix = "{}/results/{}/fold{}".format(cfg.DATA.RESULT_DIR, database, fold)
                    log_prefix = "{}/fold{}".format(database, fold)
                    print(log_prefix)

                    train_file_path = "data_train_fold%d.csv" % (fold)
                    val_file_path = "data_val_fold%d.csv" % (fold)
                    file_list = [train_file_path, val_file_path]
                    if not database in ["SIN-Locator"]:
                        test_file_path = "data_test.csv"
                        file_list.append(test_file_path)
                else:
                    path_prefix = dir_prefixs[database]
                    result_prefix = "{}/results/{}/split{}/fold{}".format(cfg.DATA.RESULT_DIR, database, split_num, fold)
                    log_prefix = "{}/split{}/fold{}".format(database, split_num, fold)
                    print(log_prefix)

                    train_file_path = "data_train_split%d_fold%d.csv" % (split_num, fold)
                    val_file_path = "data_val_split%d_fold%d.csv" % (split_num, fold)
                    file_list = [train_file_path, val_file_path]
                    if not database in ["SIN-Locator"]:
                        test_file_path = "data_test.csv" % (path_prefix)
                        file_list.append(test_file_path)

                for files in file_list:
                    datas = []
                    for aug in range(10):
                        file_name = "{}/{}/preds/test_t=0.5_aug{}_{}".format(result_prefix, classifier_model, aug, files)
                        resultData = pd.read_csv(file_name, header=0, index_col=0)
                        # resultData = resultData.reset_index()
                        datas.append(resultData)
                    datas = pd.concat(datas)
                print(datas)
                resultData = resultData.reset_index()

                for func in ["max", "avg", "pred_avg", "integrate", "integrate025", "integrate075"]:
                    resultPath = "{}/{}/preds/test_t=0.5_augment_{}".format(result_prefix, classifier_model, files)


                    get_protein_level_result(resultPath, labels, func=func, csvWriter=csvWriter, fold=fold, thresh='t=0.5', split='train')
                    resultPath = "{}/{}/preds/{}test_{}_{}".format(result_prefix, classifier_model, "", 't=0.5', val_file_path.split('/')[-1])
                    get_protein_level_result(resultPath, labels, func=func, csvWriter=csvWriter, fold=fold, thresh='t=0.5', split='val')
                    resultPath = "{}/{}/preds/{}test_{}_{}".format(result_prefix, classifier_model, "", 't=0.5', test_file_path.split('/')[-1])
                    get_protein_level_result(resultPath, labels, func=func, csvWriter=csvWriter, fold=fold, thresh='t=0.5', split='test')

            f.close()

"{}/{}/preds/test_t=0.5_aug{}_{}".format(result_prefix, classifier_model, aug, files)



# resultPath, labels, func="max", csvWriter=None, fold=0, thresh='', split=''


        #     predData.to_csv("{}/{}/preds/{}test_{}_{}".format(result_prefix, model_name, prefix, thresh, data_file.split('/')[-1]), index=True, mode='w')

        #     path_prefix = dir_prefixs[database]
        #     result_prefix = "{}/results/{}/fold{}".format(cfg.DATA.RESULT_DIR, database, fold)
        #     log_prefix = "{}/fold{}".format(database, fold)
        #     if du.is_master_proc():
        #         print(log_prefix)

        #     train_file_path = "%s_train_fold%d.csv" % (path_prefix, fold)
        #     val_file_path = "%s_val_fold%d.csv" % (path_prefix, fold)
        #     test_file_path = "%s_test.csv" % (path_prefix)

        #     dataNums = pd.read_csv("%s_data_num.csv" % (path_prefix), header=0, index_col=0)
        #     nums = dataNums.loc[fold].tolist()
        #     location_num = nums[:-1]
        #     data_num = nums[-1]

        #     # classifier_model = "cct_modified34_mlce"


        #     """ 读取数据 """
        #     train_loader = construct_loader(cfg, train_file_path, condition="normal", database=database, shuffle=True, drop_last=False)
        #     val_loader = construct_loader(cfg, val_file_path, condition="normal", database=database, shuffle=False, drop_last=False)
        #     test_loader = construct_loader(cfg, test_file_path, condition="normal", database=database, shuffle=False, drop_last=False)


        #     """ 构建SAE模型 """
        #     SAE = None
        #     if cfg.SAE.CONSTRUCT:
        #         # SAE
        #         layers_list = SAEmodel(cfg.SAE.MODEL_NAME)

        #         for layer in range(len(layers_list)):
        #             layers_list[layer].to(device)
        #             if world_size > 1:
        #                 layers_list[layer] = nn.SyncBatchNorm.convert_sync_batchnorm(layers_list[layer])
        #                 layers_list[layer] = nn.parallel.DistributedDataParallel(layers_list[layer], device_ids=[torch.distributed.get_rank()], output_device=torch.distributed.get_rank())
        #                 layers_list[layer].module.is_training_layer = True
        #             # else:
        #             #     layers_list[layer].is_training_layer = False

        #         load_best_model(cfg, layers_list, cfg.SAE.MODEL_NAME, device, result_prefix)
        #         if world_size > 1:
        #             dist.barrier()

        #     # for layer in range(len(layers_list)):
        #     #     test_layer(cfg, device, train_loader, criterion=get_loss_func("huber")(beta=cfg.SAE.BETA).to(device), layers_list=layers_list, layer=layer)
        #     #     test_layer(cfg, device, val_loader, criterion=get_loss_func("huber")(beta=cfg.SAE.BETA).to(device), layers_list=layers_list, layer=layer)
        #     #     test_layer(cfg, device, test_loader, criterion=get_loss_func("huber")(beta=cfg.SAE.BETA).to(device), layers_list=layers_list, layer=layer)

        #         if world_size > 1:
        #             for layer in range(len(layers_list)):
        #                 layers_list[layer] = layers_list[layer].cpu().module
        #         torch.cuda.empty_cache()
        #         SAE = StackedAutoEncoder(layers_list, False)

        #     """ 构建分类器模型 """
        #     # Classifier
        #     model = getClassifier(cfg, model_name=classifier_model, pretrain=False, SAE=SAE)

        #     if cfg.SAE.CONSTRUCT:
        #         SAE = SAE.to(device)
        #     model = model.to(device)
        #     if world_size > 1:
        #         model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        #         model = nn.parallel.DistributedDataParallel(model, device_ids=[torch.distributed.get_rank()], output_device=torch.distributed.get_rank())

        #     # # # # best_path = "{}/{}/{}best_model.pth".format(result_prefix, model_name, prefix)
        #     # checkpoint_path = "{}/{}/latest_model.pth".format(result_prefix, classifier_model)
        #     # # # checkpoint_path = "{}/{}/best_model.pth".format(result_prefix, cfg.CLASSIFIER.MODEL_NAME)
        #     # prefix=""
        #     # # checkpoint_path = "{}/{}/finetune_latest_model.pth".format(result_prefix, cfg.CLASSIFIER.MODEL_NAME)
        #     # # # checkpoint_path = "{}/{}/finetune_best_model.pth".format(result_prefix, cfg.CLASSIFIER.MODEL_NAME)
        #     # # prefix="finetune_"
        #     # # print(checkpoint_path)
        #     # cu.load_checkpoint_test(checkpoint_path, model)
        #     load_best_classifier_model(cfg, model, classifier_model, device, result_prefix=result_prefix)
        #     prefix=""
        #     # load_best_classifier_model(cfg, model, classifier_model, device, prefix="finetune_", result_prefix=result_prefix)
        #     # prefix="finetune_"
        #     if world_size > 1:
        #         dist.barrier()

        #     # weight = torch.tensor([math.sqrt((cfg.CLASSIFIER.DATA_NUM / x - 1) / 55) for x in cfg.CLASSIFIER.LOCATIONS_NUM]).to(device)
        #     # criterion = get_loss_func("multilabel_categorical_cross_entropy")(reduction="none", weight=weight).to(device)
        #     # criterion = get_loss_func("multilabel_categorical_cross_entropy")(reduction="none").to(device)

        #     # weight = torch.tensor([pow(max(location_num) / x, 1 / 2) if x != 0 else 1e-8 for x in location_num]).to(device)
        #     # pos_weight = torch.tensor([pow(data_num / x - 1, 1 / 2) if x != 0 else 1 for x in location_num]).to(device)
        #     # criterion = get_loss_func("multilabel_categorical_cross_entropy")(reduction="none", weight=weight, pos_weight=pos_weight).to(device)
        #     criterion = get_loss_func("multilabel_balanced_cross_entropy")(reduction="none", scale=0).to(device)

        #     # val_epoch(cfg, device, train_loader, train_file_path, SAE, model, criterion=criterion, l1_alpha=cfg.CLASSIFIER.L1_ALPHA, l2_alpha=cfg.CLASSIFIER.L2_ALPHA, cur_epoch=-1, epoch=-1, model_name="lightViT_small", writer=None, metricsWriter=None, result_prefix=result_prefix, prefix=prefix)
        #     # _, optimal_thres = test(cfg, device, train_loader, cfg.DATA.TRAIN_FILE, SAE, model, criterion=criterion, model_name=cfg.CLASSIFIER.MODEL_NAME, threshold=None, get_threshold=True)
        #     _, optimal_thres = test(cfg, device, train_loader, train_file_path, SAE, model, criterion=criterion, model_name=classifier_model, threshold=None, get_threshold=False,
        #         result_prefix=result_prefix, log_prefix=log_prefix, prefix=prefix, csvWriter=csvWriter, fold=fold, thresh='t=0.5', split='train')
        #     if du.is_master_proc():
        #         print("Optimal Threshold: ", optimal_thres)
        #     # test(cfg, device, val_loader, cfg.DATA.VAL_FILE, SAE, model, criterion=criterion, model_name=cfg.CLASSIFIER.MODEL_NAME, threshold=optimal_thres, get_threshold=False)
        #     test(cfg, device, val_loader, val_file_path, SAE, model, criterion=criterion, model_name=classifier_model, threshold=optimal_thres, get_threshold=False,
        #         result_prefix=result_prefix, log_prefix=log_prefix, prefix=prefix, csvWriter=csvWriter, fold=fold, thresh='t=0.5', split='val')
        #     test(cfg, device, test_loader, test_file_path, SAE, model, criterion=criterion, model_name=classifier_model, threshold=optimal_thres, get_threshold=False,
        #         result_prefix=result_prefix, log_prefix=log_prefix, prefix=prefix, csvWriter=csvWriter, fold=fold, thresh='t=0.5', split='test')


        #     _, optimal_thres = test(cfg, device, train_loader, train_file_path, SAE, model, criterion=criterion, model_name=classifier_model, threshold=None, get_threshold=True,
        #         result_prefix=result_prefix, log_prefix=log_prefix, prefix=prefix, beta=[1 for i in range(10)], csvWriter=csvWriter, fold=fold, thresh='threshold(f1)', split='train')
        #     if du.is_master_proc():
        #         print("Optimal Threshold: ", optimal_thres)
        #     # test(cfg, device, val_loader, cfg.DATA.VAL_FILE, SAE, model, criterion=criterion, model_name=cfg.CLASSIFIER.MODEL_NAME, threshold=optimal_thres, get_threshold=False)
        #     test(cfg, device, val_loader, val_file_path, SAE, model, criterion=criterion, model_name=classifier_model, threshold=optimal_thres, get_threshold=False,
        #         result_prefix=result_prefix, log_prefix=log_prefix, prefix=prefix, beta=[1 for i in range(10)], csvWriter=csvWriter, fold=fold, thresh='threshold(f1)', split='val')
        #     test(cfg, device, test_loader, test_file_path, SAE, model, criterion=criterion, model_name=classifier_model, threshold=optimal_thres, get_threshold=False,
        #         result_prefix=result_prefix, log_prefix=log_prefix, prefix=prefix, beta=[1 for i in range(10)], csvWriter=csvWriter, fold=fold, thresh='threshold(f1)', split='test')


        #     _, optimal_thres = test(cfg, device, train_loader, train_file_path, SAE, model, criterion=criterion, model_name=classifier_model, threshold=None, get_threshold=True,
        #         result_prefix=result_prefix, log_prefix=log_prefix, prefix=prefix, beta=[0.5 for i in range(10)], csvWriter=csvWriter, fold=fold, thresh='threshold(f0.5)', split='train')
        #     if du.is_master_proc():
        #         print("Optimal Threshold: ", optimal_thres)
        #     # test(cfg, device, val_loader, cfg.DATA.VAL_FILE, SAE, model, criterion=criterion, model_name=cfg.CLASSIFIER.MODEL_NAME, threshold=optimal_thres, get_threshold=False)
        #     test(cfg, device, val_loader, val_file_path, SAE, model, criterion=criterion, model_name=classifier_model, threshold=optimal_thres, get_threshold=False,
        #         result_prefix=result_prefix, log_prefix=log_prefix, prefix=prefix, beta=[0.5 for i in range(10)], csvWriter=csvWriter, fold=fold, thresh='threshold(f0.5)', split='val')
        #     test(cfg, device, test_loader, test_file_path, SAE, model, criterion=criterion, model_name=classifier_model, threshold=optimal_thres, get_threshold=False,
        #         result_prefix=result_prefix, log_prefix=log_prefix, prefix=prefix, beta=[0.5 for i in range(10)], csvWriter=csvWriter, fold=fold, thresh='threshold(f0.5)', split='test')


        #     _, optimal_thres = test(cfg, device, train_loader, train_file_path, SAE, model, criterion=criterion, model_name=classifier_model, threshold=None, get_threshold=True,
        #         result_prefix=result_prefix, log_prefix=log_prefix, prefix=prefix, beta=[0.25 for i in range(10)], csvWriter=csvWriter, fold=fold, thresh='threshold(f0.25)', split='train')
        #     if du.is_master_proc():
        #         print("Optimal Threshold: ", optimal_thres)
        #     # test(cfg, device, val_loader, cfg.DATA.VAL_FILE, SAE, model, criterion=criterion, model_name=cfg.CLASSIFIER.MODEL_NAME, threshold=optimal_thres, get_threshold=False)
        #     test(cfg, device, val_loader, val_file_path, SAE, model, criterion=criterion, model_name=classifier_model, threshold=optimal_thres, get_threshold=False,
        #         result_prefix=result_prefix, log_prefix=log_prefix, prefix=prefix, beta=[0.25 for i in range(10)], csvWriter=csvWriter, fold=fold, thresh='threshold(f0.25)', split='val')
        #     test(cfg, device, test_loader, test_file_path, SAE, model, criterion=criterion, model_name=classifier_model, threshold=optimal_thres, get_threshold=False,
        #         result_prefix=result_prefix, log_prefix=log_prefix, prefix=prefix, beta=[0.25 for i in range(10)], csvWriter=csvWriter, fold=fold, thresh='threshold(f0.25)', split='test')



        #     # _, optimal_thres = test(cfg, device, train_loader, cfg.DATA.TRAIN_FILE, SAE, model, criterion=criterion, model_name=cfg.CLASSIFIER.MODEL_NAME, threshold=None, get_threshold=False)
        #     # print("Optimal Threshold: ", optimal_thres)
        #     # test(cfg, device, val_loader, cfg.DATA.VAL_FILE, SAE, model, criterion=criterion, model_name=cfg.CLASSIFIER.MODEL_NAME, threshold=None, get_threshold=False)
        # if du.is_master_proc():
        #     f.close()
