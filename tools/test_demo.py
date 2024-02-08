import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # __file__获取执行文件相对路径，整行为取上一级的上一级目录
sys.path.append(BASE_DIR)
# print(sys.path)

import math

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as transforms_tv

import utils.distributed as du
import utils.checkpoint as cu
from datasets.loader import construct_loader
from datasets.build import build_dataset
from models.autoencoder import AutoEncoderLayer, StackedAutoEncoder, SAEmodel
from models.lightvit import lightvit_small
from models.losses import get_loss_func
from models.tClassifier1 import Classifier1
from models.train_autoencoder import load_best_model, train_layer, test_layer
from models.train_classifier import load_best_classifier_model, train
# from mvit.config.defaults import assert_and_infer_cfg, get_cfg
# from utils.args import load_config, parse_args
from utils.optimizer import construct_optimizer, get_optimizer_func
from utils.scheduler import construct_scheduler, get_scheduler_func
from utils.args import parse_args
from utils.config_defaults import get_cfg


def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    cfg = get_cfg()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)


    idx = 128
    writer = SummaryWriter(log_dir="logs/feature_img{}".format(idx))

    """ 读取数据 """
    # train_loader = construct_loader(cfg, cfg.DATA.TRAIN_FILE, condition="normal", shuffle=True, drop_last=False)
    # test_loader = construct_loader(cfg, cfg.DATA.VAL_FILE, condition="normal", shuffle=False, drop_last=False)
    # dataset = build_dataset("IHC", cfg, cfg.DATA.VAL_FILE, "normal")


    database="MSTLoc"
    fold = 0
    path_prefix = "data/MSTLoc/MSTLoc"
    result_prefix = "{}/results/{}/fold{}".format(cfg.DATA.RESULT_DIR, database, fold)
    val_file_path = "%s_val_fold%d.csv" % (path_prefix, fold)

    # dataset = construct_loader(cfg, val_file_path, condition="normal", database=database, shuffle=False, drop_last=False)
    dataset = build_dataset("IHC", cfg, val_file_path, "normal", database=database)

    """ 构建SAE模型 """
    # SAE
    layers_list = SAEmodel(cfg.SAE.MODEL_NAME)

    for layer in range(len(layers_list)):
        layers_list[layer] = layers_list[layer].to(device)
        layers_list[layer].is_training_layer = False
        layers_list[layer].eval()

    load_best_model(cfg, layers_list, cfg.SAE.MODEL_NAME, device, result_prefix=result_prefix)

    unloader = transforms_tv.ToPILImage()

    _, img, _, _, _ = dataset.__getitem__(idx)
    img = img.unsqueeze(dim=0)
    img = img.to(device)

    for i in range(len(img)):
        writer.add_image("orginal_img", np.array(unloader(img[i])), dataformats='HWC')

    out_features = img
    for layer in range(len(layers_list)):
        print("Test layer{}".format(layer))

        out_features = layers_list[layer](out_features)
        print(out_features.shape)
        for i in range(len(out_features)):
            for f in range(len(out_features[i])):
                writer.add_image("layer{}/{}_img".format(layer, f), np.array(unloader(out_features[i][f])), dataformats='HW')
        # for f in range(len(out_features)):
        #     writer.add_image("feature_img{}/layer{}_{}_img".format(idx, layer, f), np.array(unloader(out_features[f])), dataformats='HWC')

    for layer in range(len(layers_list)):
        SAE = StackedAutoEncoder(layers_list[:layer+1], True)
        SAE = SAE.to(device)
        SAE.eval()
        out = SAE(img)
        for i in range(len(out)):
            writer.add_image("out/out{}_img".format(layer), np.array(unloader(out[i])), dataformats='HWC')


    #     criterion = get_loss_func("huber")(beta=beta).to(device)
    #     test_layer(device, test_loader, criterion, layers_list, layer)

    #     # from torchvision import transforms as transforms_tv
    # # unloader = transforms_tv.ToPILImage()

    # # # unloader(img).save('results/out_img/orginal_img.jpg')

    # # # for i in range(len(layers_list)):
    # # #     img = layers_list[i](img)
    # # #     for j in range(len(img)):
    # # #         unloader(img[j]).save('results/out_img/layer{}_{}_img.jpg'.format(i, j))


    # if world_size > 1:
    #     for layer in range(len(layers_list)):
    #         layers_list[layer] = layers_list[layer].module
    # SAE = StackedAutoEncoder(layers_list, False)

    # """ 构建分类器模型 """
    # # Classifier
    # if classification_model == "lightViT_small":
    #     model = lightvit_small(num_classes=13, window_size=7)
    # elif classification_model == "lightViT_small_224-512":
    #     model = lightvit_small(num_classes=13, window_size=7, in_chans=512, drop_rate=0.3, attn_drop_rate=0.2, drop_path_rate=0.2,)
    # elif classification_model == "lightViT_tiny_224-512":
    #     model = lightvit_tiny(num_classes=13, window_size=7, in_chans=512, drop_rate=0.3, attn_drop_rate=0.2, drop_path_rate=0.2,)
    # elif classification_model == "classifier1":
    #     model = Classifier1(SAE)
    # else: # default: lightViT_small_224-512
    #     model = lightvit_small(num_classes=13, window_size=7, in_chans=512)

    # if du.is_master_proc():
    #     print("Test classifier {}".format(classification_model))

    # SAE = SAE.to(device)
    # model = model.to(device)
    # if world_size > 1:
    #     model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    #     model = nn.parallel.DistributedDataParallel(model, device_ids=[torch.distributed.get_rank()], output_device=torch.distributed.get_rank())

    # load_best_classifier_model(model, classification_model, device)
    # if du.get_world_size() > 1:
    #     dist.barrier()

    # criterion = get_loss_func("bce_logit")(reduction="none", weight=weight, pos_weight=pos_weight).to(device)
    # test(device, test_loader, cfg.DATA.VAL_FILE, SAE, model, criterion, model_name)
    writer.close()


if __name__ == "__main__":
    main()