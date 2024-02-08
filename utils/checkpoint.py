import os

import torch
import torch.distributed as dist

import utils.distributed as du


def save_checkpoint(checkpoint_path, model, optimizer, scheduler, scaler, epoch, loss):
    if not os.path.exists(checkpoint_path.rsplit('/', 1)[0]):
        os.makedirs(checkpoint_path.rsplit('/', 1)[0])

    state = {'model': model.module.state_dict() if du.get_world_size() > 1 else model.state_dict(),
    # state = {'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler else None,
            'scaler': scaler.state_dict(),
            'epoch': epoch,
            'min_loss': loss}
    torch.save(state, checkpoint_path)


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, scaler):
    if os.path.exists(checkpoint_path):
        # checkpoint = torch.load(checkpoint_path, map_location={"cuda:0":"cuda:{}".format(dist.get_rank())})
        # checkpoint = torch.load(checkpoint_path, map_location={"cpu":"cuda:{}".format(dist.get_rank())})
        # checkpoint = torch.load(checkpoint_path, map_location="cpu")
        checkpoint = torch.load(checkpoint_path, map_location="cuda:{}".format(dist.get_rank()) if du.get_world_size() > 1 else "cuda"  if torch.cuda.is_available() else "cpu")
        ms = model.module if du.get_world_size() > 1 else model
        # ms = model
        ms.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler:
            scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint['epoch'] + 1
        min_loss = checkpoint['min_loss']
        if du.is_master_proc():
            print("Get checkpoint from epoch {}, min_loss={}".format(start_epoch, min_loss))
    else:
        start_epoch = 0
        min_loss = float("inf")

    return start_epoch, min_loss


def save_pretrain_checkpoint(checkpoint_path, model, optimizer, scheduler, scaler, epoch, loss, dict_):
    if not os.path.exists(checkpoint_path.rsplit('/', 1)[0]):
        os.makedirs(checkpoint_path.rsplit('/', 1)[0])

    state = {'model': model.module.state_dict() if du.get_world_size() > 1 else model.state_dict(),
    # state = {'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'epoch': epoch,
            'min_loss': loss,
            'dict': dict_}
    torch.save(state, checkpoint_path)


def load_pretrain_checkpoint(checkpoint_path, model, optimizer, scheduler, scaler):
    if os.path.exists(checkpoint_path):
        # checkpoint = torch.load(checkpoint_path, map_location={"cuda:0":"cuda:{}".format(dist.get_rank())})
        # checkpoint = torch.load(checkpoint_path, map_location={"cpu":"cuda:{}".format(dist.get_rank())})
        # checkpoint = torch.load(checkpoint_path, map_location="cpu")
        checkpoint = torch.load(checkpoint_path, map_location="cuda:{}".format(dist.get_rank()) if du.get_world_size() > 1 else "cuda"  if torch.cuda.is_available() else "cpu")
        ms = model.module if du.get_world_size() > 1 else model
        # ms = model
        ms.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint['epoch'] + 1
        min_loss = checkpoint['min_loss']
        dict_ = checkpoint['dict']
        if du.is_master_proc():
            print("Get checkpoint from epoch {}, min_loss={}".format(start_epoch, min_loss))
    else:
        start_epoch = 0
        min_loss = float("inf")
        dict_ = dict()

    return start_epoch, min_loss, dict_


def load_checkpoint_test(checkpoint_path, model, load_head=True, head_layer="head"):
    if os.path.exists(checkpoint_path):
        # checkpoint = torch.load(checkpoint_path, map_location={"cpu":"cuda:{}".format(dist.get_rank())})
        # checkpoint = torch.load(checkpoint_path, map_location="cpu")
        checkpoint = torch.load(checkpoint_path, map_location="cuda:{}".format(dist.get_rank()) if du.get_world_size() > 1 else "cuda" if torch.cuda.is_available() else "cpu")
        ms = model.module if du.get_world_size() > 1 else model
        if load_head:
            ms.load_state_dict(checkpoint['model'], strict=False)
        else:
            model_dict = ms.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint['model'].items() if (k in model_dict) and (head_layer not in k)}
            model_dict.update(pretrained_dict)
            ms.load_state_dict(model_dict)

        min_loss = checkpoint['min_loss']
        if du.is_master_proc():
            print("Get checkpoint from {}, min_loss={}".format(checkpoint_path, min_loss))


def load_pretrained_lightvit(cfg, model, model_name):
    pretrained_path = ""
    if model_name == "lightViT_small":
        pretrained_path = "{}/results/lightvit_small_80.9.ckpt".format(cfg.DATA.RESULT_DIR)
    elif model_name == "lightViT_tiny":
        pretrained_path = "{}/results/lightvit_tiny_78.7.ckpt".format(cfg.DATA.RESULT_DIR)
    elif model_name == "lightViT_base":
        pretrained_path = "{}/results/lightvit_base_82.1.ckpt".format(cfg.DATA.RESULT_DIR)
    if os.path.exists(pretrained_path):
        pretrained_dict = torch.load(pretrained_path, map_location="cuda:{}".format(dist.get_rank()) if torch.cuda.is_available() else "cpu")['state_dict']

        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict) and ('neck' not in k) and ('head' not in k)}
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict) and ('head' not in k)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        # 模型微调，只调整neck和head
        for name, param in model.named_parameters():
            if ('neck' in name) or ('head' in name):
                param.requires_grad = True
            else:
                param.requires_grad = False


def split_params(model, head_layer='head'):
    pretrain_params = []
    head_params = []
    if du.get_world_size() > 1:
        ms = model.module
    else:
        ms = model
    for name, param in ms.named_parameters():
        if head_layer not in name:
            pretrain_params += [param]
        else:
            head_params += [param]

    return pretrain_params, head_params