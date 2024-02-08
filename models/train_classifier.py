import os
import time

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from contextlib import nullcontext

import utils.distributed as du
import utils.checkpoint as cu
from datasets.loader import shuffle_dataset
from models.criterion import t_criterion
from models.losses import l1_regularization, l2_regularization
from utils.eval_metrics import evaluate
from utils.config_defaults import labelLists


def train_epoch(cfg, device, train_loader, SAE, model, optimizer, scaler, criterion=nn.BCEWithLogitsLoss(reduction="none"), l1_alpha=0, l2_alpha=0, cur_epoch=0, epoch=None, writer=None):
    start_time = time.time()

    model.train()

    avg_loss = 0.

    for cur_iter, (_, inputs, labels, _, _) in enumerate(train_loader):
        iter_start_time = time.time()

        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Use no_sync only in DDP mode when the number of rounds is not an integer multiple of K
        my_context = model.no_sync if du.get_world_size() > 1 and (cur_iter % cfg.CLASSIFIER.ACCUMULATION_STEPS != 0 and cur_iter + 1 != len(train_loader)) else nullcontext
        with my_context():
            with torch.cuda.amp.autocast(enabled=True):
                preds = model(inputs)
                loss = criterion(preds, labels).mean(0)
                l1_loss = 0
                if l1_alpha > 0:
                    l1_loss = l1_regularization(model, l1_alpha)
                    loss += l1_loss
                if l2_alpha > 0:
                    loss += l2_regularization(model, l2_alpha)
            scaler.scale(loss.mean() / cfg.CLASSIFIER.ACCUMULATION_STEPS).backward()  # Scales loss.  Zoom in on the gradient first to prevent the gradient from disappearing
        if cur_iter % cfg.CLASSIFIER.ACCUMULATION_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()


        avg_loss += loss

        if du.get_world_size() > 1:
            [loss] = du.all_reduce([loss])

        if du.is_master_proc():
            writer.add_scalar(tag="classifier/loss/train", scalar_value=loss.mean(), global_step=(cur_epoch * len(train_loader) + cur_iter))

            if (cur_iter + 1) % cfg.CLASSIFIER.PRINT_STEPS == 0:
                print("Train Model, Epoch: {}/{}, Iter: {}/{}, Losses: {}, Avg Loss: {}, L1 Loss: {}, Time consuming: {:.2f}, Total time consuming: {:.2f}".format(
                    (cur_epoch + 1), epoch, (cur_iter + 1), len(train_loader), loss, loss.mean(), l1_loss, time.time() - iter_start_time, time.time() - start_time
                ))
    avg_loss /= len(train_loader)
    if du.get_world_size() > 1:
        [avg_loss] = du.all_reduce([avg_loss])
    if du.is_master_proc():
        writer.add_scalar(tag="classifier/avgloss/train", scalar_value=avg_loss.mean(), global_step=cur_epoch)
        writer.add_scalar(tag="classifier/lr/train", scalar_value=optimizer.param_groups[0]['lr'], global_step=cur_epoch)
        print("Train Model, Epoch: {}/{}, Avg Losses: {}, Avg Loss: {}, Time consuming: {:.2f}".format(
            (cur_epoch + 1), epoch, avg_loss, avg_loss.mean(), time.time() - start_time
        ))

    return avg_loss.mean()


@torch.no_grad()
def val_epoch(cfg, device, val_loader, data_file, SAE, model, criterion=nn.BCEWithLogitsLoss(reduction="none"), l1_alpha=0, l2_alpha=0, cur_epoch=0, epoch=None, model_name="lightViT_small", multilabel=True, writer=None, metricsWriter=None, result_prefix=None, log_prefix=None, prefix=""):
    start_time = time.time()

    model.eval()

    avg_loss = 0.

    all_idxs = []
    all_labels = []
    all_preds = []

    for cur_iter, (idx, inputs, labels, _, _) in enumerate(val_loader):
        iter_start_time = time.time()

        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        preds = model(inputs)

        loss = criterion(preds, labels).mean(0)
        l1_loss = 0
        if l1_alpha > 0:
            l1_loss = l1_regularization(model, l1_alpha)
            loss += l1_loss
        if l2_alpha > 0:
            loss += l2_regularization(model, l2_alpha)
        avg_loss += loss

        preds = preds.cpu()
        labels = labels.cpu()

        m = nn.Sigmoid()
        preds = m(preds)

        all_idxs.extend(idx.tolist())
        all_labels.append(labels)
        all_preds.append(torch.unsqueeze(preds, 0) if preds.dim() == 1 else preds)

        if du.get_world_size() > 1:
            [loss] = du.all_reduce([loss])

        if du.is_master_proc():
            if writer:
                writer.add_scalar(tag="classifier/loss/val", scalar_value=loss.mean(), global_step=(cur_epoch * len(val_loader) + cur_iter))

            if (cur_iter + 1) % cfg.CLASSIFIER.PRINT_STEPS == 0:
                print("Validate Model, Epoch: {}/{}, Iter: {}/{}, Losses: {}, Avg Loss: {}, L1 Loss: {}, Time consuming: {:.2f}, Total time consuming: {:.2f}".format(
                    (cur_epoch + 1), epoch, (cur_iter + 1), len(val_loader), loss, loss.mean(), l1_loss, time.time() - iter_start_time, time.time() - start_time
                ))

    avg_loss /= len(val_loader)
    if du.get_world_size() > 1:
        [avg_loss] = du.all_reduce([avg_loss])
    if du.is_master_proc():
        if writer:
            writer.add_scalar(tag="classifier/avgloss/val", scalar_value=avg_loss.mean(), global_step=cur_epoch)

    all_labels = np.concatenate(all_labels, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)

    all_idxs = torch.as_tensor(all_idxs).to(device)
    all_labels = torch.as_tensor(all_labels).to(device)
    all_preds = torch.as_tensor(all_preds).to(device)

    world_size = du.get_world_size()
    if world_size > 1:
        # dist.barrier()
        if du.is_master_proc():
            gather_idxs = [torch.zeros_like(all_idxs) for _ in range(world_size)]
            gather_labels = [torch.zeros_like(all_labels) for _ in range(world_size)]
            gather_preds = [torch.zeros_like(all_preds) for _ in range(world_size)]

            dist.gather(tensor=all_idxs, gather_list=gather_idxs, dst=0)
            dist.gather(tensor=all_labels, gather_list=gather_labels, dst=0)
            dist.gather(tensor=all_preds, gather_list=gather_preds, dst=0)

            gather_idxs = [item.cpu().detach().numpy() for item in gather_idxs]
            gather_labels = [item.cpu().detach().numpy() for item in gather_labels]
            gather_preds = [item.cpu().detach().numpy() for item in gather_preds]

            all_idxs = np.array(gather_idxs).flatten()
            all_labels = np.concatenate(gather_labels, axis=0)
            all_preds = np.concatenate(gather_preds, axis=0)

            _, ind = np.unique(all_idxs, return_index=True)
            all_idxs = all_idxs[ind]
            all_labels = all_labels[ind]
            all_preds = all_preds[ind]

            evaluate(cfg, all_idxs, all_labels, all_preds, data_file, model_name, result_prefix=result_prefix, log_prefix=log_prefix, metricsWriter=metricsWriter, cur_epoch=cur_epoch, get_threshold=False, threshold=0.5, multilabel=multilabel, prefix=prefix)

        else:
            dist.gather(tensor=all_idxs, dst=0)
            dist.gather(tensor=all_labels, dst=0)
            dist.gather(tensor=all_preds, dst=0)

        # dist.barrier()

    if du.is_master_proc():
        print("Validate Model, Epoch: {}/{}, Avg Losses: {}, Avg Loss: {}, Time consuming: {:.2f}".format(
            (cur_epoch + 1), epoch, avg_loss, avg_loss.mean(), time.time() - start_time
        ))

    return avg_loss.mean()


@torch.no_grad()
def test(cfg, device, test_loader, data_file, SAE, model, criterion=nn.BCEWithLogitsLoss(reduction="none"), l1_alpha=0, l2_alpha=0, model_name="lightViT_small", multilabel=True,
        threshold=None, get_threshold=False, result_prefix=None, log_prefix=None, prefix="", beta=[0.5 for i in range(10)], csvWriter=None, randomSplit=-1, fold=0, aug=0, thresh='', split='', getSPE=False, getAuc=False, getMcc=False):
    start_time = time.time()

    model.eval()

    avg_loss = 0.

    all_idxs = []
    all_labels = []
    all_preds = []

    for cur_iter, (idx, inputs, labels, _, _) in enumerate(test_loader):
        iter_start_time = time.time()

        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        preds = model(inputs)

        loss = criterion(preds, labels).mean(0)
        l1_loss = 0
        if l1_alpha > 0:
            l1_loss = l1_regularization(model, l1_alpha)
            loss += l1_loss
        if l2_alpha > 0:
            loss += l2_regularization(model, l2_alpha)
        avg_loss += loss

        preds = preds.cpu()
        labels = labels.cpu()

        m = nn.Sigmoid()
        preds = m(preds)

        all_idxs.extend(idx.tolist())
        all_labels.append(labels)
        all_preds.append(torch.unsqueeze(preds, 0) if preds.dim() == 1 else preds)

        if du.get_world_size() > 1:
            [loss] = du.all_reduce([loss])

        if du.is_master_proc():
            if (cur_iter + 1) % cfg.CLASSIFIER.PRINT_STEPS == 0:
                print("Test Model, Iter: {}/{}, Losses: {}, Avg Loss: {}, L1 Loss: {}, Time consuming: {:.2f}, Total time consuming: {:.2f}".format(
                    (cur_iter + 1), len(test_loader), loss, loss.mean(), l1_loss, time.time() - iter_start_time, time.time() - start_time
                ))

    avg_loss /= len(test_loader)
    if du.get_world_size() > 1:
        [avg_loss] = du.all_reduce([avg_loss])

    all_labels = np.concatenate(all_labels, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)

    all_idxs = torch.as_tensor(all_idxs).to(device)
    all_labels = torch.as_tensor(all_labels).to(device)
    all_preds = torch.as_tensor(all_preds).to(device)

    optimal_thres = threshold
    world_size = du.get_world_size()
    if world_size > 1:
        # dist.barrier()
        if du.is_master_proc():
            gather_idxs = [torch.zeros_like(all_idxs) for _ in range(world_size)]
            gather_labels = [torch.zeros_like(all_labels) for _ in range(world_size)]
            gather_preds = [torch.zeros_like(all_preds) for _ in range(world_size)]

            dist.gather(tensor=all_idxs, gather_list=gather_idxs, dst=0)
            dist.gather(tensor=all_labels, gather_list=gather_labels, dst=0)
            dist.gather(tensor=all_preds, gather_list=gather_preds, dst=0)

            gather_idxs = [item.cpu().detach().numpy() for item in gather_idxs]
            gather_labels = [item.cpu().detach().numpy() for item in gather_labels]
            gather_preds = [item.cpu().detach().numpy() for item in gather_preds]

            all_idxs = np.array(gather_idxs).flatten()
            all_labels = np.concatenate(gather_labels, axis=0)
            all_preds = np.concatenate(gather_preds, axis=0)

            _, ind = np.unique(all_idxs, return_index=True)
            all_idxs = all_idxs[ind]
            all_labels = all_labels[ind]
            all_preds = all_preds[ind]

            optimal_thres = evaluate(cfg, all_idxs, all_labels, all_preds, data_file, model_name,
                        result_prefix=result_prefix, log_prefix=log_prefix, get_threshold=get_threshold, threshold=threshold, multilabel=multilabel, prefix=prefix,
                        csvWriter=csvWriter, randomSplit=randomSplit, fold=fold, aug=aug, thresh=thresh, split=split, beta=beta, getSPE=getSPE, getAuc=getAuc, getMcc=getMcc)

        else:
            dist.gather(tensor=all_idxs, dst=0)
            dist.gather(tensor=all_labels, dst=0)
            dist.gather(tensor=all_preds, dst=0)

        # dist.barrier()

    if du.is_master_proc():
        print("Validate Model, Avg Losses: {}, Avg Loss: {}, Time consuming: {:.2f}".format(
            avg_loss, avg_loss.mean(), time.time() - start_time
        ))

    return avg_loss.mean(), optimal_thres


@torch.no_grad()
def cancerTest(cfg, device, test_loader, data_file, SAE, model, model_name="lightViT_small", multilabel=True,
        threshold=None, result_prefix=None, aug=0, thresh=''):
    start_time = time.time()

    model.eval()

    all_idxs = []
    all_preds = []

    for cur_iter, (idx, inputs, _, _, _) in enumerate(test_loader):
        iter_start_time = time.time()

        inputs = inputs.to(device, non_blocking=True)
        preds = model(inputs)

        preds = preds.cpu()

        m = nn.Sigmoid()
        preds = m(preds)

        all_idxs.extend(idx.tolist())
        all_preds.append(torch.unsqueeze(preds, 0) if preds.dim() == 1 else preds)

        if du.is_master_proc():
            if (cur_iter + 1) % cfg.CLASSIFIER.PRINT_STEPS == 0:
                print("Test Cancer Data, Iter: {}/{}, Time consuming: {:.2f}, Total time consuming: {:.2f}".format(
                    (cur_iter + 1), len(test_loader), time.time() - iter_start_time, time.time() - start_time
                ))

    all_preds = np.concatenate(all_preds, axis=0)

    all_idxs = torch.as_tensor(all_idxs).to(device)
    all_preds = torch.as_tensor(all_preds).to(device)

    world_size = du.get_world_size()
    if world_size > 1:
        if du.is_master_proc():
            gather_idxs = [torch.zeros_like(all_idxs) for _ in range(world_size)]
            gather_preds = [torch.zeros_like(all_preds) for _ in range(world_size)]

            dist.gather(tensor=all_idxs, gather_list=gather_idxs, dst=0)
            dist.gather(tensor=all_preds, gather_list=gather_preds, dst=0)

            gather_idxs = [item.cpu().detach().numpy() for item in gather_idxs]
            gather_preds = [item.cpu().detach().numpy() for item in gather_preds]

            all_idxs = np.array(gather_idxs).flatten()
            all_preds = np.concatenate(gather_preds, axis=0)

            _, ind = np.unique(all_idxs, return_index=True)
            all_idxs = all_idxs[ind]
            all_preds = all_preds[ind]


            locations = cfg.CLASSIFIER.LOCATIONS
            labels = labelLists.get('IHC', locations)
            locations_pred = [i + '_pred' for i in labels]
            locations_pred_labels = [i + '_pred_labels' for i in labels]

            all_preds = pd.DataFrame(all_preds, columns=locations)
            all_preds = np.array(all_preds[labels])

            if multilabel:
                all_pred_labels = t_criterion(all_preds, threshold or 0.5)
            else:
                all_pred_labels = max_criterion(all_preds)

            labeledData = pd.read_csv(data_file, header=0, index_col=0)
            predData = pd.DataFrame(columns=locations_pred, index=all_idxs)
            predData[locations_pred] = all_preds
            predData[locations_pred_labels] = all_pred_labels
            predData = pd.merge(labeledData, predData, how='left', left_index=True, right_index=True)

            if not os.path.exists("{}/{}/preds".format(result_prefix, model_name)):
                os.makedirs("{}/{}/preds".format(result_prefix, model_name))
            predData.to_csv("{}/{}/preds/test_t=0.5_aug{}_{}".format(result_prefix, model_name, aug, data_file.split('/')[-1]), index=True, mode='w')

        else:
            dist.gather(tensor=all_idxs, dst=0)
            dist.gather(tensor=all_preds, dst=0)

    if du.is_master_proc():
        print("Test Cancer Data, Time consuming: {:.2f}".format(
            time.time() - start_time
        ))



def train(cfg, SAE, model, loader=None,
    optimizer=None, scheduler=None, scaler=None, criterion=nn.BCEWithLogitsLoss(), l1_alpha=0, l2_alpha=0,
    epoch=None, start_epoch=0, min_loss=float("inf"), model_name="lightViT_small", multilabel=True, patch_size="default",
    result_prefix=None, log_prefix=None, train_file_path=None, val_file_path=None,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), prefix=""
):
    start_time = time.time()

    # load data
    train_loader, val_loader = loader

    summaryWriter = None
    testWriter = None
    trainMetricsWriter = None
    testMetricsWriter = None

    # dist.barrier()

    if du.is_master_proc():
        # print("Initialisation and data reading time consuming：", time.time() - start_time)
        summaryWriter = SummaryWriter(log_dir="logs/{}/classifier/{}/{}train".format(log_prefix, model_name, prefix))
        testWriter = SummaryWriter(log_dir="logs/{}/classifier/{}/{}test".format(log_prefix, model_name, prefix))
        trainMetricsWriter = SummaryWriter(log_dir="logs/{}/metrics/{}/{}{}".format(log_prefix, model_name, prefix, train_file_path.split('/')[-1]))
        testMetricsWriter = SummaryWriter(log_dir="logs/{}/metrics/{}/{}{}".format(log_prefix, model_name, prefix, val_file_path.split('/')[-1]))

    for cur_epoch in range(start_epoch, epoch):
        shuffle_dataset(train_loader, cur_epoch)
        train_epoch(cfg, device, train_loader, SAE, model, optimizer, scaler, criterion=criterion, l1_alpha=l1_alpha, l2_alpha=l2_alpha, cur_epoch=cur_epoch, epoch=epoch, writer=summaryWriter)
        # torch.cuda.empty_cache()

        if (cur_epoch + 1) % cfg.CLASSIFIER.EVALUATION_STEPS == 0:
            val_epoch(cfg, device, train_loader, train_file_path, SAE, model, criterion=criterion, l1_alpha=l1_alpha, l2_alpha=l2_alpha, cur_epoch=cur_epoch, epoch=epoch, model_name=model_name, multilabel=multilabel, writer=summaryWriter, metricsWriter=trainMetricsWriter, result_prefix=result_prefix, log_prefix=log_prefix, prefix=prefix)
            # torch.cuda.empty_cache()
            val_loss = val_epoch(cfg, device, val_loader, val_file_path, SAE, model, criterion=criterion, l1_alpha=l1_alpha, l2_alpha=l2_alpha, cur_epoch=cur_epoch, epoch=epoch, model_name=model_name, multilabel=multilabel, writer=testWriter, metricsWriter=testMetricsWriter, result_prefix=result_prefix, log_prefix=log_prefix, prefix=prefix)
            # torch.cuda.empty_cache()

            if du.is_master_proc():
                save_start_time = time.time()

                if val_loss < min_loss:
                    min_loss = val_loss
                    best_path = "{}/{}/{}best_model.pth".format(result_prefix, model_name, prefix)
                    cu.save_checkpoint(best_path, model, optimizer, scheduler, scaler, cur_epoch, val_loss)
                    # # state = {'model': model.module.state_dict() if du.get_world_size() > 1 else model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(), 'epoch': cur_epoch, 'min_loss': val_loss}
                    # state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(), 'epoch': cur_epoch, 'min_loss': val_loss}
                    # torch.save(state, best_path)
                    print("模型{}_{}best_model已保存，loss={}".format(model_name, prefix, val_loss))

                print('模型保存耗时：', time.time() - save_start_time)
                path = "{}/{}/{}latest_model.pth".format(result_prefix, model_name, prefix)
                cu.save_checkpoint(path, model, optimizer, scheduler, scaler, cur_epoch, min_loss)
                # # state = {'model': model.module.state_dict() if du.get_world_size() > 1 else model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(), 'epoch': cur_epoch, 'min_loss': min_loss}
                # state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(), 'epoch': cur_epoch, 'min_loss': min_loss}
                # torch.save(state, path)
                print("模型{}_{}model_{}已保存".format(model_name, prefix, cur_epoch))

            dist.barrier()

        if scheduler:
            scheduler.step()

    if du.is_master_proc():
        summaryWriter.close()
        testWriter.close()
        trainMetricsWriter.close()
        testMetricsWriter.close()
        end_time = time.time()
        print('Time consuming:', end_time - start_time)


def pre_train_epoch(cfg, device, train_loader, SAE, model, optimizer, scaler, criterion=nn.BCEWithLogitsLoss(reduction="none"), dict_=None, cur_epoch=0, epoch=None, writer=None):
    start_time = time.time()

    model.train()

    avg_loss = 0.

    for cur_iter, (idx, inputs, _, bag_id, condition) in enumerate(train_loader):
        iter_start_time = time.time()

        inputs = inputs.to(device, non_blocking=True)

        # 只在DDP模式下，轮数不是K整数倍的时候使用no_sync
        my_context = model.no_sync if du.get_world_size() > 1 and (cur_iter % cfg.CLASSIFIER.PRE.ACCUMULATION_STEPS != 0 and cur_iter + 1 != len(train_loader)) else nullcontext
        # my_context = model.no_sync if du.get_world_size() > 1 and cur_iter % cfg.CLASSIFIER.PRE.ACCUMULATION_STEPS != 0 else nullcontext
        with my_context():
            # with torch.autograd.set_detect_anomaly(True):
            with torch.cuda.amp.autocast(enabled=True):
                features = model(inputs)
                features = F.normalize(features, p=2, dim=-1)

                world_size = du.get_world_size()
                if world_size > 1:

                    idx = idx.to(device, non_blocking=True)
                    bag_id = bag_id.to(device, non_blocking=True)
                    condition = condition.to(device, non_blocking=True)

                    gather_features = [torch.zeros_like(features) for _ in range(world_size)]
                    gather_idx = [torch.zeros_like(idx) for _ in range(world_size)]
                    gather_bag_id = [torch.zeros_like(bag_id) for _ in range(world_size)]
                    gather_condition = [torch.zeros_like(condition) for _ in range(world_size)]

                    dist.all_gather(tensor=features, tensor_list=gather_features)
                    dist.all_gather(tensor=idx, tensor_list=gather_idx)
                    dist.all_gather(tensor=bag_id, tensor_list=gather_bag_id)
                    dist.all_gather(tensor=condition, tensor_list=gather_condition)

                    dist.barrier()

                    gather_features = torch.cat(gather_features, dim=0).to(device)
                    gather_idx = torch.cat(gather_idx, dim=0).to(device)
                    gather_bag_id = torch.cat(gather_bag_id, dim=0).to(device)
                    gather_condition = torch.cat(gather_condition, dim=0).to(device)

                    for k in range(len(gather_bag_id)):
                        new_bag_id = str(gather_condition[k].item()) + "-" + str(gather_bag_id[k].item())
                        if new_bag_id in dict_:
                            if gather_idx[k].item() in dict_[new_bag_id]:
                                dict_[new_bag_id].update({gather_idx[k].item(): dict_[new_bag_id][gather_idx[k].item()] * cfg.CLASSIFIER.PRE.M + gather_features[k].detach().data * (1 - cfg.CLASSIFIER.PRE.M)})
                            dict_[new_bag_id].update({gather_idx[k].item(): gather_features[k].detach().data})
                        else:
                            dict_.update({new_bag_id: {gather_idx[k].item(): gather_features[k].detach().data}})

                loss = criterion(features, dict_, idx.detach(), bag_id.detach(), condition.detach(), gather_bag_id.detach(), gather_condition.detach())

            scaler.scale(loss / cfg.CLASSIFIER.PRE.ACCUMULATION_STEPS).backward()  # Scales loss.  先将梯度放大 防止梯度消失

        if cur_iter % cfg.CLASSIFIER.PRE.ACCUMULATION_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        avg_loss += loss

        if du.get_world_size() > 1:
            [loss] = du.all_reduce([loss])

        if du.is_master_proc():
            writer.add_scalar(tag="classifier/loss/train", scalar_value=loss.mean(), global_step=(cur_epoch * len(train_loader) + cur_iter))

            if (cur_iter + 1) % cfg.CLASSIFIER.PRE.PRINT_STEPS == 0:
                print("Train Model, Epoch: {}/{}, Iter: {}/{}, Losses: {}, Avg Loss: {}, Time consuming: {:.2f}, Total time consuming: {:.2f}".format(
                    (cur_epoch + 1), epoch, (cur_iter + 1), len(train_loader), loss, loss.mean(), time.time() - iter_start_time, time.time() - start_time
                ))
    avg_loss /= len(train_loader)
    if du.get_world_size() > 1:
        [avg_loss] = du.all_reduce([avg_loss])
    if du.is_master_proc():
        writer.add_scalar(tag="classifier/avgloss/train", scalar_value=avg_loss.mean(), global_step=cur_epoch)
        writer.add_scalar(tag="classifier/lr/train", scalar_value=optimizer.param_groups[0]['lr'], global_step=cur_epoch)
        print("Train Model, Epoch: {}/{}, Avg Losses: {}, Avg Loss: {}, Time consuming: {:.2f}".format(
            (cur_epoch + 1), epoch, avg_loss, avg_loss.mean(), time.time() - start_time
        ))

    return avg_loss.mean()


@torch.no_grad()
def pre_val_epoch(cfg, device, val_loader, data_file, SAE, model, criterion=nn.BCEWithLogitsLoss(reduction="none"), dict_=None, cur_epoch=0, epoch=None, model_name="lightViT_small", writer=None):
    start_time = time.time()

    model.eval()

    avg_loss = 0.

    for cur_iter, (idx, inputs, _, bag_id, condition) in enumerate(val_loader):
        iter_start_time = time.time()

        inputs = inputs.to(device, non_blocking=True)

        features = model(inputs)
        features = F.normalize(features, p=2, dim=-1)

        world_size = du.get_world_size()
        if world_size > 1:

            idx = idx.to(device, non_blocking=True)
            bag_id = bag_id.to(device, non_blocking=True)
            condition = condition.to(device, non_blocking=True)

            gather_features = [torch.zeros_like(features) for _ in range(world_size)]
            gather_idx = [torch.zeros_like(idx) for _ in range(world_size)]
            gather_bag_id = [torch.zeros_like(bag_id) for _ in range(world_size)]
            gather_condition = [torch.zeros_like(condition) for _ in range(world_size)]

            dist.all_gather(tensor=features, tensor_list=gather_features)
            dist.all_gather(tensor=idx, tensor_list=gather_idx)
            dist.all_gather(tensor=bag_id, tensor_list=gather_bag_id)
            dist.all_gather(tensor=condition, tensor_list=gather_condition)

            dist.barrier()

            gather_features = torch.cat(gather_features, dim=0).to(device)
            gather_idx = torch.cat(gather_idx, dim=0).to(device)
            gather_bag_id = torch.cat(gather_bag_id, dim=0).to(device)
            gather_condition = torch.cat(gather_condition, dim=0).to(device)

            for k in range(len(gather_bag_id)):
                new_bag_id = str(gather_condition[k].item()) + "-" + str(gather_bag_id[k].item())
                if new_bag_id in dict_:
                    if gather_idx[k].item() in dict_[new_bag_id]:
                        dict_[new_bag_id].update({gather_idx[k].item(): dict_[new_bag_id][gather_idx[k].item()] * cfg.CLASSIFIER.PRE.M + gather_features[k].detach().data * (1 - cfg.CLASSIFIER.PRE.M)})
                    dict_[new_bag_id].update({gather_idx[k].item(): gather_features[k].detach().data})
                else:
                    dict_.update({new_bag_id: {gather_idx[k].item(): gather_features[k].detach().data}})

        loss = criterion(features, dict_, idx.detach(), bag_id.detach(), condition.detach(), gather_bag_id.detach(), gather_condition.detach())

        avg_loss += loss

        if du.get_world_size() > 1:
            [loss] = du.all_reduce([loss])

        if du.is_master_proc():
            writer.add_scalar(tag="classifier/loss/val", scalar_value=loss.mean(), global_step=(cur_epoch * len(val_loader) + cur_iter))

            if (cur_iter + 1) % cfg.CLASSIFIER.PRE.PRINT_STEPS == 0:
                print("Validate Model, Epoch: {}/{}, Iter: {}/{}, Losses: {}, Avg Loss: {}, Time consuming: {:.2f}, Total time consuming: {:.2f}".format(
                    (cur_epoch + 1), epoch, (cur_iter + 1), len(val_loader), loss, loss.mean(), time.time() - iter_start_time, time.time() - start_time
                ))

    avg_loss /= len(val_loader)
    if du.get_world_size() > 1:
        [avg_loss] = du.all_reduce([avg_loss])
    if du.is_master_proc():
        writer.add_scalar(tag="classifier/avgloss/val", scalar_value=avg_loss.mean(), global_step=cur_epoch)

    if du.is_master_proc():
        print("Validate Model, Epoch: {}/{}, Avg Losses: {}, Avg Loss: {}, Time consuming: {:.2f}".format(
            (cur_epoch + 1), epoch, avg_loss, avg_loss.mean(), time.time() - start_time
        ))

    return avg_loss.mean()


def pretrain(cfg, SAE, model, loader=None,
    optimizer=None, scheduler=None, scaler=None, criterion=nn.BCEWithLogitsLoss(), dict_=None,
    epoch=None, start_epoch=0, min_loss=float("inf"), model_name="lightViT_small", patch_size="default",
    result_prefix=None, log_prefix=None, train_file_path=None, val_file_path=None,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    start_time = time.time()

    # load data
    train_loader, val_loader = loader

    summaryWriter = None
    testWriter = None

    dist.barrier()

    if du.is_master_proc():
        summaryWriter = SummaryWriter(log_dir="logs/{}/classifier/{}/pre_train".format(log_prefix, model_name))
        testWriter = SummaryWriter(log_dir="logs/{}/classifier/{}/pre_test".format(log_prefix, model_name))

    for cur_epoch in range(start_epoch, epoch):
        shuffle_dataset(train_loader, cur_epoch)
        pre_train_epoch(cfg, device, train_loader, SAE, model, optimizer, scaler, criterion, dict_, cur_epoch, epoch, summaryWriter)

        if (cur_epoch + 1) % cfg.CLASSIFIER.PRE.EVALUATION_STEPS == 0:
            pre_val_epoch(cfg, device, train_loader, train_file_path, SAE, model, criterion, dict_, cur_epoch, epoch, model_name, summaryWriter)
            val_loss = pre_val_epoch(cfg, device, val_loader, val_file_path, SAE, model, criterion, dict_, cur_epoch, epoch, model_name, testWriter)

            if du.is_master_proc():
                save_start_time = time.time()

                if val_loss < min_loss:
                    min_loss = val_loss
                    best_path = "{}/{}/best_model.pth".format(result_prefix, model_name)
                    cu.save_pretrain_checkpoint(best_path, model, optimizer, scheduler, scaler, cur_epoch, val_loss, dict_)
                    pre_best_path = "{}/{}/pretrain_best_model.pth".format(result_prefix, model_name)
                    cu.save_pretrain_checkpoint(pre_best_path, model, optimizer, scheduler, scaler, cur_epoch, val_loss, dict_)
                    print("模型{}_best_model已保存，loss={}".format(model_name, val_loss))

                print('模型保存耗时：', time.time() - save_start_time)
                path = "{}/{}/latest_model.pth".format(result_prefix, model_name)
                cu.save_pretrain_checkpoint(path, model, optimizer, scheduler, scaler, cur_epoch, min_loss, dict_)
                pre_path = "{}/{}/pretrain_latest_model.pth".format(result_prefix, model_name)
                cu.save_pretrain_checkpoint(pre_path, model, optimizer, scheduler, scaler, cur_epoch, min_loss, dict_)
                print("模型{}_model_{}已保存".format(model_name, cur_epoch))

            dist.barrier()

        scheduler.step()

    if du.is_master_proc():
        summaryWriter.close()
        testWriter.close()
        end_time = time.time()
        print('Time consuming:', end_time - start_time)



def load_best_classifier_model(cfg, model, model_name, device, prefix="", load_head=True, head_layer="head", result_prefix=None):
    checkpoint_path = "{}/{}/{}best_model.pth".format(result_prefix, model_name, prefix)
    if du.is_master_proc():
        print("Load {} best {}model from {}".format(model_name, prefix, checkpoint_path))
    cu.load_checkpoint_test(checkpoint_path, model, load_head, head_layer)
    if du.get_world_size() > 1:
        dist.barrier()
    model.to(device)

    return model


def load_best_pretrain_classifier_model(cfg, model, model_name, device, result_prefix=None):
    checkpoint_path = "{}/{}/pretrain_best_model.pth".format(result_prefix, model_name)
    if du.is_master_proc():
        print("Load {} best pretrain model from {}".format(model_name, checkpoint_path))
    cu.load_checkpoint_test(checkpoint_path, model)
    if du.get_world_size() > 1:
        dist.barrier()
    model.to(device)

    return model