import math

import torch


def construct_scheduler(
    cfg, optimizer, scheduler_name,
    step_size=10, gamma=0.5, last_epoch=-1,
    milestones=[30,60,100], T_max=10, eta_min=1e-8, lr_lambda=None,
    mode='min', factor=0.1, patience=10, verbose=False, threshold=0.0001, threshold_mode='rel',
    cooldown=0, min_lr=1e-9, eps=1e-08, T_0=5, T_mult=2
):
    if scheduler_name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
            last_epoch=last_epoch
        )
    elif scheduler_name == "multiStep":
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=gamma,
            last_epoch=last_epoch
        )
    elif scheduler_name == "exponential": # 指数衰减
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=gamma,
            last_epoch=last_epoch
        )
    elif scheduler_name == "cosineAnnealing":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=eta_min,
            last_epoch=last_epoch
        )
    elif scheduler_name == "adaptive":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            verbose=verbose,
            threshold=threshold,
            threshold_mode=threshold_mode,
            cooldown=cooldown,
            min_lr=min_lr,
            eps=eps
        )
    elif scheduler_name == "lambda":
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lr_lambda,
            last_epoch=last_epoch
        )
    elif scheduler_name == "cosineAnnealingWarmRestarts":
        return  torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=T_0,
            T_mult=T_mult,
            eta_min=eta_min,
            last_epoch=last_epoch,
            verbose=verbose)
    elif scheduler_name == "warmupCosine":
        lr_lambda = lambda cur_epoch: (0.99 * cur_epoch / cfg.T0 + cfg.END_SCALE) if cur_epoch < cfg.T0 else \
            cfg.END_SCALE if cfg.N_T * (1 + math.cos(math.pi * (cur_epoch - cfg.T0) / (cfg.EPOCH_NUM - cfg.T0))) < cfg.END_SCALE else \
            cfg.N_T * (1 + math.cos(math.pi * (cur_epoch - cfg.T0) / (cfg.EPOCH_NUM - cfg.T0)))

        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lr_lambda,
            last_epoch=last_epoch
        )
    elif scheduler_name == "warmupExponential":
        lr_lambda = lambda cur_epoch: (0.99 * cur_epoch / cfg.T0 + cfg.END_SCALE) if cur_epoch < cfg.T0 else \
            cfg.END_SCALE if cfg.GAMMA ** (cur_epoch - cfg.T0) < cfg.END_SCALE else \
            cfg.GAMMA ** (cur_epoch - cfg.T0)
            # cfg.END_SCALE if cfg.GAMMA ** ((cur_epoch - cfg.T0) // cfg.T1) < cfg.END_SCALE else \
            # cfg.GAMMA ** ((cur_epoch - cfg.T0) // cfg.T1)
            # cfg.END_SCALE if cfg.GAMMA ** (cur_epoch // cfg.T0 - 1) < cfg.END_SCALE else \
            # cfg.GAMMA ** (cur_epoch // cfg.T0 - 1)
            # cfg.END_SCALE if cfg.GAMMA ** (cur_epoch - cfg.T0) < cfg.END_SCALE else \
            # cfg.GAMMA ** (cur_epoch - cfg.T0)
            # 1 if cur_epoch < cfg.T0 * 2 else \
            # cfg.END_SCALE if cfg.GAMMA ** (cur_epoch - cfg.T0 * 2) < cfg.END_SCALE else \
            # cfg.GAMMA ** (cur_epoch - cfg.T0 * 2)

        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lr_lambda,
            last_epoch=last_epoch
        )



        # scheduler = construct_scheduler(cfg, optimizer, "lambda", lr_lambda=lambda2)
        # return  torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     optimizer,
        #     T_0=T_0,
        #     T_mult=T_mult,
        #     eta_min=eta_min,
        #     last_epoch=last_epoch,
        #     verbose=verbose)
    else:
        raise NotImplementedError(
            "Does not support {} scheduler".format(scheduler_name)
        )


_SCHEDULERS = {
    "step": torch.optim.lr_scheduler.StepLR,
    "multiStep": torch.optim.lr_scheduler.MultiStepLR,
    "exponential": torch.optim.lr_scheduler.ExponentialLR,
    "cosineAnnealing": torch.optim.lr_scheduler.CosineAnnealingLR,
    "adaptive": torch.optim.lr_scheduler.ReduceLROnPlateau,
    "lambda": torch.optim.lr_scheduler.LambdaLR,
    "cosineAnnealingWarmRestarts": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
}


def get_scheduler_func(scheduler_name):
    if scheduler_name not in _SCHEDULERS.keys():
        raise NotImplementedError( "Does not support {} scheduler".format(scheduler_name))
    return _SCHEDULERS[scheduler_name]