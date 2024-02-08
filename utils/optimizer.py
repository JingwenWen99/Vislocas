import torch

from adabelief_pytorch import AdaBelief


def construct_optimizer(params, optimizer_name,
    lr=3e-04, momentum=0.9, dampening=0, weight_decay=0.05, nesterov=True,
    lr_decay=0, rho=0.9, eps=1e-08, alpha=0.99, centered=False, betas=(0.9, 0.999), amsgrad=False
):
    """_summary_

    Args:
        params (_type_): 需要优化的网络参数
        optimizer_name (_type_): 使用的优化器名称
        lr (float, optional): 学习率. Defaults to 3e-04.
        momentum (float, optional): 动量因子. Defaults to 0.9.
        dampening (int, optional): 动量的抑制因子. Defaults to 0.
        weight_decay (float, optional): 权重衰减（L2惩罚）. Defaults to 0.05.
        nesterov (bool, optional): 使用Nesterov动量. Defaults to True.
        lr_decay (int, optional): 学习率衰减. Defaults to 0.
        rho (float, optional): 用于计算平方梯度的运行平均值的系数. Defaults to 0.9.
        eps (_type_, optional): 为了增加数值计算的稳定性而加到分母里的项. Defaults to 1e-08.
        alpha (float, optional): 平滑常数. Defaults to 0.99.
        centered (bool, optional): 如果为True，计算中心化的RMSProp，并且用它的方差预测值对梯度进行归一化. Defaults to False.
        betas (tuple, optional): 用于计算梯度以及梯度平方的运行平均值的系数. Defaults to (0.9, 0.999).
        amsgrad (bool, optional): 是否使用从论文On the Convergence of Adam and Beyond中提到的算法的AMSGrad变体. Defaults to False.

    Raises:
        NotImplementedError: 不支持的优化器名称

    Returns:
        _type_: optimizer
    """
    if optimizer_name == "sgd":
        return torch.optim.SGD(
            params,
            lr=lr,
            momentum=0,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=False
        )
    elif optimizer_name == "momentum":
        return torch.optim.SGD(
            params,
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=False
        )
    elif optimizer_name == "adagrad":
        return torch.optim.Adagrad(
            params,
            lr=lr,
            lr_decay=lr_decay,
            weight_decay=weight_decay
        )
    elif optimizer_name == "adadelta":
        return torch.optim.Adadelta(
            params,
            lr=1.0, # 在delta被应用到参数更新之前对它缩放的系数
            rho=rho,
            eps=eps,
            weight_decay=weight_decay
        )
    elif optimizer_name == "rmsprop":
        return torch.optim.RMSprop(
            params,
            lr=lr,
            alpha=alpha,
            eps=eps,
            weight_decay=weight_decay,
            momentum=momentum,
            centered=centered
        )
    elif optimizer_name == "adam":
        return torch.optim.Adam(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay
        )
    elif optimizer_name == "adamw":
        return torch.optim.AdamW(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad
        )
    elif optimizer_name == "adabelief":
        return AdaBelief(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            weight_decouple = True, rectify = False
        )
    else:
        raise NotImplementedError(
            "Does not support {} optimizer".format(optimizer_name)
        )


_OPTIMIZERS = {
    "sgd": torch.optim.SGD,
    "momentum": torch.optim.SGD,
    "adagrad": torch.optim.Adagrad,
    "adadelta": torch.optim.Adadelta,
    "rmsprop": torch.optim.RMSprop,
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "adabelief": AdaBelief,
}


def get_optimizer_func(optimizer_name):
    if optimizer_name not in _OPTIMIZERS.keys():
        raise NotImplementedError( "Does not support {} optimizer".format(optimizer_name))
    return _OPTIMIZERS[optimizer_name]