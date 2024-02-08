import torch

from adabelief_pytorch import AdaBelief


def construct_optimizer(params, optimizer_name,
    lr=3e-04, momentum=0.9, dampening=0, weight_decay=0.05, nesterov=True,
    lr_decay=0, rho=0.9, eps=1e-08, alpha=0.99, centered=False, betas=(0.9, 0.999), amsgrad=False
):
    """_summary_

    Args:
        params (_type_): Network parameters to be optimised
        optimizer_name (_type_): Name of the optimiser used
        lr (float, optional): learning rate. Defaults to 3e-04.
        momentum (float, optional): momentum factor. Defaults to 0.9.
        dampening (int, optional): Suppressor of Momentum. Defaults to 0.
        weight_decay (float, optional): Weight decay (L2 penalty). Defaults to 0.05.
        nesterov (bool, optional): Using Nesterov momentum. Defaults to True.
        lr_decay (int, optional): Learning rate decay. Defaults to 0.
        rho (float, optional): Coefficients used to calculate the running mean of the squared gradient. Defaults to 0.9.
        eps (_type_, optional): Terms added to the denominator to increase the stability of numerical calculations. Defaults to 1e-08.
        alpha (float, optional): smoothness constant. Defaults to 0.99.
        centered (bool, optional): If True, compute the centred RMSProp and normalise the gradient with its variance prediction value. Defaults to False.
        betas (tuple, optional): Coefficients used to calculate the gradient and the running average of the gradient squared. Defaults to (0.9, 0.999).
        amsgrad (bool, optional): Whether or not to use the AMSGrad variant of the algorithm mentioned in the paper On the Convergence of Adam and Beyond. Defaults to False.

    Raises:
        NotImplementedError: Unsupported optimiser names

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
            lr=1.0, # The factor by which delta is scaled before it is applied to the parameter update
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