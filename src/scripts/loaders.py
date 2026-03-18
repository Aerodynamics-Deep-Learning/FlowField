from ..src.models.NIF import NIF
from ..src.models.MLP import MLP
from ..src.models.FNO import FNO
from ..src.utils import loss as custom_losses

import torch.optim as optim
import torch.nn as nn

def load_model(cfg_model_setup):
    
    model_type = cfg_model_setup['model_type']
    cfg_model = cfg_model_setup['cfg_model']

    if hasattr(MLP, model_type):
        model_cls = getattr(MLP, model_type)
    
    elif hasattr(NIF, model_type):
        model_cls = getattr(NIF, model_type)

    elif hasattr(FNO, model_type):
        model_cls = getattr(FNO, model_type)

    else:
        raise ValueError(f"Unknown/Unsupported model type: {model_type}")
    
    model = model_cls(**cfg_model)

    return model

def load_optim(cfg_optim_setup, model):

    optim_type = cfg_optim_setup['optim_type']
    cfg_optim = cfg_optim_setup.get('cfg_optim', {})

    try:
        opt_cls = getattr(optim, optim_type)
    except:
        raise ValueError(f"Optimizer '{optim_type}' is not a valid optimizer in torch.optim. "
                         f"Check spelling (e.g., 'Adam' vs 'adam').")
    
    optimizer = opt_cls(model.parameters(), **cfg_optim)

    scheduler = None

    if 'cfg_scheduler_setup' in cfg_optim_setup and cfg_optim_setup['cfg_scheduler_setup']:

        scheduler_setup = cfg_optim_setup['cfg_scheduler_setup']
        scheduler_type = scheduler_setup['scheduler_type']
        cfg_scheduler = scheduler_setup.get('cfg_scheduler', {})

        try:
            scheduler_cls = getattr(optim.lr_scheduler, scheduler_type)
        except:
            raise ValueError(f"Scheduler: {scheduler_type} does not exist in torch.optim.lr_scheduler")
        
        scheduler = scheduler_cls(optimizer, **cfg_scheduler)

    return optimizer, scheduler

def load_loss(cfg_loss_setup):

    loss_type= cfg_loss_setup['loss_type']
    cfg_loss = cfg_loss_setup.get('cfg_loss', {})

    if hasattr(custom_losses, loss_type):
        loss_cls = getattr(custom_losses, loss_type)
    
    elif hasattr(nn, loss_type):
        loss_cls = getattr(nn, loss_type)

    else:
        raise ValueError(f"Unknown/Unsupported loss type: {loss_type}")
    
    loss = loss_cls(**cfg_loss)
    
    return loss
