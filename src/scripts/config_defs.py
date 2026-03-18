import os
import argparse
import yaml
from .auxiliary import avoid_MKL_bug

def cfg_from_args():
    """
    Function to get configs from the .yaml arguments
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the config file.")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found at: {args.config}")
    
    with open(args.config, 'r') as f:
        cfg= yaml.safe_load(f)
    
    return cfg

def parse_cfg_dict(cfg_run):

    """
    Parses the cfg dict

    Args
        cfg_run
    
    Returns
        cfg_data
        cfg_loader
        cfg_train
        cfg_export
        cfg_model_setup
        cfg_optim_setup
    """

    # Config parsing
    cfg_data = cfg_run['cfg_data']
    cfg_train = cfg_run['cfg_train']
    cfg_loader = cfg_train['cfg_loader']
    cfg_export = cfg_run['cfg_export']

    cfg_model_setup = cfg_run['cfg_model_setup']
    cfg_optim_setup = cfg_run['cfg_optim_setup']
    cfg_loss_setup = cfg_run['cfg_loss_setup']

    avoid_MKL_bug(cfg_model_setup['model_type'])
    # PyTorch is safe to import now

    import torch

    # Dtype config
    dtype_str = cfg_train['dtype']
    dtype_map = {
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
        'float32': torch.float32,
        'float64': torch.float64
    }

    if dtype_str not in dtype_map:
        raise ValueError(f"Unknown/Unsupported dtype: {dtype_str}. Choose one of: 'float16', 'bfloat16', 'float32', 'float64'")
    cfg_train['dtype'] = dtype_map[dtype_str]

    # Device config
    device_str = cfg_train['device']
    if device_str == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError("Requested CUDA but is not available.")
    elif device_str != 'cpu':
        raise ValueError(f"Unknown/Unsupported device: {device_str}. Choose one of: 'cuda', 'cpu'")
    cfg_train['device'] = torch.device(device_str)
    
    return cfg_data, cfg_loader, cfg_train, cfg_export, cfg_model_setup, cfg_optim_setup, cfg_loss_setup
