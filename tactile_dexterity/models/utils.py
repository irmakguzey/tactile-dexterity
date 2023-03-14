import hydra
import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from torch.nn.parallel import DistributedDataParallel as DDP

# Taken from https://github.com/SridharPandian/Holo-Dex/blob/main/holodex/utils/models.py
def create_fc(input_dim, output_dim, hidden_dims, use_batchnorm=False, dropout=None, is_moco=False):
    if hidden_dims is None:
        return nn.Sequential(*[nn.Linear(input_dim, output_dim)])

    layers = [nn.Linear(input_dim, hidden_dims[0]), nn.ReLU()]

    if use_batchnorm:
        layers.append(nn.BatchNorm1d(hidden_dims[0]))

    if dropout is not None:
        layers.append(nn.Dropout(p = dropout))

    for idx in range(len(hidden_dims) - 1):
        layers.append(nn.Linear(hidden_dims[idx], hidden_dims[idx + 1]))
        layers.append(nn.ReLU())

        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden_dims[idx + 1]))

        if dropout is not None:
            layers.append(nn.Dropout(p = dropout))

    layers.append(nn.Linear(hidden_dims[-1], output_dim))
    if is_moco:
        layers.append(nn.BatchNorm1d(hidden_dims[-1], affine=False))
    return nn.Sequential(*layers)


def load_model(cfg, device, model_path, bc_model_type=None):
    # Initialize the model
    if cfg.learner_type == 'bc':
        if bc_model_type == 'image':
            model = hydra.utils.instantiate(cfg.encoder.image_encoder)
        elif bc_model_type == 'tactile':
            model = hydra.utils.instantiate(cfg.encoder.tactile_encoder)
        elif bc_model_type == 'last_layer':
            model = hydra.utils.instantiate(cfg.encoder.last_layer)

    elif 'byol' in cfg.learner_type: # load the encoder
        model = hydra.utils.instantiate(cfg.encoder)  

    state_dict = torch.load(model_path)
    
    # Modify the state dict accordingly - this is needed when multi GPU saving was done
    new_state_dict = modify_multi_gpu_state_dict(state_dict)
    
    if 'byol' in cfg.learner_type:
        new_state_dict = modify_byol_state_dict(new_state_dict)

    # Load the new state dict to the model 
    model.load_state_dict(new_state_dict)

    # Turn it into DDP - it was saved that way 
    model = DDP(model.to(device), device_ids=[0])

    return model

def modify_multi_gpu_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v 
    return new_state_dict

def modify_byol_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'encoder.net' in k:
            name = k[12:] # Everything after encoder.net
            new_state_dict[name] = v
    return new_state_dict
