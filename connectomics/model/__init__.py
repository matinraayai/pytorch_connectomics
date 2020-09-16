import os,datetime
import torch
import torch.nn as nn
import numpy as np

from .zoo import *
from .norm import patch_replication_callback
from .utils import Monitor, Criterion

def build_model(cfg, device, checkpoint=None):
    MODEL_MAP = {'unet_residual_3d': unet_residual_3d,
                 'unet_residual_2d': unet_residual_2d,
                 'fpn': fpn,
                 'super':SuperResolution,
                 'unet_super':Unet_super}

    assert cfg.model.architecture in MODEL_MAP.keys()
    if cfg.model.architecture == 'super':
        model = MODEL_MAP[cfg.model.architecture](in_channel=cfg.model.in_planes, out_channel=cfg.model.out_planes, filters=cfg.model.filters)
    elif cfg.model.architecture == 'unet_residual_2d':
        model = MODEL_MAP[cfg.model.architecture](in_channel=cfg.model.in_planes, out_channel=cfg.model.out_planes, filters=cfg.model.filters, \
                                                  pad_mode=cfg.model.pad_mode, norm_mode=cfg.model.norm_mode, act_mode=cfg.model.activation_mode,
                                                  head_depth=cfg.model.head_depth)
    else:
        model = MODEL_MAP[cfg.model.architecture](in_channel=cfg.model.in_planes, out_channel=cfg.model.out_planes, filters=cfg.model.filters, \
                                                  pad_mode=cfg.model.pad_mode, norm_mode=cfg.model.norm_mode, act_mode=cfg.model.activation_mode,
                                                  do_embedding=(cfg.model.embedding == 1), head_depth=cfg.model.head_depth)

    print('model: ', model.__class__.__name__)
    model = nn.DataParallel(model, device_ids=range(cfg.system.num_gpus))
    patch_replication_callback(model)
    model = model.to(device)

    if checkpoint is not None:
        print('Load pretrained model: ', checkpoint)
        if cfg.model.exact:
            # exact matching: the weights shape in pretrain model and current model are identical
            weight = torch.load(checkpoint)
            # change channels if needed
            if cfg.model.PRE_MODEL_LAYER[0] != '':
                if cfg.model.PRE_MODEL_LAYER_SELECT[0]==-1: # replicate channels
                    for kk in cfg.model.PRE_MODEL_LAYER:
                        sz = list(np.ones(weight[kk][0:1].ndim,int))
                        sz[0] = cfg.model.model.out_planes
                        weight[kk] = weight[kk][0:1].repeat(sz)
                else: # select channels
                    for kk in cfg.model.PRE_MODEL_LAYER:
                        weight[kk] = weight[kk][cfg.model.PRE_MODEL_LAYER_SELECT]
            model.load_state_dict(weight)
        else:
            pretrained_dict = torch.load(cfg.model.PRE_MODEL)
            model_dict = model.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # 2. overwrite entries in the existing state dict 
            if cfg.model.SIZE_MATCH:
                model_dict.update(pretrained_dict) 
            else:
                for param_tensor in pretrained_dict:
                    if model_dict[param_tensor].size() == pretrained_dict[param_tensor].size():
                        model_dict[param_tensor] = pretrained_dict[param_tensor]       
            # 3. load the new state dict
            model.load_state_dict(model_dict)     
    
    return model

def build_monitor(cfg):
    time_now = str(datetime.datetime.now()).split(' ')
    date = time_now[0]
    time = time_now[1].split('.')[0].replace(':','-')
    log_path = os.path.join(cfg.dataset.output_path, 'log' + date + '_' + time)
    if not os.path.isdir(log_path):
        os.makedirs(log_path)
    return Monitor(log_path, cfg.MONITOR.LOG_OPT + [cfg.SOLVER.SAMPLES_PER_BATCH], \
                   cfg.MONITOR.VIS_OPT, cfg.MONITOR.ITERATION_NUM, cfg.dataset.do_2D)

def build_criterion(cfg, device):
    return Criterion(device, cfg.model.target_opt, cfg.model.loss_option, cfg.model.loss_weight, \
                     cfg.model.regu_opt, cfg.model.regu_weight)
