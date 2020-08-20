import os
import numpy as np
from scipy.ndimage import zoom

import torch
import torch.utils.data

from .dataset_volume import VolumeDataset
from .dataset_tile import TileDataset
from ..utils import collate_fn_target, collate_fn_test, seg_widen_border, readvol

__all__ = ['VolumeDataset', 'TileDataset', 'create_dataloader', 'create_dataset']


def _make_path_list(dir_name, file_name):
    """
    Concatenates directory path(s) and filenames and returns the complete file paths.
    """
    assert len(dir_name) == 1 or len(dir_name) == len(file_name)
    if len(dir_name) == 1:
        file_name = [os.path.join(dir_name[0], x) for x in file_name]
    else:
        file_name = [os.path.join(dir_name[i], file_name[i]) for i in range(len(file_name))]
    return file_name


def _get_input(cfg, mode='train'):
    dir_name = cfg.DATASET.INPUT_PATH.split('@')
    img_name = cfg.DATASET.IMAGE_NAME.split('@')
    img_name = _make_path_list(dir_name, img_name)

    label = None
    volume = [None] * len(img_name)
    if mode == 'train':
        label_name = cfg.DATASET.LABEL_NAME.split('@')
        assert len(label_name) == len(img_name)
        label_name = _make_path_list(dir_name, label_name)
        label = [None]*len(label_name)

    for i in range(len(img_name)):
        volume[i] = readvol(img_name[i])
        print(f"volume shape (original): {volume[i].shape}")
        if (np.array(cfg.DATASET.DATA_SCALE) != 1).any():
            volume[i] = zoom(volume[i], cfg.DATASET.DATA_SCALE, order=1) 
        volume[i] = np.pad(volume[i], ((cfg.DATASET.PAD_SIZE[0],cfg.DATASET.PAD_SIZE[0]), 
                                       (cfg.DATASET.PAD_SIZE[1],cfg.DATASET.PAD_SIZE[1]), 
                                       (cfg.DATASET.PAD_SIZE[2],cfg.DATASET.PAD_SIZE[2])), 'reflect')
        print(f"volume shape (after scaling and padding): {volume[i].shape}")

        if mode == 'train':
            label[i] = readvol(label_name[i])
            if (np.array(cfg.DATASET.DATA_SCALE) != 1).any():
                label[i] = zoom(label[i], cfg.DATASET.DATA_SCALE, order=0) 
            if cfg.DATASET.LABEL_EROSION != 0:
                label[i] = seg_widen_border(label[i], cfg.DATASET.LABEL_EROSION)
            if cfg.DATASET.LABEL_BINARY and label[i].max()>1:
                label[i] = label[i] // 255
            if cfg.DATASET.LABEL_MAG != 0:
                label[i] = (label[i] / cfg.DATASET.LABEL_MAG).astype(np.float32)
                
            label[i] = np.pad(label[i], ((cfg.DATASET.PAD_SIZE[0],cfg.DATASET.PAD_SIZE[0]), 
                                         (cfg.DATASET.PAD_SIZE[1],cfg.DATASET.PAD_SIZE[1]), 
                                         (cfg.DATASET.PAD_SIZE[2],cfg.DATASET.PAD_SIZE[2])), 'reflect')
            print(f"label shape: {label[i].shape}")
                 
    return volume, label


def create_dataset(cfg, augmentor, mode='train'):
    """
    Creates a Pytorch dataset for training and inference using the configuration and augmentor.
    Args:
        cfg: YACS Configuration object specifications
        augmentor:
        mode: Either 'train' or 'test'
    Return:
        a Pytorch Dataset
    """
    assert mode in ['train', 'test']

    label_erosion = 0
    sample_label_size = cfg.MODEL.OUTPUT_SIZE
    sample_invalid_thres = cfg.DATASET.DATA_INVALID_THRES
    t_opt, w_opt = -1, -1
    if mode == 'train':
        sample_volume_size = augmentor.sample_size
        sample_label_size = sample_volume_size
        label_erosion = cfg.DATASET.LABEL_EROSION
        sample_stride = (1, 1, 1)
        t_opt, w_opt = cfg.MODEL.TARGET_OPT, cfg.MODEL.WEIGHT_OPT
        iter_num = cfg.SOLVER.ITERATION_TOTAL * cfg.SOLVER.SAMPLES_PER_BATCH 
    elif mode == 'test':
        sample_stride = cfg.INFERENCE.STRIDE
        sample_volume_size = cfg.MODEL.INPUT_SIZE
        iter_num = -1
      
    # dataset
    if cfg.DATASET.DO_CHUNK_TITLE == 1:
        label_json = cfg.DATASET.INPUT_PATH + cfg.DATASET.LABEL_NAME if mode == 'train' else ''
        dataset = TileDataset(chunk_num=cfg.DATASET.DATA_CHUNK_NUM, chunk_num_ind=cfg.DATASET.DATA_CHUNK_NUM_IND,
                              chunk_iter=cfg.DATASET.DATA_CHUNK_ITER, chunk_stride=cfg.DATASET.DATA_CHUNK_STRIDE,
                              volume_json=cfg.DATASET.INPUT_PATH + cfg.DATASET.IMAGE_NAME, label_json=label_json,
                              sample_volume_size=sample_volume_size, sample_label_size=sample_label_size,
                              sample_stride=sample_stride, sample_invalid_thres=sample_invalid_thres,
                              augmentor=augmentor, target_opt=t_opt, weight_opt=w_opt, mode=mode, do_2d=cfg.DATASET.DO_2D,
                              iter_num=iter_num, label_erosion=label_erosion, pad_size=cfg.DATASET.PAD_SIZE)

    else:
        if cfg.DATASET.PRE_LOAD_DATA[0] is None:  # load from cfg
            volume, label = _get_input(cfg, mode=mode)
        else:
            volume, label = cfg.DATASET.PRE_LOAD_DATA

        dataset = VolumeDataset(volume=volume, 
                                label=label, 
                                sample_volume_size=sample_volume_size, 
                                sample_label_size=sample_label_size,
                                sample_stride=sample_stride, 
                                sample_invalid_thres=sample_invalid_thres, 
                                augmentor=augmentor, 
                                target_opt=t_opt,
                                weight_opt=w_opt,
                                mode=mode,
                                do_2d=cfg.DATASET.DO_2D,
                                iter_num=iter_num,
                                # Specify options for rejection sampling:
                                reject_size_thres=cfg.DATASET.REJECT_SAMPLING.SIZE_THRES, 
                                reject_after_aug=cfg.DATASET.REJECT_SAMPLING.AFTER_AUG,
                                reject_p=cfg.DATASET.REJECT_SAMPLING.P)

    return dataset


def create_dataloader(cfg, augmentor, mode='train', dataset=None):
    """
    Prepare dataloader for training and inference.
    """
    assert mode in ['train', 'test']

    shuffle = mode == 'train'

    if mode == 'train':
        cf = collate_fn_target
        batch_size = cfg.SOLVER.SAMPLES_PER_BATCH
    else:
        cf = collate_fn_test
        batch_size = cfg.INFERENCE.SAMPLES_PER_BATCH

    if dataset is None:
        dataset = create_dataset(cfg, augmentor, mode)
    
    data_loader = torch.utils.data.DataLoader(
          dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=cf,
          num_workers=cfg.SYSTEM.NUM_CPUS, pin_memory=True)

    return data_loader
