import os
import re

import numpy as np
import torch
from torch import optim
from torch.backends import cudnn
from torch.utils.data import Subset
from model import ResNet50
from util import list_split


def create_model(ema=False):
    model = ResNet50()
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        cudnn.benchmark = True
    if ema:
        for param in model.parameters():
            param.detach_()
    return model


def create_optimizer(opt, model, optimizer):
    ignored_params = list(map(id, model.head.parameters())) + \
                     list(map(id, model.classifier.classifier.parameters())) + \
                     list(map(id, model.encoder.fc.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    if optimizer == 'cci':
        incr_params = model.classifier.classifier.parameters()
        optimizer = optim.SGD([
            {'params': base_params, 'lr': 0.01 * opt.lr},
            {'params': incr_params, 'lr': opt.lr}],
            weight_decay=opt.weight_decay, momentum=opt.momentum, nesterov=True)
        return optimizer
    elif optimizer == 'con':
        cont_params = model.head.parameters()
        optimizer = optim.SGD([
            {'params': base_params, 'lr': 0.01 * opt.lr},
            {'params': cont_params, 'lr': opt.lr}],
            weight_decay=opt.weight_decay, momentum=opt.momentum, nesterov=True)
        return optimizer
    else:
        raise RuntimeError('Invalid optimizer type.')


def create_loader(opt, _train_dataset, ith_indices):
    train_dataset = Subset(_train_dataset, ith_indices)
    print('Current dataset size: {}'.format(len(train_dataset)))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True,
                                               num_workers=opt.num_workers, pin_memory=True)
    return train_loader


def get_camera_person_info(_train_dataset, ith_indices):
    labels = np.array(_train_dataset.targets)[ith_indices]
    targets, counts = np.unique(labels, return_counts=True)
    camera_person = len(targets)
    lab_dict = {}
    counter = 0
    for lab in labels:
        if lab not in lab_dict.keys():
            lab_dict[lab] = counter
            counter += 1

    return camera_person, lab_dict


def create_continual_index_list(dataset, _train_dataset):
    if dataset == 'Market-1501':
        # format: 0002_c1s1_000451_03.jpg
        # 6 cameras
        sequence_dict = {}
        for i, (img_path, t) in enumerate(_train_dataset.imgs):
            img_name = os.path.basename(img_path)
            ts = img_name.split('_')[1]
            if ts in sequence_dict.keys():
                sequence_dict[ts].append(i)
            else:
                sequence_dict[ts] = [i]
        continual_index_list = list(sequence_dict.values())
        return continual_index_list

    elif dataset == 'DukeMTMC':
        # format: 0315_c5_f0112364.jpg
        # 8 cameras
        sequence_dict = {}
        for i, (img_path, t) in enumerate(_train_dataset.imgs):
            img_name = os.path.basename(img_path)
            cs = img_name.split('_')[1]
            if cs in sequence_dict.keys():
                sequence_dict[cs].append(i)
            else:
                sequence_dict[cs] = [i]
        # split for data balance
        new_sequence_dict = {}
        split_index = 0
        for k, v in sequence_dict.items():
            if len(v) > 1600:
                new_sequence_dict, split_index = list_split(v, new_sequence_dict, split_index)
            else:
                new_sequence_dict[k] = v
        continual_index_list = list(new_sequence_dict.values())
        return continual_index_list
    elif dataset == 'MARS':
        # name formant: 0000C6T3036F006.jpg
        reg = r'T(.*?)F'
        sequence_dict = {}
        for i, (img_path, t) in enumerate(_train_dataset.imgs):
            img_name = os.path.basename(img_path)
            ts = re.findall(reg, img_name)[0]
            if ts in sequence_dict.keys():
                sequence_dict[ts].append(i)
            else:
                sequence_dict[ts] = [i]
        continual_index_list = list(sequence_dict.values())
        return continual_index_list
    elif dataset == 'MSMT17':
        pass
    else:
        raise RuntimeError("Invalid dataset name. Please select from {'Market-1501', 'DukeMTMC', 'MARS', 'MSMT17'}")
