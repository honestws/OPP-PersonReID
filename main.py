import random
import time
import numpy as np
import torch
from torch import nn
from torchvision.transforms import transforms, InterpolationMode
from tqdm import tqdm
from argpaser import argparse_option
from builder import create_model, create_optimizer, create_loader, create_continual_index_list, get_camera_person_info
from dreamer import DeepInversionDreamer
from evaluator import Evaluator
from lossfun import MixLoss
from trainer import Trainer
from tensorboardX import SummaryWriter
from util import RandomErasing, DataFolder, save_network, WeightEMA, load_network, TransformTwice, fuse_all_conv_bn
from pytorch_metric_learning import losses

if __name__ == '__main__':
    opt = argparse_option()
    torch.manual_seed(opt.seed)
    random.seed(opt.seed)
    np.random.seed(opt.seed)

    # create tensorboardx summary writer
    writer = SummaryWriter("log")

    # create and load teacher model
    model = create_model()
    ema_model = create_model(ema=True)
    load_network(ema_model)

    # create loss functions
    con_loss = losses.ContrastiveLoss()
    mix_loss = MixLoss()
    cross_entropy_loss = nn.CrossEntropyLoss()

    # create optimizer
    optimizer_sft = create_optimizer(opt, model, optimizer='cci')  # mix
    optimizer_ctr = create_optimizer(opt, model, optimizer='con')
    optimizer_ema = WeightEMA(opt, model, ema_model)

    # create image dataset
    train_transform = [
        transforms.Resize((256, 128), interpolation=InterpolationMode.BICUBIC),
        transforms.Pad(10),
        transforms.RandomCrop((256, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    if opt.erasing_p > 0:
        train_transform += [RandomErasing(probability=opt.erasing_p, mean=[0.0, 0.0, 0.0])]
    train_data_transform = transforms.Compose(train_transform)
    dream_transform = [
        transforms.Pad(5),
        transforms.RandomCrop((256, 128)),
        transforms.RandomHorizontalFlip(),
    ]
    dream_data_transform = transforms.Compose(dream_transform)
    _train_dataset = DataFolder(root=opt.data_folder + 'train_all', transform=TransformTwice(train_data_transform))
    # create continual index list
    continual_index_list = create_continual_index_list(opt, _train_dataset)
    print('The length of the continual index list is %d.' % len(continual_index_list))

    # create trainer
    tr = Trainer(opt, optimizer_ctr, optimizer_sft, optimizer_ema,
                 model, ema_model, writer, con_loss, mix_loss,
                 cross_entropy_loss, continual_index_list)

    # create dreamer
    dr = DeepInversionDreamer(opt, ema_model, cross_entropy_loss, dream_data_transform, writer)

    # create data loader
    dream_dataloader = None
    camera_person_list = []
    begin_time = time.time()
    for i, ith_indices in enumerate(continual_index_list):
        print('-----------------Processing the {}-th camera images-----------------'.format(i+1))
        train_dataloader = create_loader(opt, _train_dataset, ith_indices)
        camera_person, lab_dict = get_camera_person_info(_train_dataset, ith_indices)
        print('Number of camera person IDs is {}.'.format(camera_person))
        camera_person_list.append(camera_person)

        model.classifier.reset_classifier(camera_person)
        ema_model.classifier.reset_classifier(camera_person)

        for e in tqdm(range(1, 5*opt.epochs+1), desc='1. Training within camera view'):
            tr.train_within_camera_view(train_dataloader, e, i, lab_dict, camera_person_list)

        dream_dataloader = dr.dream_images()
        tr.create_feature_buffer(dream_dataloader)

        for e in tqdm(range(1, opt.epochs+1), desc='4. Training across camera view'):
            tr.train_across_camera_view(train_dataloader, dream_dataloader, e, i)
        print('Current output dimension is {}.'.format(ema_model.classifier.output_dim))
    end_time = time.time()
    run_time = round(end_time - begin_time)
    hour = run_time // 3600
    minute = (run_time - 3600 * hour) // 60
    second = run_time - 3600 * hour - 60 * minute
    print(f'Runing time：{hour}h-{minute}m-{second}s')

    # save network
    save_network(ema_model)
    # ema_model.classifier.reset_classifier(3541)
    ema_model = load_network(ema_model)
    ema_model = ema_model.eval()
    ema_model = fuse_all_conv_bn(ema_model)

    # evaluate
    eva = Evaluator(opt, ema_model)
    with torch.no_grad():
        query_feature = eva.extract_feature(dataloader='query')
        gallery_feature = eva.extract_feature(dataloader='gallery')
    eva.evaluate(query_feature, gallery_feature)
    writer.close()
