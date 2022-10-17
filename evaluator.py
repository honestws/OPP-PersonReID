import math
import os
import numpy as np
import torch
from torch import nn
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from util import DataFolder, fliplr, compute_mAP


class Evaluator(object):
    def __init__(self, opt, ema_model):
        self.opt = opt
        self.msc = []
        for s in self.opt.msc:
            s_f = float(s)
            self.msc.append(math.sqrt(s_f))
        data_transforms = transforms.Compose([
            transforms.Resize((256, 128), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        if opt.dataset == 'MARS':
            image_datasets = {x: DataFolder(
                os.path.join(opt.data_folder + 'pytorch', x), data_transforms) for x in ['gallery', 'query']}
        else:
            image_datasets = {x: DataFolder(
                os.path.join(opt.data_folder, x), data_transforms) for x in ['gallery', 'query']}

        gallery_path = image_datasets['gallery'].imgs
        query_path = image_datasets['query'].imgs

        self.gallery_cam, self.gallery_label = self.get_id(gallery_path)
        self.query_cam, self.query_label = self.get_id(query_path)

        self.dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batch_size, shuffle=False,
                                                           num_workers=opt.num_workers) for x in ['gallery', 'query']}
        self.ema_model = ema_model

    def extract_feature(self, dataloader='gallery'):
        for iter, data in enumerate(self.dataloaders[dataloader]):
            img, label, _ = data
            n, c, h, w = img.size()
            ff = torch.FloatTensor(n, 512).zero_().cuda()

            for i in range(2):
                if i == 1:
                    img = fliplr(img)
                input_img = img.cuda()
                for scale in self.msc:
                    if scale != 1:
                        # bicubic is only available in pytorch >= 1.1
                        input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bicubic',
                                                              align_corners=False)
                    _, _, outputs = self.ema_model(input_img)
                    ff += outputs
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))

            if iter == 0:
                features = torch.FloatTensor(len(self.dataloaders[dataloader].dataset), ff.shape[1])
            # features = torch.cat((features,ff.data.cpu()), 0)
            start = iter * self.opt.batch_size
            end = min((iter + 1) * self.opt.batch_size, len(self.dataloaders[dataloader].dataset))
            features[start:end, :] = ff
        return features

    def mAP(self, qf, ql, qc, gf, gl, gc):
        query = qf.view(-1, 1)
        # print(query.shape)
        score = torch.mm(gf, query)
        score = score.squeeze(1).cpu()
        score = score.numpy()
        # predict index
        index = np.argsort(score)  # from small to large
        index = index[::-1]
        # index = index[0:2000]
        # good index
        query_index = np.argwhere(gl == ql)
        camera_index = np.argwhere(gc == qc)

        good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
        junk_index1 = np.argwhere(gl == -1)
        junk_index2 = np.intersect1d(query_index, camera_index)
        junk_index = np.append(junk_index2, junk_index1)  # .flatten())
        return compute_mAP(index, good_index, junk_index)

    def get_id(self, img_path):
        camera_id = []
        labels = []
        for path, v in img_path:
            filename = os.path.basename(path)
            label = filename[0:4]
            if self.opt.dataset == 'MARS':
                camera = filename.split('C')[1]
                if label[2:] == '-1':
                    labels.append(-1)
                else:
                    labels.append(int(label))
                camera_id.append(int(camera[0]))
            else:
                camera = filename.split('c')[1]
                if label[0:2] == '-1':
                    labels.append(-1)
                else:
                    labels.append(int(label))
                camera_id.append(int(camera[0]))
        return np.array(camera_id), np.array(labels)

    def evaluate(self, query_feature, gallery_feature):
        query_feature = query_feature.cuda()
        gallery_feature = gallery_feature.cuda()

        CMC = torch.IntTensor(len(self.gallery_label)).zero_()
        ap = 0.0
        # print(query_label)
        for i in range(len(self.query_label)):
            ap_tmp, CMC_tmp = self.mAP(query_feature[i], self.query_label[i], self.query_cam[i],
                                       gallery_feature, self.gallery_label, self.gallery_cam)
            if CMC_tmp[0] == -1:
                continue
            CMC = CMC + CMC_tmp
            ap += ap_tmp

        CMC = CMC.float()
        CMC = CMC / len(self.query_label)
        print('Rank@1: %f Rank@5: %f Rank@10: %f Rank@20: %f mAP: %f' %
              (CMC[0], CMC[4], CMC[9], CMC[19], ap / len(self.query_label)))
