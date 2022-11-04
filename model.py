import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torchvision import models


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
    if hasattr(m, 'bias') and m.bias is not None:
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


class IncrementalBlock(nn.Module):
    def __init__(self, input_dim, output_dim, drop=0.5, relu=False, bnorm=True, mid_dim=512):
        super(IncrementalBlock, self).__init__()
        add_block = []
        if mid_dim > 0:
            add_block += [nn.Linear(input_dim, mid_dim)]
        else:
            mid_dim = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(mid_dim)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if drop > 0:
            add_block += [nn.Dropout(p=drop)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        inc_classifier = []
        inc_classifier += [nn.Linear(mid_dim, output_dim)]
        inc_classifier = nn.Sequential(*inc_classifier)
        inc_classifier.apply(weights_init_classifier)
        self.add_block = add_block
        self.classifier = inc_classifier
        self.mid_dim = mid_dim
        self.output_dim = output_dim

    def forward(self, x):
        x = self.add_block(x)
        f = x
        x = self.classifier(x)
        return x, f

    def reset_classifier(self, num_new_ids):
        past_weight = self.classifier[0].weight.data.clone().detach()
        past_bias = self.classifier[0].bias.data.clone().detach()
        past_output_dim = len(past_bias)
        self.output_dim = num_new_ids + past_output_dim
        inc_classifier = []
        inc_classifier += [nn.Linear(self.mid_dim, self.output_dim)]
        inc_classifier = nn.Sequential(*inc_classifier)
        inc_classifier.apply(weights_init_classifier)
        self.classifier = inc_classifier.cuda()
        if past_output_dim > 0:
            self.classifier[0].weight.data[:past_output_dim, :] = past_weight
            self.classifier[0].bias.data[:past_output_dim] = past_bias


# Define the ResNet50 Model for the OPP-PersonReID experiment setup
class ResNet50(nn.Module):
    def __init__(self, stride=1, feat_dim=128):
        super(ResNet50, self).__init__()
        model = models.resnet50(pretrained=False)
        if stride == 1:
            model.layer4[0].downsample[0].stride = (1, 1)
            model.layer4[0].conv2.stride = (1, 1)
        model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.encoder = model
        self.head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, feat_dim)
        )
        self.classifier = IncrementalBlock(2048, 0)

    def forward(self, x):
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)
        x = self.encoder.layer1(x)
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)
        x = self.encoder.layer4(x)
        x = self.encoder.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        logit, feat = self.classifier(x)
        z = F.normalize(self.head(x), dim=1)
        # z for contrastive learning; dim=128
        # logit for incremental soft-label learning; dim++
        # feat for the Re-ID evaluation and feature buffer; dim=512
        return z, logit, feat


if __name__ == '__main__':
    resnet = ResNet50()
    inputs = torch.randn((2, 3, 256, 128))
    outputs = resnet(inputs)
    pass
