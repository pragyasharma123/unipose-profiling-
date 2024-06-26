import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm, dataset = None):
        super(Decoder, self).__init__()
        if backbone == 'resnet':
            low_level_inplanes = 256

        limbsNum = 13
        if dataset == "NTID":
            limbsNum = 18
        else:
            limbsNum = 13

        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = BatchNorm(48)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(2048, 256, 1, bias=False)
        self.bn2 = BatchNorm(256)
        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, num_classes+1, kernel_size=1, stride=1))
#                                        nn.Conv2d(256, num_classes+5+1, kernel_size=1, stride=1)) # Use in case of extacting the bounding box

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self._init_weight()


    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)
        
        #x = self.conv2(x)
        #x = self.bn2(x)
        #x = self.relu(x)

        low_level_feat = self.maxpool(low_level_feat)

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv(x)

        #x = self.maxpool(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def build_decoder(num_classes, backbone, BatchNorm, dataset=None):
    return Decoder(num_classes, backbone, BatchNorm, dataset)
