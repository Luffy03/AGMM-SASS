import torch
from torch import nn
import torch.nn.functional as F
import model.backbone.resnet as resnet


class DeepLabV3Plus(nn.Module):
    def __init__(self, cfg, aux=True):
        super(DeepLabV3Plus, self).__init__()

        self.backbone = \
            resnet.__dict__[cfg['backbone']](True, multi_grid=cfg['multi_grid'],
                                             replace_stride_with_dilation=cfg['replace_stride_with_dilation'])

        low_channels = 256
        high_channels = 2048

        self.aux = aux

        self.head = ASPPModule(high_channels, cfg['dilations'])

        self.reduce = nn.Sequential(nn.Conv2d(low_channels, 48, 1, bias=False),
                                    nn.BatchNorm2d(48),
                                    nn.ReLU(True))

        self.fuse = nn.Sequential(nn.Conv2d(high_channels // 8 + 48, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True),
                                  nn.Conv2d(256, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True))

        self.classifier = nn.Conv2d(256, cfg['nclass'], 1, bias=True)
        if self.aux:
            self.aux_classifier = Aux_Module(high_channels, cfg['nclass'])

    def forward(self, x):
        h, w = x.shape[-2:]
        f1, f2, f3, f4 = self.backbone.base_forward(x)

        c4 = self.head(f4)
        c4 = F.interpolate(c4, size=f1.shape[-2:], mode="bilinear", align_corners=True)
        c1 = self.reduce(f1)
        feat = torch.cat([c1, c4], dim=1)
        feat = self.fuse(feat)

        out = self.classifier(feat)
        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=True)

        if self.aux:
            aux_feat, aux_out = self.aux_classifier(f4)
            aux_out = F.interpolate(aux_out, size=(h, w), mode="bilinear", align_corners=True)
            if self.training:
                return aux_feat, aux_out, feat, out

            else:
                aux_out = F.softmax(aux_out, dim=1)
                out = F.softmax(out, dim=1)
                return aux_out * 0.4 + out * 0.6
        else:
            if self.training:
                return feat, out
            else:
                return out.softmax(1)

    def forward_feat(self, x):
        h, w = x.shape[-2:]
        f1, f2, f3, f4 = self.backbone.base_forward(x)

        c4 = self.head(f4)
        c4 = F.interpolate(c4, size=f1.shape[-2:], mode="bilinear", align_corners=True)
        c1 = self.reduce(f1)
        feat = torch.cat([c1, c4], dim=1)
        feat = self.fuse(feat)

        out = self.classifier(feat)
        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=True)

        if self.aux:
            aux_feat, aux_out = self.aux_classifier(f4)
            aux_out = F.interpolate(aux_out, size=(h, w), mode="bilinear", align_corners=True)
            return aux_feat, aux_out, feat, out
        else:
            return feat, out


def ASPPConv(in_channels, out_channels, atrous_rate):
    block = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate,
                                    dilation=atrous_rate, bias=False),
                          nn.BatchNorm2d(out_channels),
                          nn.ReLU(True))
    return block


class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__()
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU(True))

    def forward(self, x):
        h, w = x.shape[-2:]
        pool = self.gap(x)
        return F.interpolate(pool, (h, w), mode="bilinear", align_corners=True)


class ASPPModule(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPPModule, self).__init__()
        out_channels = in_channels // 8
        rate1, rate2, rate3 = atrous_rates

        self.b0 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                nn.BatchNorm2d(out_channels),
                                nn.ReLU(True))
        self.b1 = ASPPConv(in_channels, out_channels, rate1)
        self.b2 = ASPPConv(in_channels, out_channels, rate2)
        self.b3 = ASPPConv(in_channels, out_channels, rate3)
        self.b4 = ASPPPooling(in_channels, out_channels)

        self.project = nn.Sequential(nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
                                     nn.BatchNorm2d(out_channels),
                                     nn.ReLU(True),
                                     nn.Dropout2d(0.5, False))

    def forward(self, x):
        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        feat4 = self.b4(x)
        y = torch.cat((feat0, feat1, feat2, feat3, feat4), 1)
        return self.project(y)


class Aux_Module(nn.Module):
    def __init__(self, high_level_channels, nclass):
        super(Aux_Module, self).__init__()
        self.head = ASPPModule(high_level_channels, (12, 24, 36))
        self.classifier = nn.Conv2d(256, nclass, 1, bias=True)

    def forward(self, f4):
        feat = self.head(f4)
        out = self.classifier(feat)
        return feat, out




