import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.nn.utils
from torch.autograd import Function
import copy

class ObjectMultiLabelAdv(nn.Module):

    def __init__(self, args, num_object, hid_size, dropout, adv_lambda):

        super(ObjectMultiLabelAdv, self).__init__()
        print("Build a ObjectMultiLabelAdv Model[{}]".format(args.layer))

        self.num_object = num_object
        self.args = args
        self.base_network = models.resnet50(pretrained = True)
        print('Load weights from Resnet18/50 done')

        if not args.finetune:
            for param in self.base_network.parameters():
                param.requires_grad = False

        output_size = self.num_object
        self.finalLayer = nn.Linear(self.base_network.fc.in_features, output_size)

        if args.layer == 'conv4':
            if args.adv_conv:
                self.adv_layer4 = copy.deepcopy(self.base_network.layer4)
                self.avgpool = copy.deepcopy(self.base_network.avgpool)
            elif not args.no_avgpool: self.avgpool = nn.AvgPool2d(14, stride=1)
        elif args.layer == 'conv3':
            if args.adv_conv:
                self.adv_layer4 = copy.deepcopy(self.base_network.layer4)
                self.adv_layer3 = copy.deepcopy(self.base_network.layer3)
                self.avgpool = copy.deepcopy(self.base_network.avgpool)
            elif not args.no_avgpool: self.avgpool = nn.AvgPool2d(28, stride=1)
        elif args.layer == 'conv2':
            if args.adv_conv:
                self.adv_layer4 = copy.deepcopy(self.base_network.layer4)
                self.adv_layer3 = copy.deepcopy(self.base_network.layer3)
                self.adv_layer2 = copy.deepcopy(self.base_network.layer2)
                self.avgpool = copy.deepcopy(self.base_network.avgpool)
            elif not args.no_avgpool: self.avgpool = nn.AvgPool2d(56, stride=1)
        elif args.layer == 'conv1':
            if args.adv_conv:
                self.adv_layer4 = copy.deepcopy(self.base_network.layer4)
                self.adv_layer3 = copy.deepcopy(self.base_network.layer3)
                self.adv_layer2 = copy.deepcopy(self.base_network.layer2)
                self.adv_layer1 = copy.deepcopy(self.base_network.layer1)
                self.avgpool = copy.deepcopy(self.base_network.avgpool)
            elif not args.no_avgpool: self.avgpool = nn.AvgPool2d(112, stride=1)
        elif args.layer == 'conv5':
            if args.adv_conv:
                adv_convs = []
                adv_convs.append(nn.Conv2d(2048, 1024, kernel_size=1, stride=1, bias=False))
                adv_convs.append(nn.Conv2d(1024, 512, kernel_size=1, stride=1, bias=False))
                adv_convs.append(nn.Conv2d(512, 512, kernel_size=3, stride=1, \
                    padding=1, bias=False))
                adv_convs.append(nn.Conv2d(512, 1024, kernel_size=1, stride=1, bias=False))
                adv_convs.append(nn.Conv2d(1024, 2048, kernel_size=1, stride=1, bias=False))
                self.adv_convs = nn.Sequential(*adv_convs)
            if not args.no_avgpool: self.avgpool = nn.AvgPool2d(7, stride=1)

        adv_mlp = []
        if args.layer == 'conv5':
            if args.adv_conv:
                adv_mlp.append(nn.Linear(2048, hid_size, bias=True))
            elif args.no_avgpool:
                adv_mlp.append(nn.Linear(2048 * 7 * 7, hid_size, bias=True))
            else:
                adv_mlp.append(nn.Linear(2048, hid_size, bias=True))

        if args.layer == 'conv4':
            if args.adv_conv:
                adv_mlp.append(nn.Linear(2048, hid_size, bias=True))
            elif args.no_avgpool:
                adv_mlp.append(nn.Linear(1024 * 14 * 14, hid_size, bias=True))
            else:
                adv_mlp.append(nn.Linear(1024, hid_size, bias=True))

        if args.layer == 'conv3':
            if args.adv_conv:
                adv_mlp.append(nn.Linear(2048, hid_size, bias=True))
            elif args.no_avgpool:
                adv_mlp.append(nn.Linear(512 * 28 * 28, hid_size, bias=True))
            else:
                adv_mlp.append(nn.Linear(512, hid_size, bias=True))

        if args.layer == 'conv2':
            if args.adv_conv:
                adv_mlp.append(nn.Linear(2048, hid_size, bias=True))
            elif args.no_avgpool:
                adv_mlp.append(nn.Linear(256 * 56 * 56, hid_size, bias=True))
            else:
                adv_mlp.append(nn.Linear(256, hid_size, bias=True))

        # mlp.append(nn.Dropout(p=dropout))
        adv_mlp.append(nn.BatchNorm1d(hid_size))
        adv_mlp.append(nn.LeakyReLU())

        adv_mlp.append(nn.Linear(hid_size, hid_size, bias=True))
        adv_mlp.append(nn.BatchNorm1d(hid_size))
        adv_mlp.append(nn.LeakyReLU())

        adv_mlp.append(nn.Linear(hid_size, hid_size, bias=True))
        adv_mlp.append(nn.BatchNorm1d(hid_size))
        adv_mlp.append(nn.LeakyReLU())

        adv_mlp.append(nn.Linear(hid_size, 2, bias=True))
        self.adv_mlp = nn.Sequential(*adv_mlp)
        self.adv_lambda = adv_lambda

    def forward(self, image):
        x = self.base_network.conv1(image)
        x = self.base_network.bn1(x)
        x = self.base_network.relu(x)
        conv1_feature = self.base_network.maxpool(x)

        layer1_feature = self.base_network.layer1(conv1_feature)
        layer2_feature = self.base_network.layer2(layer1_feature)
        layer3_feature = self.base_network.layer3(layer2_feature)
        layer4_feature = self.base_network.layer4(layer3_feature)

        final_feature = self.base_network.avgpool(layer4_feature)
        final_feature = final_feature.view(final_feature.size(0), -1)

        preds = self.finalLayer(final_feature)

        # adv component forward pass
        if self.args.layer == 'conv5':
            if self.args.adv_on:
                    adv_feature = ReverseLayerF.apply(layer4_feature, self.adv_lambda)
            else:
                adv_feature = layer4_feature
            if self.args.adv_conv:
                adv_feature = self.adv_convs(adv_feature)

        elif self.args.layer == 'conv4':
            if self.args.adv_on:
                adv_feature = ReverseLayerF.apply(layer3_feature, self.adv_lambda)
            else:
                adv_feature = layer3_feature
            if self.args.adv_conv:
                adv_feature = self.adv_layer4(adv_feature)

        elif self.args.layer == 'conv3':
            if self.args.adv_on:
                adv_feature = ReverseLayerF.apply(layer2_feature, self.adv_lambda)
            else:
                adv_feature = layer2_feature
            if self.args.adv_conv:
                adv_feature = self.adv_layer3(adv_feature)
                adv_feature = self.adv_layer4(adv_feature)

        elif self.args.layer == 'conv2':
            if self.args.adv_on:
                adv_feature = ReverseLayerF.apply(layer1_feature, self.adv_lambda)
            else:
                adv_feature = layer1_feature
            if self.args.adv_conv:
                adv_feature = self.adv_layer2(adv_feature)
                adv_feature = self.adv_layer3(adv_feature)
                adv_feature = self.adv_layer4(adv_feature)

        elif self.args.layer == 'conv1':
            if self.args.adv_on:
                adv_feature = ReverseLayerF.apply(conv1_feature, self.adv_lambda)
            else:
                adv_feature = conv1_feature
            if self.args.adv_conv:
                adv_feature = self.adv_layer1(adv_feature)
                adv_feature = self.adv_layer2(adv_feature)
                adv_feature = self.adv_layer3(adv_feature)
                adv_feature = self.adv_layer4(adv_feature)

        if self.args.no_avgpool:
            adv_feature = adv_feature.view(-1, adv_feature.size(1), adv_feature.size(2)*adv_feature.size(3))
            adv_feature = adv_feature.view(-1, adv_feature.size(1)*adv_feature.size(2))

        else:
            adv_feature = self.avgpool(adv_feature)
            adv_feature = adv_feature.view(adv_feature.size(0), -1)

        adv_preds = self.adv_mlp(adv_feature)

        return preds, adv_preds

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class GenderClassifier(nn.Module):

    def __init__(self, args, num_object):

        super(GenderClassifier, self).__init__()
        print("Build a GenderClassifier Model")

        hid_size = args.hid_size
        # dropout = 0.1

        mlp = []
        mlp.append(nn.BatchNorm1d(num_object))
        mlp.append(nn.Linear(num_object, hid_size, bias=True))
        # mlp.append(nn.Dropout(p=dropout))

        mlp.append(nn.BatchNorm1d(hid_size))
        mlp.append(nn.LeakyReLU())
        mlp.append(nn.Linear(hid_size, hid_size, bias=True))
        # mlp.append(nn.Dropout(p=dropout))

        mlp.append(nn.BatchNorm1d(hid_size))
        mlp.append(nn.LeakyReLU())
        mlp.append(nn.Linear(hid_size, hid_size, bias=True))
        # mlp.append(nn.Dropout(p=dropout))

        mlp.append(nn.BatchNorm1d(hid_size))
        mlp.append(nn.LeakyReLU())
        mlp.append(nn.Linear(hid_size, 2, bias=True))
        # mlp.append(nn.Dropout(p=dropout))

        self.mlp = nn.Sequential(*mlp)

    def forward(self, input):
        return self.mlp(input)
