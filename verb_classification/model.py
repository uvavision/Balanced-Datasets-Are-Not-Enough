import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.nn.utils

import numpy as np

class VerbClassification(nn.Module):

    def __init__(self, args, num_verb):

        super(VerbClassification, self).__init__()
        print("Build a VerbClassification Model")
        self.num_verb = num_verb

        self.base_network = models.resnet50(pretrained = True)
        print('Load weights from Resnet18/50 done')

        if not args.finetune:
            for param in self.base_network.parameters():
                param.requires_grad = False

        output_size = self.num_verb
        self.finalLayer = nn.Linear(self.base_network.fc.in_features, output_size)

    def forward(self, image):
        x = self.base_network.conv1(image)
        x = self.base_network.bn1(x)
        x = self.base_network.relu(x)
        x = self.base_network.maxpool(x)

        x = self.base_network.layer1(x)
        x = self.base_network.layer2(x)
        x = self.base_network.layer3(x)
        x = self.base_network.layer4(x)

        # avg pool or max pool
        x = self.base_network.avgpool(x)
        image_features = x.view(x.size(0), -1)

        preds = self.finalLayer(image_features)

        return preds


class GenderClassifier(nn.Module):
    def __init__(self, args, num_verb):
        super(GenderClassifier, self).__init__()
        print('Build a GenderClassifier Model')

        hid_size = args.hid_size

        mlp = []
        mlp.append(nn.BatchNorm1d(num_verb))
        mlp.append(nn.Linear(num_verb, hid_size, bias=True))

        mlp.append(nn.BatchNorm1d(hid_size))
        mlp.append(nn.LeakyReLU())
        mlp.append(nn.Linear(hid_size, hid_size, bias=True))

        mlp.append(nn.BatchNorm1d(hid_size))
        mlp.append(nn.LeakyReLU())
        mlp.append(nn.Linear(hid_size, hid_size, bias=True))

        mlp.append(nn.BatchNorm1d(hid_size))
        mlp.append(nn.LeakyReLU())
        mlp.append(nn.Linear(hid_size, hid_size, bias=True))

        mlp.append(nn.BatchNorm1d(hid_size))
        mlp.append(nn.LeakyReLU())
        mlp.append(nn.Linear(hid_size, 2, bias=True))
        self.mlp = nn.Sequential(*mlp)

    def forward(self, input_rep):

        return self.mlp(input_rep)
