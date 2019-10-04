import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.nn.utils
import numpy as np

class ObjectMultiLabel(nn.Module):

    def __init__(self, args, num_object):

        super(ObjectMultiLabel, self).__init__()
        print("Build a ObjectMultiLabel Model")
        self.num_object = num_object

        self.base_network = models.resnet50(pretrained = True)

        if not args.finetune:
            for param in self.base_network.parameters():
                param.requires_grad = False

        output_size = self.num_object
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

        x = self.base_network.avgpool(x)
        image_features = x.view(x.size(0), -1)

        preds = self.finalLayer(image_features)

        return preds

class ObjectMultiLabelEncoder(ObjectMultiLabel):

    def __init__(self, args, num_object):
        ObjectMultiLabel.__init__(self, args, num_object)
        self.args = args

    def forward(self, image):
        x = self.base_network.conv1(image)
        x = self.base_network.bn1(x)
        x = self.base_network.relu(x)
        x = self.base_network.maxpool(x)

        x = self.base_network.layer1(x)
        x = self.base_network.layer2(x)
        x = self.base_network.layer3(x)
        x = self.base_network.layer4(x)

        x = self.base_network.avgpool(x)
        image_features = x.view(x.size(0), -1)

        if self.args.noise:
            image_features += torch.Tensor(np.random.normal(loc=0, scale=self.args.noise_scale, \
                size=image_features.shape)).cuda()

        preds = self.finalLayer(image_features)

        return image_features, preds

class GenderClassifier(nn.Module):

    def __init__(self, args, num_object):

        super(GenderClassifier, self).__init__()
        hid_size = args.hid_size

        mlp = []
        mlp.append(nn.BatchNorm1d(num_object))
        mlp.append(nn.Linear(num_object, hid_size, bias=True))

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

    def forward(self, input):
        return self.mlp(input)
