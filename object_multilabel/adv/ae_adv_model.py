import torch
import torch.nn as nn
import functools
import torch.nn.functional as F
import torchvision.models as models
import torch.nn.utils
from torch.autograd import Function
import copy


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

# with skip connection and pixel connection and smoothed
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        use_bias = True
        # construct unet structure
        self.downsample_0 = nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1, bias=use_bias)

        self.downRelu_1 = nn.LeakyReLU(0.2, True)
        self.downSample_1 = nn.Conv2d(ngf, ngf * 2, kernel_size=4, stride=2, padding=1, bias=use_bias)
        self.downNorm_1 = norm_layer(ngf * 2)

        self.downRelu_2 = nn.LeakyReLU(0.2, True)
        self.downSample_2 = nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1, bias=use_bias)
        self.downNorm_2 = norm_layer(ngf * 4)

        self.downRelu_3 = nn.LeakyReLU(0.2, True)
        self.downSample_3 = nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias)
        self.downNorm_3 = norm_layer(ngf * 8)

        self.innerLeakyRelu = nn.LeakyReLU(0.2, True)
        self.innerDownSample = nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias)

        self.innerRelu = nn.ReLU(True)
        innerUpSample = []
        innerUpSample.append(nn.Upsample(scale_factor = 2, mode='bilinear'))
        innerUpSample.append(nn.ReflectionPad2d((2, 1, 2, 1)))
        innerUpSample.append(nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=1, padding=0, bias=use_bias))
        self.innerUpSample = nn.Sequential(*innerUpSample)

        self.innerNorm = norm_layer(ngf * 8)

        self.upRelu_3 = nn.ReLU(True)
        upSample_3 = []
        upSample_3.append(nn.Upsample(scale_factor = 2, mode='bilinear'))
        upSample_3.append(nn.ReflectionPad2d((2, 1, 2, 1)))
        upSample_3.append(nn.Conv2d(ngf * 16, ngf * 4, kernel_size=4, stride=1, padding=0, bias=use_bias))
        self.upSample_3 = nn.Sequential(*upSample_3)
        self.upNorm_3 = norm_layer(ngf * 4)

        self.upRelu_2 = nn.ReLU(True)
        upSample_2 = []
        upSample_2.append(nn.Upsample(scale_factor = 2, mode='bilinear'))
        upSample_2.append(nn.ReflectionPad2d((2, 1, 2, 1)))
        upSample_2.append(nn.Conv2d(ngf * 8, ngf * 2, kernel_size=4, stride=1, padding=0, bias=use_bias))
        self.upSample_2 = nn.Sequential(*upSample_2)
        self.upNorm_2 = norm_layer(ngf * 2)

        self.upRelu_1 = nn.ReLU(True)
        upSample_1 = []
        upSample_1.append(nn.Upsample(scale_factor = 2, mode='bilinear'))
        upSample_1.append(nn.ReflectionPad2d((2, 1, 2, 1)))
        upSample_1.append(nn.Conv2d(ngf * 4, ngf, kernel_size=4, stride=1, padding=0, bias=use_bias))
        self.upSample_1 = nn.Sequential(*upSample_1)
        self.upNorm_1 = norm_layer(ngf)

        self.upRelu_0 = nn.ReLU(True)
        upSample_0 = []
        upSample_0.append(nn.Upsample(scale_factor = 2, mode='bilinear'))
        upSample_0.append(nn.ReflectionPad2d((2, 1, 2, 1)))
        upSample_0.append(nn.Conv2d(ngf * 2, 1, kernel_size=4, stride=1, padding=0, bias=use_bias))
        self.upSample_0 = nn.Sequential(*upSample_0)

        ## initialize bias
        nn.init.normal_(self.upSample_0[-1].bias, mean=3, std=1)

        self.activation = nn.Sigmoid()

    def forward(self, input):
        # assume input image size = 224
        x_down_0 = self.downsample_0(input) # (ngf, 112, 112)

        x_down_1 = self.downNorm_1(self.downSample_1(self.downRelu_1(x_down_0))) # (ngf*2, 56, 56)
        x_down_2 = self.downNorm_2(self.downSample_2(self.downRelu_2(x_down_1))) # (ngf*4, 28, 28)
        x_down_3 = self.downNorm_3(self.downSample_3(self.downRelu_3(x_down_2))) # (ngf*8, 14, 14)

        latent = self.innerDownSample(self.innerLeakyRelu(x_down_3)) # (ngf*8, 7, 7)

        x = self.innerNorm(self.innerUpSample(self.innerRelu(latent))) # (ngf*8, 14, 14)

        x_up_3 = self.upNorm_3(self.upSample_3(self.upRelu_3(torch.cat([x, x_down_3], 1)))) # (ngf*4, 28, 28)
        x_up_2 = self.upNorm_2(self.upSample_2(self.upRelu_2(torch.cat([x_up_3, x_down_2], 1)))) # (ngf*2, 56, 56)
        x_up_1 = self.upNorm_1(self.upSample_1(self.upRelu_1(torch.cat([x_up_2, x_down_1], 1)))) # (ngf, 112, 112)

        encoded_image = self.activation(self.upSample_0(self.upRelu_0(torch.cat([x_up_1, x_down_0], 1)))) # (3, 224, 224)

        return torch.mul(input, encoded_image), latent

class ObjectMultiLabelAdv(nn.Module):

    def __init__(self, args, num_object, hid_size, dropout, adv_lambda):

        super(ObjectMultiLabelAdv, self).__init__()
        print("Build a ObjectMultiLabelAdv Model[{}]".format(args.layer))
        self.num_object = num_object
        self.args = args
        self.base_network = models.resnet50(pretrained = True)
        self.adv_lambda = adv_lambda
        print('Load weights from Resnet18/50 done')

        norm_layer = 'batch'
        use_dropout = False
        norm_layer = get_norm_layer(norm_type=norm_layer)
        self.autoencoder = UnetGenerator(3, 3, 5, 64, \
            norm_layer=norm_layer, use_dropout=use_dropout)

        output_size = self.num_object
        self.finalLayer = nn.Linear(self.base_network.fc.in_features, output_size)

        if not args.autoencoder_finetune:
            for param in self.autoencoder.parameters():
                param.requires_grad = False

        if not args.finetune:
            for param in self.base_network.parameters():
                param.requires_grad = False

            for param in self.finalLayer.parameters():
                param.requires_grad = False

        assert  args.layer == 'generated_image'
        self.adv_component = GenderClassification(args)
        pretrained_gender_classifier_path = './model_best_object_balanced.pth.tar'
        gender_clssifier_checkpoint = torch.load(pretrained_gender_classifier_path)
        self.adv_component.load_state_dict(gender_clssifier_checkpoint['state_dict'])
        print("Loaded pretrained gender classifier from {}".format(pretrained_gender_classifier_path))

        if not args.finetune:
            for param in self.adv_component.parameters():
                param.requires_grad = False

    def forward(self, image):

        autoencoded_image, latent = self.autoencoder(image)

        x = self.base_network.conv1(autoencoded_image)
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

        adv_feature = ReverseLayerF.apply(autoencoded_image, self.adv_lambda)
        adv_preds = self.adv_component(adv_feature)
        return preds, adv_preds, autoencoded_image

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class GenderClassification(nn.Module):

    def __init__(self, args):

        super(GenderClassification, self).__init__()
        print("Build a GenderClassification Model")

        self.base_network = models.resnet18(pretrained = True)
        print('Load weights from Resnet18 done')

        self.finalLayer = nn.Linear(self.base_network.fc.in_features, 2)

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
