import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision.utils import save_image
from torch.utils.data import DataLoader

import math, os, random, json, pickle, sys, pdb
import string, shutil, time, argparse, operator, collections
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import functools

from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from tqdm import tqdm as tqdm

from data_loader import CocoObjectGender
from ae_adv_model import UnetGenerator, get_norm_layer, ObjectMultiLabelAdv

object_id_map = pickle.load(open('../data/object_id.map'))
object2id = object_id_map['object2id']
id2object = object_id_map['id2object']

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225])
transf = transforms.Compose([UnNormalize(normalize.mean, normalize.std),
                             transforms.ToPILImage()])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', type=str,
            help='experiment id, e.g. conv4_300_1.0_0.2_1')

    parser.add_argument('--annotation_dir', type=str,
            default='../data',
            help='annotation files path')
    parser.add_argument('--image_dir',
            default = '../data',
            help='image directory')

    parser.add_argument('--num_object', type=int,
            default = 79)

    parser.add_argument('--no_image', action='store_true')

    parser.add_argument('--gender_balanced', action='store_true',
            help='use gender balanced subset for training')
    parser.add_argument('--balanced', action='store_true',
            help='use balanced subset for training')
    parser.add_argument('--ratio', type=str,
            default = '0')

    parser.add_argument('--adv_on', action='store_true',
            help='start adv training')
    parser.add_argument('--layer', type=str,
            help='extract image feature for adv at this layer')
    parser.add_argument('--adv_capacity', type=int, default=300,
            help='linear layer dimension for adv component')
    parser.add_argument('--adv_conv', action='store_true',
            help='add conv layers to adv component')
    parser.add_argument('--adv_lambda', type=float, default=1.0,
            help='weight assigned to adv loss')
    parser.add_argument('--no_avgpool', action='store_true',
            help='remove avgpool layer for adv component')
    parser.add_argument('--adv_dropout', type=float, default=0.2,
            help='parameter for dropout layter in adv component')

    parser.add_argument('--blackout', action='store_true')
    parser.add_argument('--blackout_face', action='store_true')
    parser.add_argument('--blackout_box', action='store_true')
    parser.add_argument('--blur', action='store_true')
    parser.add_argument('--grayscale', action='store_true')
    parser.add_argument('--edges', action='store_true')

    parser.add_argument('--hid_size', type=int, default=300,
            help='linear layer dimension for attacker')

    ## training setting for attacker
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--autoencoder_finetune', action='store_true')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.00005,
            help='attacker learning rate')

    args = parser.parse_args()

    train_transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])
    test_transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.crop_size),
        transforms.ToTensor(),
        normalize])

    #Build adv model
    args.layer = 'generated_image'
    args.adv_on = True
    args.adv_conv = True
    args.no_avgpool = False
    adv_model_path = os.path.join('./models', args.exp_id)
    adv_model = ObjectMultiLabelAdv(args, args.num_object, args.adv_capacity, args.adv_dropout, args.adv_lambda).cuda()

    if os.path.isfile(os.path.join(adv_model_path, 'checkpoint.pth.tar')):
        print("=> loading adv model from '{}'".format(adv_model_path))
        loaded_model_name = 'model_best.pth.tar'
        checkpoint = torch.load(os.path.join(adv_model_path, loaded_model_name))
        best_performance = checkpoint['best_performance']
        adv_model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(adv_model_path))

    adv_model.eval()

    # Data samplers.

    test_data = CocoObjectGender(args, annotation_dir = args.annotation_dir, \
            image_dir = args.image_dir,split = 'test', transform = test_transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size = 16, \
            shuffle = False, num_workers = 4,pin_memory = True)

    save_dir = os.path.join('./sample_images/auto_debias', args.exp_id+'_'+str(checkpoint['epoch']))
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    results = list()
    for batch_idx, (images, targets, genders, image_ids) in enumerate(test_loader):
        if batch_idx == 10: break # constrain epoch size
        images = images.cuda()

        # save original images
        for i in range(len(images)):
            image = transf(images[i].clone().cpu())
            image.save('./sample_images/origin/{}.jpg'.format(image_ids[i].item()))

        # Forward, Backward and Optimizer
        task_pred, adv_pred, encoded_images = adv_model(images)
        for i in range(len(encoded_images)):
            image = transf(encoded_images[i].clone().cpu())
            imageID = image_ids[i].item()
            image.save('{}/{}.jpg'.format(save_dir, imageID))
            results.append({'imageID': imageID,
                           'original_image_path': './origin/{}.jpg'.format(imageID),
                           'auto_debias_image_path': './auto_debias/{}/{}.jpg'.format( \
                           args.exp_id+'_'+str(checkpoint['epoch']), imageID)})
    # render result
    import jinja2
    templateLoader = jinja2.FileSystemLoader(searchpath='.')
    templateEnv = jinja2.Environment(loader = templateLoader)
    template = templateEnv.get_template('vis_template.html')
    txt = template.render({'results': results, 'exp_id': args.exp_id})
    fh = open(os.path.join('./sample_images/', '{}_epoch_{}_predictions.html'.format(args.exp_id, checkpoint['epoch'])), 'w')
    fh.write(txt)
    fh.close()

if __name__ == '__main__':
    main()
