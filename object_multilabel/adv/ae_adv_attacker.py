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

from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from tqdm import tqdm as tqdm

from data_loader import CocoObjectGender, CocoObjectGenderFeature
from ae_adv_model import ObjectMultiLabelAdv, get_norm_layer, UnetGenerator
from model import GenderClassifier

####### data preparation #########################
object_id_map = pickle.load(open('../data/object_id.map'))
object2id = object_id_map['object2id']
id2object = object_id_map['id2object']

def generate_image_feature(split, image_features_path, data_loader, encoder):
    targets = list()
    genders = list()
    image_ids = list()
    potentials = list()

    for ind, (images_, targets_, genders_, image_ids_) in enumerate(data_loader):
        images_ = images_.cuda(0)
        preds, _, _ = encoder(images_)
        potentials.append(preds.detach().cpu())
        targets.append(targets_.cpu())
        genders.append(genders_.cpu())
        image_ids.append(image_ids_.cpu())

    targets = torch.cat(targets, 0)
    genders = torch.cat(genders, 0)
    image_ids = torch.cat(image_ids, 0)
    potentials = torch.cat(potentials, 0)

    torch.save(targets, os.path.join(image_features_path, '{}_targets.pth'.format(split)))
    torch.save(genders, os.path.join(image_features_path, '{}_genders.pth'.format(split)))
    torch.save(image_ids, os.path.join(image_features_path, '{}_image_ids.pth'.format(split)))
    torch.save(potentials, os.path.join(image_features_path, '{}_potentials.pth'.format(split)))

def test(epoch, model, data_loader):
    model.eval()
    nProcessed = 0
    task_preds, adv_preds = [], []
    task_truth, adv_truth = [], []

    t = tqdm(data_loader, desc = 'Val')

    for batch_idx, (images, targets, genders, image_ids) in enumerate(t):
        #if batch_idx == 100: break # constrain epoch size

        # Set mini-batch dataset
        images = images.cuda()
        targets = targets.cuda()
        genders = genders.cuda()

        # Forward, Backward and Optimizer
        task_pred, adv_pred, autoencoded_images = model(images)

        adv_pred = np.argmax(F.softmax(adv_pred, dim=1).cpu().detach().numpy(), axis=1)
        adv_preds += adv_pred.tolist()
        adv_truth += genders.cpu().max(1, keepdim=False)[1].numpy().tolist()

        task_pred = torch.sigmoid(task_pred)

        if batch_idx > 0 and len(task_preds) > 0:
            task_preds = torch.cat((task_preds, task_pred.detach().cpu()), 0)
            task_truth = torch.cat((task_truth, targets.cpu()), 0)
            total_genders = torch.cat((total_genders, genders.cpu()), 0)
        else:
            task_preds = task_pred.detach().cpu()
            task_truth = targets.cpu()
            total_genders = genders.cpu()

    task_f1_score = f1_score(task_truth.numpy(), (task_preds >= 0.5).long().numpy(), average = 'macro')
    meanAP = average_precision_score(task_truth.numpy(), task_preds.numpy(), average='macro')
    adv_acc = accuracy_score(adv_truth, adv_preds)

    print('Test epoch   : meanAP: {:.2f}, adv acc: {:.2f}, '.format(meanAP*100, adv_acc*100))

    return task_f1_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', type=str,
            help='experiment id, e.g. conv4_300_1.0_0.2_1')

    parser.add_argument('--num_rounds', type=int,
            default = 5)

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

    normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225])
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
    val_data = CocoObjectGender(args, annotation_dir = args.annotation_dir, \
            image_dir = args.image_dir,split = 'val', transform = test_transform)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size = 16, \
            shuffle = False, num_workers = 4,pin_memory = True)

    test_data = CocoObjectGender(args, annotation_dir = args.annotation_dir, \
            image_dir = args.image_dir,split = 'test', transform = test_transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size = 16, \
            shuffle = False, num_workers = 4,pin_memory = True)

    print('val set performance:')
    test(args, adv_model, val_loader)

    print('test set performance:')
    test(args, adv_model, test_loader)


    acc_list = {}
    acc_list['potential'] = []

    args.gender_balanced = True
    for i in range(args.num_rounds):

        train_data = CocoObjectGender(args, annotation_dir = args.annotation_dir, \
                image_dir = args.image_dir,split = 'train', transform = train_transform)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size = 16,
                    shuffle = True, num_workers = 6, pin_memory = True)

        val_data = CocoObjectGender(args, annotation_dir = args.annotation_dir, \
                image_dir = args.image_dir,split = 'val', transform = test_transform)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size = 16, \
                shuffle = False, num_workers = 4,pin_memory = True)

        test_data = CocoObjectGender(args, annotation_dir = args.annotation_dir, \
                image_dir = args.image_dir,split = 'test', transform = test_transform)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size = 16, \
                shuffle = False, num_workers = 4,pin_memory = True)


        image_features_path = os.path.join(adv_model_path, 'image_features')
        if not os.path.exists(image_features_path):
            os.makedirs(image_features_path)
        # get image features from encoder
        generate_image_feature('train', image_features_path, train_loader, adv_model)
        generate_image_feature('val', image_features_path, val_loader, adv_model)
        generate_image_feature('test', image_features_path, test_loader, adv_model)


        train_data = CocoObjectGenderFeature(args, image_features_path, split = 'train')
        train_loader = torch.utils.data.DataLoader(train_data, batch_size = args.batch_size,
                    shuffle = True, num_workers = 6, pin_memory = True)

        val_data = CocoObjectGenderFeature(args, image_features_path, split = 'val')
        val_loader = torch.utils.data.DataLoader(val_data, batch_size = args.batch_size,
                    shuffle = True, num_workers = 6, pin_memory = True)

        test_data = CocoObjectGenderFeature(args, image_features_path, split = 'test')
        test_loader = torch.utils.data.DataLoader(test_data, batch_size = args.batch_size,
                    shuffle = True, num_workers = 6, pin_memory = True)

        model_save_dir = './attacker'
        model_save_dir = os.path.join(model_save_dir, str(args.exp_id))

        if not os.path.exists(model_save_dir): os.makedirs(model_save_dir)

        for feature_type in acc_list.keys():

            #import pdb
            #pdb.set_trace()
            attacker = GenderClassifier(args, args.num_object)

            attacker = attacker.cuda()

            optimizer = optim.Adam(attacker.parameters(), lr=args.learning_rate, weight_decay = 1e-5)

            train_attacker(args.num_epochs, optimizer, attacker, train_loader, val_loader, \
               model_save_dir, feature_type)

            # evaluate best attacker on balanced test split
            best_attacker = torch.load(model_save_dir + '/best_attacker.pth.tar')
            attacker.load_state_dict(best_attacker['state_dict'])
            _, val_acc = epoch_pass(0, val_loader, attacker, None, False, feature_type)
            val_acc = 0.5 + abs(val_acc - 0.5)
            _, test_acc = epoch_pass(0, test_loader, attacker, None, False, feature_type)
            test_acc = 0.5 + abs(test_acc - 0.5)
            acc_list[feature_type].append(test_acc)
            print('round {} feature type: {}, test acc: {}, val acc: {}'.format(i, feature_type, test_acc, val_acc))

    for feature_type in acc_list.keys():
        print(acc_list[feature_type], np.std(np.array(acc_list[feature_type])))
        print('{} average leakage: {}'.format(feature_type, np.mean(np.array(acc_list[feature_type]))))


def train_attacker(num_epochs, optimizer, attacker, train_loader, test_loader, model_save_dir, feature_type, print_every=500):
    # training setting
    attacker.train()

    train_acc_arr, train_loss_arr = [], []
    dev_acc_arr, dev_loss_arr = [], []
    best_model_epoch = 1
    best_score = 0.0

    for epoch in xrange(1, num_epochs + 1):

        # train
        loss, train_task_acc = epoch_pass(epoch, train_loader, attacker, optimizer, True, feature_type, print_every)
        train_acc_arr.append(train_task_acc)
        train_loss_arr.append(loss)
        if epoch % 10 == 0:
            print('train, {0}, adv acc: {1:.2f}'.format(epoch, train_task_acc*100))

        # dev
        loss, dev_task_acc = epoch_pass(epoch, test_loader, attacker, optimizer, False, feature_type, print_every)
        dev_acc_arr.append(dev_task_acc)
        dev_loss_arr.append(loss)
        if epoch % 10 == 0:
            print('dev, {0}, adv acc: {1:.2f}'.format(epoch, dev_task_acc*100))

        if dev_task_acc > best_score:
            best_score = dev_task_acc
            best_model_epoch = epoch
            torch.save({'epoch': epoch, 'state_dict': attacker.state_dict()}, model_save_dir + '/best_attacker.pth.tar')

        if epoch % 10 == 0:
            print('current best dev score: {:.2f}'.format(best_score*100))

def epoch_pass(epoch, data_loader, attacker, optimizer, training, feature_type, print_every=500):

    t_loss = 0.0
    preds, truth = [], []
    n_processed = 0

    if training:
        attacker.train()
    else:
        attacker.eval()

    for ind, (targets, genders, image_ids, potentials) in enumerate(data_loader):

        features = potentials.float().cuda()

        adv_pred = attacker(features)
        loss = F.cross_entropy(adv_pred, genders.cuda().max(1, keepdim=False)[1], reduction='elementwise_mean')

        adv_pred = np.argmax(F.softmax(adv_pred, dim=1).cpu().detach().numpy(), axis=1)
        preds += adv_pred.tolist()
        truth += genders.cpu().max(1, keepdim=False)[1].numpy().tolist()

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        t_loss += loss.item()
        n_processed += len(targets)

        acc_score = accuracy_score(truth, preds)

    return t_loss / n_processed, acc_score

if __name__ == '__main__':
    main()
