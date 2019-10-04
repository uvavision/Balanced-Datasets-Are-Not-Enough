import json, os, string, random, time, pickle, gc, pdb
from PIL import Image
from PIL import ImageFilter
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms

from random import shuffle
import random

class CocoObjectGender(data.Dataset):
    def __init__(self, args, annotation_dir, image_dir, split = 'train', transform = None, \
            balanced_val=False, balanced_test=False):
        print("CocoObjectGender dataloader")

        self.split = split
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transform = transform
        self.args = args

        print("loading %s annotations.........." % self.split)
        self.ann_data = pickle.load(open(os.path.join(annotation_dir, split+".data")))

        if args.balanced and split == 'train':
            balanced_subset = pickle.load(open("../data/{}_ratio_{}.ids".format(split, \
                args.ratio)))
            self.ann_data = [self.ann_data[i] for i in balanced_subset]

        if balanced_val and split == 'val':
            balanced_subset = pickle.load(open("../data/{}_ratio_{}.ids".format(split, \
                args.ratio)))
            self.ann_data = [self.ann_data[i] for i in balanced_subset]

        if balanced_test and split == 'test':
            balanced_subset = pickle.load(open("../data/{}_ratio_{}.ids".format(split, \
                args.ratio)))
            self.ann_data = [self.ann_data[i] for i in balanced_subset]

        print(len(self.ann_data))
        self.object_ann = np.zeros((len(self.ann_data), self.args.num_object))
        self.gender_ann = np.zeros((len(self.ann_data), 2), dtype=int)
        for index, ann in enumerate(self.ann_data):
            self.object_ann[index] = np.asarray(ann['objects'])
            self.gender_ann[index] = np.asarray(ann['gender'])

        if args.gender_balanced:
            man_idxs = np.nonzero(self.gender_ann[:, 0])[0]
            woman_idxs = np.nonzero(self.gender_ann[:, 1])[0]
            random.shuffle(man_idxs)
            random.shuffle(woman_idxs)
            min_len = 3000 if split == 'train' else 1500
            selected_idxs = list(man_idxs[:min_len]) + list(woman_idxs[:min_len])

            self.ann_data = [self.ann_data[idx] for idx in selected_idxs]
            self.object_ann = np.take(self.object_ann, selected_idxs, axis=0)
            self.gender_ann = np.take(self.gender_ann, selected_idxs, axis=0)

        print("man size : {} and woman size: {}".format(len(np.nonzero( \
                self.gender_ann[:, 0])[0]), len(np.nonzero(self.gender_ann[:, 1])[0])))

        if args.blackout_face:
            self.faces = pickle.load(open('./data/{}_faces.p'.format(split)))

    def __getitem__(self, index):
        if self.args.no_image:
            return torch.Tensor([1]), torch.Tensor(self.object_ann[index]), \
                torch.LongTensor(self.gender_ann[index]), torch.Tensor([1])

        img = self.ann_data[index]
        img_id = img['image_id']
        img_file_name = img['file_name']

        if self.split == 'train':
            image_path_ = os.path.join(self.image_dir,"train2014", img_file_name)
        else:
            image_path_ = os.path.join(self.image_dir,"val2014", img_file_name)

        img_ = Image.open(image_path_).convert('RGB')

        if self.transform is not None:
            img_ = self.transform(img_)

        return img_, torch.Tensor(self.object_ann[index]), \
                torch.LongTensor(self.gender_ann[index]), torch.LongTensor([img_id])

    def getGenderWeights(self):
        return (self.gender_ann == 0).sum(axis = 0) / (1e-15 + \
                (self.gender_ann.sum(axis = 0) + (self.gender_ann == 0).sum(axis = 0) ))

    def getObjectWeights(self):
        return (self.object_ann == 0).sum(axis = 0) / (1e-15 + self.object_ann.sum(axis = 0))

    def __len__(self):
        return len(self.ann_data)


class CocoObjectGenderFeature(data.Dataset):
    def __init__(self, args, feature_dir, split = 'train'):
        print("CocoObjectGenderFeature dataloader")

        self.split = split
        self.args = args

        print("loading %s annotations.........." % self.split)

        self.targets = torch.load(os.path.join(feature_dir, '{}_targets.pth'.format(split)))
        self.genders = torch.load(os.path.join(feature_dir, '{}_genders.pth'.format(split)))
        self.image_ids = torch.load(os.path.join(feature_dir, '{}_image_ids.pth'.format(split)))
        self.potentials = torch.load(os.path.join(feature_dir, '{}_potentials.pth'.format(split)))

        print("man size : {} and woman size: {}".format(len(self.genders[:, 0].nonzero().squeeze()), \
            len(self.genders[:, 1].nonzero().squeeze())))

        assert len(self.targets) == len(self.genders) == len(self.image_ids) == len(self.potentials)

    def __getitem__(self, index):
        return self.targets[index], self.genders[index], self.image_ids[index], self.potentials[index]

    def __len__(self):
        return len(self.targets)
