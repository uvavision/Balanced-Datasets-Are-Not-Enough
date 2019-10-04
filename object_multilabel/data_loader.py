import json, os, string, random, time, pickle, gc, pdb
from PIL import Image
from PIL import ImageFilter
import numpy as np
import random
from random import shuffle
from pycocotools.coco import COCO

import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms

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
            balanced_subset = pickle.load(open("./data/{}_ratio_{}.ids".format(split, \
                args.ratio)))
            self.ann_data = [self.ann_data[i] for i in balanced_subset]

        if balanced_val and split == 'val':
            balanced_subset = pickle.load(open("./data/{}_ratio_{}.ids".format(split, \
                args.ratio)))
            self.ann_data = [self.ann_data[i] for i in balanced_subset]

        if balanced_test and split == 'test':
            balanced_subset = pickle.load(open("./data/{}_ratio_{}.ids".format(split, \
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
            random.shuffle(man_idxs) # need to do random sample every time
            random.shuffle(woman_idxs)
            min_len = 3000 if split == 'train' else 1500
            selected_idxs = list(man_idxs[:min_len]) + list(woman_idxs[:min_len])

            self.ann_data = [self.ann_data[idx] for idx in selected_idxs]
            self.object_ann = np.take(self.object_ann, selected_idxs, axis=0)
            self.gender_ann = np.take(self.gender_ann, selected_idxs, axis=0)

        print("man size : {} and woman size: {}".format(len(np.nonzero( \
                self.gender_ann[:, 0])[0]), len(np.nonzero(self.gender_ann[:, 1])[0])))

        # load mask annotations
        if args.blackout or args.blackout_box or args.blur or args.grayscale or args.edges:
            self.cocoAnnDir = os.path.join(self.annotation_dir, 'annotations_pytorch')
            if self.split == 'train':
                self.root = os.path.join(self.image_dir, '/train2014')
                self.captionFile = os.path.join(self.cocoAnnDir, 'captions_train2014.json')
                self.annFile = os.path.join(self.cocoAnnDir, 'instances_train2014.json')
            else:
                self.root = os.path.join(self.image_dir, '/val2014')
                self.captionFile = os.path.join(self.cocoAnnDir, 'captions_val2014.json')
                self.annFile = os.path.join(self.cocoAnnDir, 'instances_val2014.json')

            self.cocoAPI = COCO(self.annFile)

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
        if self.args.blackout:
            ann_ids = self.cocoAPI.getAnnIds(imgIds = img_id)
            img_ = self.blackout(img_, ann_ids, 'people')
        elif self.args.blackout_box:
            ann_ids = self.cocoAPI.getAnnIds(imgIds = img_id)
            img_ = self.blackout(img_, ann_ids, 'people_box')
        elif self.args.blur:
            ann_ids = self.cocoAPI.getAnnIds(imgIds = img_id)
            img_ = self.blur(img_, ann_ids, 'people')
        elif self.args.grayscale:
            ann_ids = self.cocoAPI.getAnnIds(imgIds = img_id)
            img_ = self.grey(img_, ann_ids)
        elif self.args.edges:
            ann_ids = self.cocoAPI.getAnnIds(imgIds = img_id)
            img_ = self.find_edges(img_, ann_ids)
        elif self.args.blackout_face:
            img_ = self.blackout_face(img_, img_id)

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

    def find_edges(self, img, ann_ids):
        anns = [ann for ann in self.cocoAPI.loadAnns(ann_ids)
                if ann['category_id'] == 1]
        if len(anns) > 0:
            outlined_img = img.convert(mode='L').filter(ImageFilter.CONTOUR)
            mask = self.cocoAPI.annToMask(anns[0])
            for ann in anns[1:]:
                mask = mask + self.cocoAPI.annToMask(ann)
            img_mask = Image.fromarray(255 * (mask > 0).astype('uint8'))
            return Image.composite(outlined_img, img, img_mask)
        return img

    def grey(self, img, ann_ids):
        anns = [ann for ann in self.cocoAPI.loadAnns(ann_ids)
                if ann['category_id'] == 1]
        if len(anns) > 0:
            grey_img = img.convert(mode='L')
            mask = self.cocoAPI.annToMask(anns[0])
            for ann in anns[1:]:
                mask = mask + self.cocoAPI.annToMask(ann)
            img_mask = Image.fromarray(255 * (mask > 0).astype('uint8'))
            return Image.composite(grey_img, img, img_mask)
        return img

    def blur(self, img, ann_ids, processed_area):
        # Only people category, category_id == 1.
        anns = [ann for ann in self.cocoAPI.loadAnns(ann_ids)
                if ann['category_id'] == 1]
        if len(anns) > 0:
            blurred_img = img.filter(ImageFilter.GaussianBlur(radius = 10))
            mask = self.cocoAPI.annToMask(anns[0])
            for ann in anns[1:]:
                mask = mask + self.cocoAPI.annToMask(ann)
            if processed_area == 'people':
                img_mask = Image.fromarray(255 * (mask > 0).astype('uint8'))
            elif processed_area == 'background':
                img_mask = Image.fromarray(255 * (mask == 0).astype('uint8'))
            else:
                print("Please specify blur people or background")
            return Image.composite(blurred_img, img, img_mask)
        return img

    def blackout(self, img, ann_ids, processed_area):
        # Only people category, category_id == 1.
        anns = [ann for ann in self.cocoAPI.loadAnns(ann_ids)
                if ann['category_id'] == 1]
        if len(anns) > 0:
            black_img = Image.fromarray(np.zeros((img.size[1], img.size[0])))
            mask = self.cocoAPI.annToMask(anns[0])
            for ann in anns[1:]:
                mask = mask + self.cocoAPI.annToMask(ann)
            if processed_area == 'people_box':
                self.box_mask(mask)
            if processed_area == 'people' or processed_area == 'people_box':
                img_mask = Image.fromarray(255 * (mask > 0).astype('uint8'))
            elif processed_area == 'background':
                img_mask = Image.fromarray(255 * (mask == 0).astype('uint8'))
            else:
                print("Please specify blackout people or background")
            return Image.composite(black_img, img, img_mask)
        return img

    def box_mask(self, mask):
        x_limits = np.nonzero(mask.sum(axis = 0))
        xmin = x_limits[0][0]; xmax = x_limits[0][-1]
        y_limits = np.nonzero(mask.sum(axis = 1))
        ymin = y_limits[0][0]; ymax = y_limits[0][-1]
        mask[ymin:ymax, xmin:xmax] = 1

    def blackout_face(self, img, img_name):
        try:
            vertices = self.faces[int(img_name)]
        except:
            return img

        width = img.size[1]
        height = img.size[0]

        black_img = Image.fromarray(np.zeros((img.size[1], img.size[0])))
        mask = np.zeros((width, height))
        for poly in vertices:
            xmin, ymin = poly[0].strip('()').split(',')
            xmax, ymax = poly[2].strip('()').split(',')
            for i in range(int(xmin), int(xmax)):
                for j in range(int(ymin), int(ymax)):
                    mask[j][i] = 1
        img_mask = Image.fromarray(255 * (mask > 0).astype('uint8')).resize((img.size[0], \
                img.size[1]), Image.ANTIALIAS)

        return Image.composite(black_img, img, img_mask)


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

