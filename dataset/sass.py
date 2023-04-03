import numpy as np
import matplotlib.pyplot as plt
from dataset.transform import *
import math
import os
from PIL import Image
import random
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
from util.utils import *
import torch
from torch.utils.data import DataLoader
from model.tools import *


class VocDataset(Dataset):
    def __init__(self, name, root, mode, size, aug=True):
        """
        :param name: dataset name, pascal or cityscapes
        :param root: root path of the dataset.
        :param mode: train: supervised learning only with labeled images, no unlabeled images are leveraged.
                     label: pseudo labeling the remaining unlabeled images.
                     semi_train: semi-supervised learning with both labeled and unlabeled images.
                     val: validation.

        :param size: crop size of training images.
        """
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size
        self.aug = aug
        self.ignore_class = 21  # voc

        self.img_path = root + '/JPEGImages/'
        self.true_mask_path = root + '/SegmentationClass/'

        if mode == 'val':
            self.label_path = self.true_mask_path
            id_path = 'dataset/splits/%s/val.txt' % name
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()

        else:
            id_path = 'dataset/splits/%s/train.txt' % name
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()

            if mode == 'full':
                self.label_path = self.true_mask_path
            elif mode == 'point':
                self.label_path = root + '/point/segcls'
            elif mode == 'scribble':
                self.label_path = root + '/scribble/seginst'

            else:
                self.label_path = root + '/scribble/seginst'

    def get_cls_label(self, cls_label):
        cls_label_set = list(cls_label)

        if self.ignore_class in cls_label_set:
            cls_label_set.remove(self.ignore_class)
        if 255 in cls_label_set:
            cls_label_set.remove(255)

        cls_label = np.zeros(self.ignore_class)
        for i in cls_label_set:
            cls_label[i] += 1
        cls_label = torch.from_numpy(cls_label).float()
        return cls_label

    def __getitem__(self, item):
        id = self.ids[item]
        img = Image.open(os.path.join(self.img_path, id.split(' ')[0]))
        mask = Image.open(os.path.join(self.label_path, id.split(' ')[1]))

        if self.mode == 'val':
            img, mask = normalize(img, mask)
            return img, mask, id

        cls_label = np.unique(np.asarray(mask))

        # basic augmentation on all training images
        img, mask = resize([img, mask], (0.5, 2.0))
        img, mask = crop([img, mask], self.size)
        img, mask = hflip(img, mask, p=0.5)

        # # strong augmentation
        if self.aug:
            if random.random() < 0.5:
                img = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img)
            img = transforms.RandomGrayscale(p=0.1)(img)
            img = blur(img, p=0.5)
            img = random_bright(img, p=0.5)

        img, mask = normalize(img, mask)
        cls_label = self.get_cls_label(cls_label)
        return img, mask, cls_label, id

    def __len__(self):
        return len(self.ids)


class CityDataset(Dataset):
    def __init__(self, name, root, mode, size, aug=True):
        """
        :param name: dataset name, pascal or cityscapes
        :param root: root path of the dataset.
        :param mode: train: supervised learning only with labeled images, no unlabeled images are leveraged.
                     label: pseudo labeling the remaining unlabeled images.
                     semi_train: semi-supervised learning with both labeled and unlabeled images.
                     val: validation.

        :param size: crop size of training images.
        """
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size
        self.aug = aug
        self.ignore_class = 19  # city

        self.img_path = root
        self.true_mask_path = root

        if mode == 'val':
            self.label_path = self.true_mask_path
            id_path = 'dataset/splits/%s/val.txt' % name
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()

        else:
            id_path = 'dataset/splits/%s/train.txt' % name
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()

            if mode == 'label':
                self.label_path = self.true_mask_path

            elif mode == 'point':
                self.label_path = root + '/100clicks'

            else:
                self.label_path = root + '/random_50clicks'

    def get_cls_label(self, cls_label):
        cls_label_set = list(cls_label)

        if self.ignore_class in cls_label_set:
            cls_label_set.remove(self.ignore_class)
        if 255 in cls_label_set:
            cls_label_set.remove(255)

        cls_label = np.zeros(self.ignore_class)
        for i in cls_label_set:
            cls_label[i] += 1
        cls_label = torch.from_numpy(cls_label).float()
        return cls_label

    def __getitem__(self, item):
        id = self.ids[item]
        img = Image.open(os.path.join(self.img_path, id.split(' ')[0]))

        if self.mode == 'val':
            mask = Image.open(os.path.join(self.label_path, id.split(' ')[1]))
            img, mask = normalize(img, mask)
            return img, mask, id

        if self.mode == 'view':
            mask = Image.open(os.path.join(self.label_path, id.split(' ')[1].split('/')[-1]))
            true_mask = Image.open(os.path.join(self.true_mask_path, id.split(' ')[1]))
            img, mask, true_mask = resize([img, mask, true_mask], (0.5, 2.0))
            img, mask, true_mask = crop([img, mask, true_mask], self.size)

            img, mask = normalize(img, mask)
            true_mask = torch.from_numpy(np.asarray(true_mask)).long()
            return img, mask, true_mask, id

        # weak train
        mask = Image.open(os.path.join(self.label_path, id.split(' ')[1].split('/')[-1]))
        cls_label = np.unique(np.asarray(mask))
        # basic augmentation on all training images
        img, mask = resize([img, mask], (0.5, 2.0))
        img, mask = crop([img, mask], self.size)
        img, mask = hflip(img, mask, p=0.5)

        if self.aug:
            # # strong augmentation
            if random.random() < 0.5:
                img = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img)
            img = transforms.RandomGrayscale(p=0.1)(img)
            img = blur(img, p=0.5)
            img = random_bright(img, p=0.5)

        img, mask = normalize(img, mask)
        cls_label = self.get_cls_label(cls_label)
        return img, mask, cls_label, id

    def __len__(self):
        return len(self.ids)

