from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
import pdb

from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image, ImageOps
from random import random, randint

import warnings

warnings.filterwarnings("ignore")

def make_dataset(root, mode):
    assert mode in ['trainFull', 'trainSemi', 'pseudoLabels','val', 'test']
    items = []

    if mode == 'trainFull':
        train_img_path = os.path.join(root, 'trainFull', 'Img')
        train_mask_path = os.path.join(root, 'trainFull', 'GT')
        train_mask_w_path = os.path.join(root, 'trainFull', 'WeaklyAnnotations')

        images = os.listdir(train_img_path)
        labels = os.listdir(train_mask_path)
        labelsw = os.listdir(train_mask_w_path)

        images.sort()
        labels.sort()
        labelsw.sort()

        for it_im, it_gt, it_gtw in zip(images, labels, labelsw):
            item = (os.path.join(train_img_path, it_im), os.path.join(train_mask_path, it_gt),os.path.join(train_mask_w_path, it_gtw))
            items.append(item)

    elif mode == 'trainSemi':
        train_img_path = os.path.join(root, 'trainSemi', 'Img')
        train_mask_w_path = os.path.join(root, 'trainSemi', 'WeaklyAnnotations')

        images = os.listdir(train_img_path)
        labelsw = os.listdir(train_mask_w_path)

        images.sort()
        labelsw.sort()

        for it_im, it_gtw in zip(images, labelsw):
            item = (os.path.join(train_img_path, it_im),
                    os.path.join(train_mask_w_path, it_gtw))
            items.append(item)

    elif mode == 'pseudoLabels':
        train_img_path = os.path.join(root, 'trainSemi', 'Img')
        train_mask_path = os.path.join(root, 'trainSemi', 'GT')
        train_mask_w_path = os.path.join(root, 'trainSemi', 'WeaklyAnnotations')

        images = os.listdir(train_img_path)
        labels = os.listdir(train_mask_path)
        labelsw = os.listdir(train_mask_w_path)

        images.sort()
        labels.sort()
        labelsw.sort()

        for it_im, it_gt, it_gtw in zip(images, labels, labelsw):
            item = (os.path.join(train_img_path, it_im), os.path.join(train_mask_path, it_gt),
                    os.path.join(train_mask_w_path, it_gtw))
            items.append(item)

    elif mode == 'val':
        train_img_path = os.path.join(root, 'val', 'Img')
        train_mask_path = os.path.join(root, 'val', 'GT')

        images = os.listdir(train_img_path)
        labels = os.listdir(train_mask_path)

        images.sort()
        labels.sort()

        for it_im, it_gt in zip(images, labels):
            item = (os.path.join(train_img_path, it_im), os.path.join(train_mask_path, it_gt))
            items.append(item)
    else:
        train_img_path = os.path.join(root, 'test', 'Img')
        train_mask_path = os.path.join(root, 'test', 'GT')

        images = os.listdir(train_img_path)
        labels = os.listdir(train_mask_path)

        images.sort()
        labels.sort()

        for it_im, it_gt in zip(images, labels):
            item = (os.path.join(train_img_path, it_im), os.path.join(train_mask_path, it_gt))
            items.append(item)

    return items


class MedicalImageDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, mode, root_dir, transform=None, mask_transform=None, augment=False, equalize=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.imgs = make_dataset(root_dir, mode)
        self.augmentation = augment
        self.equalize = equalize
        self.mode = mode

    def __len__(self):
        return len(self.imgs)

    def augment2(self, img, mask):
        if random() > 0.5:
            img = ImageOps.flip(img)
            mask = ImageOps.flip(mask)
        if random() > 0.5:
            img = ImageOps.mirror(img)
            mask = ImageOps.mirror(mask)
        if random() > 0.5:
            angle = random() * 60 - 30
            img = img.rotate(angle)
            mask = mask.rotate(angle)
        return img, mask

    def augment(self, img, mask, mask_w):
        if random() > 0.5:
            img = ImageOps.flip(img)
            mask = ImageOps.flip(mask)
            mask_w = ImageOps.flip(mask_w)
        if random() > 0.5:
            img = ImageOps.mirror(img)
            mask = ImageOps.mirror(mask)
            mask_w = ImageOps.mirror(mask_w)
        if random() > 0.5:
            angle = random() * 60 - 30
            img = img.rotate(angle)
            mask = mask.rotate(angle)
            mask_w = mask_w.rotate(mask_w)
        return img, mask, mask_w

    def __getitem__(self, index):
        if self.mode == 'trainFull':
            img_path, mask_path, mask_w_path = self.imgs[index]
            img = Image.open(img_path)
            mask = Image.open(mask_path).convert('L')
            mask_w = Image.open(mask_w_path).convert('L')

            if self.equalize:
                img = ImageOps.equalize(img)

            if self.augmentation:
                img, mask, mask_w = self.augment(img, mask, mask_w)

            if self.transform:
                img = self.transform(img)
                mask = self.mask_transform(mask)
                mask_w = self.mask_transform(mask_w)

            return [img, mask, mask_w, img_path]

        elif self.mode == 'trainSemi':
            img_path, mask_w_path = self.imgs[index]
            img = Image.open(img_path)
            mask_w = Image.open(mask_w_path).convert('L')

            if self.equalize:
                img = ImageOps.equalize(img)

            if self.augmentation:
                img, mask_w = self.augment(img, mask_w)

            if self.transform:
                img = self.transform(img)
                mask_w = self.mask_transform(mask_w)

            return [img, mask_w, img_path]
        elif self.mode == 'pseudoLabels':
            img_path, mask_path, mask_w_path = self.imgs[index]
            img = Image.open(img_path)
            mask_w = Image.open(mask_w_path).convert('L')

            if self.equalize:
                img = ImageOps.equalize(img)

            if self.augmentation:
                img, mask_w = self.augment(img, mask_w)

            if self.transform:
                img = self.transform(img)
                mask_w = self.mask_transform(mask_w)

            return [img, mask, mask_w, img_path]
        else:
            img_path, mask_path = self.imgs[index]
            img = Image.open(img_path)
            mask = Image.open(mask_path).convert('L')

            if self.equalize:
                img = ImageOps.equalize(img)

            if self.augmentation:
                img, mask = self.augment2(img, mask)

            if self.transform:
                img = self.transform(img)
                mask = self.mask_transform(mask)

            return [img, mask, img_path]