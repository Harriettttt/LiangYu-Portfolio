import os
import collections
import torch
import numpy as np
from skimage.transform import resize
from skimage.io import imread
import matplotlib.pyplot as plt
from torch.utils import data
import os.path as pathlib
import torchvision.transforms as T

class_labels_dict = {
    "Sky": 0,
    "Building": 1,
    "Pole": 2,
    "Road": 3,
    "LaneMarking": 4,
    "SideWalk": 5,
    "Pavement": 6,
    "Tree": 7,
    "SignSymbol": 8,
    "Fence": 9,
    "Car_Bus": 10,
    "Pedestrian": 11,
    "Bicyclist": 12,
    "Unlabelled": 13
}

Sky = [[128, 128, 128]]
Building = [[128, 0, 0]]
Pole = [[192, 192, 128], [0, 0, 64]]
Road = [[128, 64, 128], [128, 128, 192]]
LaneMarking = [[128, 0, 192], [128, 0, 64]]
SideWalk = [[0, 0, 192]]
Pavement = [[60, 40, 222]]
Tree = [[128, 128, 0]]
SignSymbol = [[192, 128, 128]]
Fence = [[64, 64, 128]]
Car_Bus = [[64, 0, 128], [64, 128, 192], [192, 128, 192]]
Pedestrian = [[64, 64, 0]]
Bicyclist = [[0, 128, 192]]
Unlabelled = [[0, 0, 0]]

label_colours = [
        Sky,
        Building,
        Pole,
        Road,
        LaneMarking,
        SideWalk,
        Pavement,
        Tree,
        SignSymbol,
        Fence,
        Car_Bus,
        Pedestrian,
        Bicyclist,
        Unlabelled,
    ]

def from_label_to_rgb(label):
    n_classes = len(label_colours)
    r = label.copy()
    g = label.copy()
    b = label.copy()
    for l in range(0, n_classes):
        r[label == l] = label_colours[l][0][0]
        g[label == l] = label_colours[l][0][1]
        b[label == l] = label_colours[l][0][2]

    rgb = np.zeros((label.shape[0], label.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    return rgb

def from_rgb_to_label(rgb):
    n_classes = len(label_colours)
    label = np.ones((rgb.shape[0], rgb.shape[1])) * (-1)
    r = rgb[:,:,0]
    g = rgb[:,:,1]
    b = rgb[:,:,2]
    for l in range(0, n_classes):
        final_mask = np.zeros_like(label).astype(bool)
        for j in label_colours[l]:
            curr_mask = (r == j[0]) &  (g == j[1]) & (b == j[2])
            final_mask = final_mask | curr_mask
        label[final_mask] = l
    label[label==-1] = 11 # other labels are all considered as "unlabelled"
    label = label.astype(np.uint8)
    return label

class camvidLoader(data.Dataset):
    def __init__(
        self,
        root,
        split="train",
        is_aug=False,
        aug = None,
        is_pytorch_transform=True,
        img_size=None
    ):
        self.root = root
        self.split = split
        self.img_size = img_size
        self.is_aug = is_aug
        self.is_pytorch_transform = is_pytorch_transform
        self.augmentation = aug
        self.mean = np.array([104.00699/255.0, 116.66877/255.0, 122.67892/255.0])
        self.n_classes = 12
        self.files = collections.defaultdict(list)
        for split in ["train", "test", "val"]:
            file_list = os.listdir(root + "/" + split)
            self.files[split] = file_list

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_name = self.files[self.split][index]
        img_path = self.root + "/" + self.split + "/" + img_name
        lbl_path = self.root + "/" + self.split + "_labels/" + pathlib.splitext(img_name)[0] + '_L.png'
        img = imread(img_path)
        lbl = imread(lbl_path)
        lbl = from_rgb_to_label(lbl)
        img = T.ToPILImage()(img)
        lbl = T.ToPILImage()(lbl)
        if self.is_aug:
            img, lbl = self.augmentation(img, lbl)
        img, lbl = self.transform(img, lbl)
        return img, lbl, img_path

    def transform(self, img, lbl):
        img, lbl = np.array(img), np.array(lbl)
        img = resize(img, (self.img_size[0], self.img_size[1]), order=2)  # uint8 with RGB mode
        lbl = resize(lbl, (self.img_size[0], self.img_size[1]), order=0)
        if self.is_pytorch_transform:
            img = img[:, :, ::-1]  # RGB -> BGR
            img = img.astype(np.float64)
            img -= self.mean
            img = img.transpose(2, 0, 1) # NHWC -> NCHW
            img = torch.from_numpy(img).float()
            lbl = torch.from_numpy(lbl).long()
        return img, lbl

