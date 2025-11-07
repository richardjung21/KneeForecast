from torch.utils.data import Dataset
import torch

import numpy as np
import cv2

import os
import pickle
import random

class KneeXRDataset(Dataset):
    
    ############################################################
    # All pickle files are in the format of {subject_id: {time_point: data}}
    # list_dict.pkl: {subject_id: {time_point: image_path}}
    # labels.pkl: {subject_id: {time_point: label}}
    # changes.pkl: {subject_id: {time_point: change}}
    # bmi.pkl: {subject_id: {time_point: bmi}}
    ############################################################
    
    def __init__(self, 
                 pickle_dir:str = "/mnt/sdg2/seungho/Dataset/Forecasting/list_dict.pkl", 
                 target_dim:tuple = (512,512), 
                 split:str = 'train',
                 binary:bool = False):
        super().__init__()

        self.target_dim = target_dim
        label_dir = pickle_dir.replace(pickle_dir.split("/")[-1], "labels.pkl")
        changes_dir = pickle_dir.replace(pickle_dir.split("/")[-1], "changes.pkl")
        bmi_dir = pickle_dir.replace(pickle_dir.split("/")[-1], "bmi.pkl")

        with open(pickle_dir, "rb") as f:
            self.all_image_folders = pickle.load(f)
        self.all_image_folders = {k: v for k, v in self.all_image_folders.items() if len(v) >= 2}

        with open(label_dir, "rb") as f:
            self.labels = pickle.load(f)
            
        with open(changes_dir, "rb") as f:
            self.changes = pickle.load(f)
            
        with open(bmi_dir, "rb") as f:
            self.bmi = pickle.load(f)

        train_end, val_end = int(len(self.all_image_folders) * 0.6), int(len(self.all_image_folders) * 0.8)
        if split == 'train':
            self.all_image_folders = dict(list(self.all_image_folders.items())[:train_end])
        elif split == 'val':
            self.all_image_folders = dict(list(self.all_image_folders.items())[train_end:val_end])
        elif split == 'test':
            self.all_image_folders = dict(list(self.all_image_folders.items())[val_end:])

        self.max_T = 4
        self.binary = binary

    @staticmethod
    def resize_img(img, target_dim):
        img = cv2.resize(img, (target_dim))
        return img
    
    @staticmethod
    def normalize_img(img):
        img = img/(img.max()/2) - 1
        return img
    
    @staticmethod
    def to_torch(npy):
        if len(npy.shape) == 2:
            npy = torch.from_numpy(npy).float().unsqueeze(0)
        else:
            npy = torch.from_numpy(npy).float()
        return npy
    
    def load_img(self, path):
        img = cv2.imread(path)
        img = self.resize_img(img, self.target_dim)
        img = img.transpose((2, 0, 1))
        img = self.normalize_img(img)
        return img

    def __len__(self):
        return len(self.all_image_folders)
    
    def __getitem__(self, idx):
        id = list(self.all_image_folders.keys())[idx]
        bmi = np.array(self.bmi[id[:-1]]["00"])
        while np.isnan(bmi):
            idx += 1
            id = list(self.all_image_folders.keys())[idx]
            bmi = np.array(self.bmi[id[:-1]]["00"])
        time_points = list(self.all_image_folders[id].keys())
        if self.binary:
            t0, t1 = 0, 12
        else:
            t0, t1 = sorted([int(t) for t in random.sample(time_points, 2)])

        img_dir0 = self.all_image_folders[id][f"{t0:02d}"]
        img_dir1 = self.all_image_folders[id][f"{t1:02d}"]

        kl_0 = self.labels[id][f"{t0:02d}"]
        kl_1 = self.labels[id][f"{t1:02d}"]
        change = self.changes[id][f"{t0:02d}"]
        # Remove this portion if you want to use the original KL grades
        # kl_0 = kl_0-1 if kl_0 > 0 else kl_0
        # kl_1 = kl_1-1 if kl_1 > 0 else kl_1

        t0 = t0//12
        t1 = t1//12

        img0 = self.to_torch(self.load_img(img_dir0))
        img1 = self.to_torch(self.load_img(img_dir1))
        t0 = self.to_torch(np.array(t0))
        t1 = self.to_torch(np.array(t1))
        kl_0 = self.to_torch(np.array(kl_0))
        kl_1 = self.to_torch(np.array(kl_1))
        change = self.to_torch(np.array(change))
        bmi = self.to_torch(bmi)
        return img0, img1, t0, t1, kl_0, kl_1, change, bmi