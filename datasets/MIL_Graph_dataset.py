import random
import torch
import pandas as pd
import os
from pathlib import Path

import torch.utils.data as data
import h5py
import numpy as np

# class MyTransform(object):
#     def __init__(self, p=0.5, ratio=0.8):
#         self.p = p
#         self.ratio = ratio

#     def __call__(self, features):
#         if random.random() < self.p:
#             index = [x for x in range(features.shape[0])]
#             random.shuffle(index)
#             features = features[index]
#             features = features[:int(features.shape[0]*self.ratio)]
#         return features

class MilGraphDataset(data.Dataset):
    def __init__(self, dataset_cfg=None, state=None):
        # Set all input args as attributes
        self.__dict__.update(locals())
        self.dataset_cfg = dataset_cfg

        #---->data and label
        self.feature_dir = dataset_cfg.data_dir
        if not isinstance(self.feature_dir, list):
            self.feature_dir = [self.feature_dir]
        self.split_dir = dataset_cfg.split_dir

        used_label = dataset_cfg.label_dict.values()
        print(f'used_label: {dataset_cfg.label_dict.keys()}')
        fold = dataset_cfg.fold

        label_csv = pd.read_csv(dataset_cfg.label_csv_path)
        #---->split dataset
        data = pd.read_csv(os.path.join(self.split_dir, f'splits_{fold}.csv'))[state].dropna().to_list()
        label = [dataset_cfg.label_dict[label_csv.loc[label_csv['slide_id'] == i, 'label'].values[0]] for i in data]
        
        self.data, self.label = [], []
        self.slide_cls_ids = [[] for _ in range(len(used_label))]
        for image_id, lbl in zip(data, label):
            if lbl in used_label:
                self.data.append(image_id)
                self.label.append(lbl)
                self.slide_cls_ids[lbl].append(image_id)

        # self.data = [os.path.splitext(x)[0] for x in self.data]
        
        # self.transform = MyTransform(p=0.5, ratio=0.8)

        self.shuffle = dataset_cfg.data_shuffle
        self.state = state

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        slide_id = self.data[idx]
        label = int(self.label[idx])
        
        if self.state == 'train':
            feature_dir = random.choice(self.feature_dir)
        else:
            feature_dir = self.feature_dir[0]

        full_path = Path(feature_dir) / f'{slide_id}.pt'
        G = torch.load(full_path)
        # features = G.x
        # postorder_index = G.postorder_index
        # preorder_index = G.preorder_index
        # levelorder_index = G.levelorder_index

        return {'data': G, 'label': label, 'slide_id': slide_id}

    def getlabel(self, idx):
        return self.label[idx]
