# -*- coding: utf-8 -*-
# @Time : 2020/12/1 下午1:38 
# @Author : midaskong 
# @File : dataloader.py 
# @Description:
from __future__ import print_function, absolute_import
import torch.utils.data as data
import numpy as np
import cv2
import os
import torch
from torch.utils.data import DataLoader

class CRNNData(data.Dataset):
    def __init__(self, data_root, img_h, img_w, maxlen, batchsize, datalist, is_train=True):

        self.root = data_root
        self.is_train = is_train
        self.inp_h = img_h
        self.inp_w = img_w

        self.mean = 0.5
        self.std = 0.5

        txt_file = datalist['train'] if is_train else datalist['val']

        # convert name:indices to name:string
        with open(txt_file, 'r', encoding='utf-8') as file:
            self.labels = [{c.split()[0]: c.split()[1]} for c in file.readlines()]

        # print('self.labels::',self.labels[0].values())
        # print('self.labels~~~~', len(list(self.labels[0].values())[0]))
        self.labels = sorted(self.labels, key=lambda item:len(list(item.values())[0]))
        # print('self.labels',self.labels)
        labeltmp = []
        for i in range(maxlen):
            labeltmp.append([])
        # print('labeltmp',labeltmp)

        for item in self.labels:
            labeltmp[len(list(item.values())[0])-1].append(item)
        labeltmpnew = []
        # print("len 1:",len(labeltmp[0]))
        for i in range(maxlen):
            ntmp = len(labeltmp[i]) // batchsize
            labeltmpnew.extend(labeltmp[i][:ntmp*batchsize])

        # print('labeltmpnew:',labeltmpnew)
        self.labels = labeltmpnew
       # print('self.labels :',self.labels)

        print("load {} images!".format(self.__len__()))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        img_name = list(self.labels[idx].keys())[0]
        labelen = len(list(self.labels[idx].values())[0])
        tw = int(self.inp_h*1.5*labelen)
        # print(labelen)
        # print()
        img = cv2.imread(os.path.join(self.root, img_name))
        # print(img.shape)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = cv2.resize(img, (tw, self.inp_h))
        # print("resize image:",img.shape)
        # cv2.imwrite("resize.jpg",img)
        img = np.reshape(img, (self.inp_h, tw, 3))

        img = img.astype(np.float32)
        img = (img/255. - self.mean) / self.std
        img = img.transpose([2, 0, 1])

        return img, idx
