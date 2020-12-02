# -*- coding: utf-8 -*-
# @Time : 2020/12/1 下午3:52 
# @Author : midaskong 
# @File : dataloadershuffle.py 
# @Description:

from __future__ import print_function, absolute_import
import torch.utils.data as data
import numpy as np
import cv2
from PIL import Image
import random
import os
import torch
from torch.utils.data import DataLoader

class CRNNData(data.Dataset):
    def __init__(self, data_root, img_h, maxlen, batchsize, datalist, transform=None, is_train=True):
        self.root = data_root
        self.is_train = is_train
        self.inp_h = img_h
        self.bs = batchsize
        self.transform = transform
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
            for j in range(ntmp):
                labeltmpnew.append(labeltmp[i][batchsize*j:batchsize*(j+1)])
        random.shuffle(labeltmpnew)
        # labeltmpnew = [i for k in labeltmpnew for i in k]
        # print('labeltmpnew:',labeltmpnew)
        self.labels = [i for k in labeltmpnew for i in k]
       # print('self.labels :',self.labels)

        print("load {} images!".format(self.__len__()))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        img_name = list(self.labels[idx].keys())[0]
        labelen = len(list(self.labels[idx].values())[0])
        random.seed(int(idx//self.bs))
        randconf = random.randint(50,150) / 100  #multi-scale (0.5~1.5) length
        tw = int(self.inp_h*randconf*labelen)
        img = cv2.imread(os.path.join(self.root, img_name))
        # print(img.shape)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if self.is_train:
            img = cv2.resize(img, (tw, self.inp_h))
        else:
            tw = int(self.inp_h * labelen)
            img = cv2.resize(img, (tw, self.inp_h))

        # print("resize image:",img.shape)
        # cv2.imwrite("resize.jpg",img)
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # img = np.reshape(img, (self.inp_h, tw, 3))
        #
        # img = img.astype(np.float32)
        #
        if self.transform is not None:
            img = self.transform(img)
        # img = (img/255. - self.mean) / self.std
        # img = img.transpose([2, 0, 1])
        return img, idx