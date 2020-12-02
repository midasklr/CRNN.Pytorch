# -*- coding: utf-8 -*-
# @Time : 2020/11/30 下午6:07 
# @Author : midaskong 
# @File : demo.py 
# @Description:

from __future__ import print_function, absolute_import
import numpy as np
import cv2
import os
import torch
from model import *
from alphabet2 import *
from utils import *
if __name__=='__main__':
    alphabet = alphabet2
    # print(alphabet)
    resize_h = 32
    resize_w = 512
    net = CRNN(32,3,len(alphabet)+1,256)
    net.load_state_dict(torch.load("/home/hwits/Documents/TextRecognition/CRNN.Pytorch/weights/checkpoint_56_acc_0.6162.pth")['state_dict'])
    net.to("cuda")
    net.eval()
    converter = strLabelConverter(alphabet)
    image = cv2.imread("/home/hwits/Documents/TextRecognition/CRNN.Pytorch/Screenshot from 2020-12-02 09-55-13.png")
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # print("image : ",image.shape)
    rw = image.shape[1]*32/image.shape[0]
    # print("rw:",rw)
    image = cv2.resize(image, (int(rw),resize_h))
    image = np.reshape(image, (resize_h, int(rw), 3))
    image = image.astype(np.float32)
    image = (image/255. - 0.5) / 0.5
    image = image.transpose([2, 0, 1])
    image = torch.Tensor(image)
    image = torch.unsqueeze(image, dim=0)

    with torch.no_grad():
        image = image.to("cuda")
        preds = net(image).cpu()
        # print("preds:",preds.size())
        # print(preds)
        preds_size = torch.IntTensor([preds.size(0)])
        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        print(preds)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        print("Pred: %s"%sim_preds)
