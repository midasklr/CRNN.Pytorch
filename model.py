# -*- coding: utf-8 -*-
# @Time : 2020/12/1 下午1:35 
# @Author : midaskong 
# @File : model.py 
# @Description:

from __future__ import print_function, absolute_import
import torch.utils.data as data
import numpy as np
import cv2
import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import alphabet
import torch.optim as optim
import time


class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out


class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, in_size // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size // reduction, in_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size),
            hsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class Block(nn.Module):
    '''expand + depthwise + pointwise'''

    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(Block, self).__init__()
        self.stride = stride
        self.se = semodule

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride,
                               padding=kernel_size // 2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_size),
            )

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


class BidirectionalLSTM(nn.Module):
    # Inputs hidden units Out
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


class CRNN(nn.Module):
    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)  # 32*96
        self.bn1 = nn.BatchNorm2d(32)
        self.hs1 = hswish()
        self.bneck = nn.Sequential(
            Block(3, 32, 32, 32, nn.ReLU(inplace=True), SeModule(32), 1),  # 16*48
            Block(3, 32, 72, 48, nn.ReLU(inplace=True), None, 2),  # 8*24
            Block(3, 48, 88, 48, nn.ReLU(inplace=True), None, 1),
            Block(3, 48, 96, 64, hswish(), SeModule(64), 2),  # 4*12
            Block(3, 64, 240, 96, hswish(), SeModule(96), 1),
            Block(3, 96, 256, 128, hswish(), SeModule(128), 1),
            Block(3, 128, 512, 256, hswish(), SeModule(256), (2, 1)),  # 2*12
            Block(3, 256, 576, 256, hswish(), SeModule(256), 1),
        )
        self.conv2 = nn.Conv2d(256, 512, kernel_size=2, stride=1, padding=0, bias=False)  # 1*12
        self.bn2 = nn.BatchNorm2d(512)
        self.hs2 = hswish()

        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            # elif isinstance(m, nn.LSTM):
            #     # init.xavier_normal(m._all_weights)
            #     if m.bias is not None:
            #         init.constant_(m.bias, 0)


    def forward(self, input):
        #         print("input size : ",input.size())
        conv = self.hs1(self.bn1(self.conv1(input)))
        #         print(conv.size())
        conv = self.bneck(conv)
        #         print(conv.size())
        conv = self.hs2(self.bn2(self.conv2(conv)))

        # conv features
        #         conv = self.cnn(input)
        b, c, h, w = conv.size()
        # print(conv.size())
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)  # b *512 * width
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        output = F.log_softmax(self.rnn(conv), dim=2)
        return output
