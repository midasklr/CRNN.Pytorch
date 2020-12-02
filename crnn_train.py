# -*- coding: utf-8 -*-
# @Time : 2020/11/30 下午6:28
# @Author : midaskong
# @File : train_varlen.py
# @Description:

from __future__ import print_function, absolute_import
import numpy as np
import cv2
import os
import torch
from torch.utils.data import DataLoader
from model import *
from dataloadershuffle import *
from utils import *
import argparse
from alphabet import *
import torchvision.transforms as transforms
import torch.optim as optim
import time

parser = argparse.ArgumentParser(
    description='CRNN Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--num_class', default=3, type=int, help='number of class in ur dataset')
parser.add_argument('--data_root', default='./data/han',
                    help='Dataset root directory path')
parser.add_argument('--datalist',default={'train': './data/han.train.txt', 'val': './data/han.val.txt'},type=dict,
                    help='train val datalist')
parser.add_argument('--maxlen', default=10, type=int, help='max lenght of label in dataset')
parser.add_argument('--resize_h', default=32, type=int, help='target height of resized image')
parser.add_argument('--train_epoch', default=15, type=int, help='number of epochs to train')
parser.add_argument('--lr_steps', default=[7,12], type=list, help='lr update steps')
parser.add_argument('--warm_up_iter', default=100, type=int, help='lr warmup iter')
parser.add_argument('--batchsize', default=64, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--lr_ini', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--lr_factor', default=0.1, type=float,
                    help='Gamma update for AdamW')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()


if __name__=='__main__':
    alphabet = alphabet
    print(len(alphabet))
    train_dataset = CRNNData(args.data_root, args.resize_h, args.maxlen, args.batchsize, args.datalist,transform=transforms.Compose([transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation =0.1, hue=0) , transforms.RandomRotation(5),
                                                                      transforms.ToTensor(),transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                                                      ]))
    val_dataset = CRNNData(args.data_root, args.resize_h, args.maxlen, args.batchsize, args.datalist, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]), is_train=False)
    train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=args.batchsize,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        )
    converter = strLabelConverter(alphabet)
    net = CRNN(args.resize_h,3,len(alphabet)+1,256)
    net = net.to("cuda")
    print(net)
    criterion = torch.nn.CTCLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()),lr=args.lr_ini)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_steps, args.lr_factor, -1)
    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)
    best_acc = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        if 'state_dict' in checkpoint.keys():
            net.load_state_dict(checkpoint['state_dict'])
            last_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        else:
            net.load_state_dict(checkpoint)
    for epoch in range(args.train_epoch):
        train( train_loader, train_dataset, converter, net, criterion, optimizer, epoch, args.warm_up_iter, args.lr_ini)
        lr_scheduler.step()
        acc = validate(val_loader, val_dataset, converter, net, criterion, epoch, args.batchsize)
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        print("\tIs best:", is_best)
        print("\tBest acc is:", best_acc)
        print("="*150)
        # save checkpoint
        torch.save(
            {
                "state_dict": net.state_dict(),
                "epoch": epoch + 1,
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "best_acc": best_acc,
            },  os.path.join(args.save_folder, "checkpoint_{}_acc_{:.4f}.pth".format(epoch, acc))
        )
