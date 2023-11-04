import argparse
import os
import shutil
import time
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torchvision.utils import make_grid, save_image
from data_utils import get_datasets
from models.sfocus import sfocus18
from PIL import Image
from datetime import datetime
import torch.nn.functional as F
import numpy as np
import cv2
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--dataset', default='cifar10' , help='Dataor Integral Object Attention githubor Integral Object Attention githubset to train')
parser.add_argument('--plus', default=False, type=str, 
                    help='whether apply icasc++')
parser.add_argument('--ngpu', default=1, type=int, metavar='G',
                    help='number of gpus to use')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--milestones', type=int, default=[50,100], nargs='+', help='LR decay milestones')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default=False, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument("--seed", type=int, default=1234, metavar='BS', help='input batch size for training (default: 64)')
parser.add_argument("--prefix", default="Result", type=str, required=False, metavar='PFX', help='prefix for logging & checkpoint saving')
parser.add_argument('--evaluate',default=False, dest='evaluate', action='store_true', help='evaluation only')
best_prec1 = 0

global result_dir

def main():
    global args, best_prec1
    global viz, train_lot, test_lot
    args = parser.parse_args()
    print ("args", args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    
    train_dataset, val_dataset, num_classes, unorm = get_datasets(args.dataset)
    # create model
    model = sfocus18(num_classes, pretrained=False, plus=args.plus)

    model = torch.nn.DataParallel(model, device_ids=list(range(args.ngpu)))
    #model = torch.nn.DataParallel(model).cuda()
    model = model.cuda()

    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    # optionally resume from a checkpoint
    model.load_state_dict(torch.load('C:/Users/rhtn9/Results/2023-10-30_16H/model.pth'))

    cudnn.benchmark = True

    # Data loading code
    train_sampler = None
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=args.workers, pin_memory=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    
    validate(val_loader, model)

def mask_img(imgs, grad_cam_map, idx):

    for i in range(len(imgs)):
        img = imgs[i]
        grad_cam_map = get_mask(grad_cam_map, i)
        masked_img = img * grad_cam_map
        save_image(masked_img, "C:/Users/rhtn9/OneDrive/바탕 화면/code/ICASC++/result/masked_imgs/{}.jpg".format(idx * 32 + i))
    

def get_mask(grad_cam_map, idx):

    grad_cam_map = grad_cam_map[idx].unsqueeze(dim=0)
    grad_cam_map = F.interpolate(grad_cam_map, size=(32, 32), mode='bilinear', align_corners=False) # (1, 1, W, H)
    map_min, map_max = grad_cam_map.min(), grad_cam_map.max()
    grad_cam_map = (grad_cam_map - map_min).div(map_max - map_min).data # (1, 1, W, H), min-max scaling
    grad_cam_map = torch.cat([grad_cam_map.squeeze().unsqueeze(dim=0), grad_cam_map.squeeze().unsqueeze(dim=0), grad_cam_map.squeeze().unsqueeze(dim=0)])

    return grad_cam_map

def train(train_loader, model, criterion, optimizer, epoch, dir):

    global result_dir

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.eval()

    end = time.time()
    for i, (inputs, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        optimizer.zero_grad()
        
        target = target.cuda()
        inputs = inputs.cuda()
        
        # compute output
        if args.plus == False:
            output, l1, l2, l3, hmaps, _ = model(inputs, target)
            loss = criterion(output, target)+l1+l2+l3
        else:
            output, l1, l2, l3, hmaps, _, bw = model(inputs, target)
            loss = criterion(output, target)+l1+l2+l3+bw

        mask_img(inputs, hmaps,i)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
def validate(val_loader, model):
      batch_time = AverageMeter()

      # switch to evaluate mode
      model.eval()
      end = time.time()
      for i, (inputs, target) in enumerate(val_loader):
        
        if i == 50:
            break
        
        target = target.cuda()
        inputs = inputs.cuda()
        
        # compute output
        if args.plus == False:
            output, l1, l2, l3, hmaps, hmaps_conf  = model(inputs, target)
        else:
            output, l1, l2, l3, hmaps, hmaps_conf, bw = model(inputs, target)
        

        mask_img(inputs, hmaps,i)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    main()
