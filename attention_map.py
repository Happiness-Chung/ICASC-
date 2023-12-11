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
import math

import pandas as pd
import warnings

warnings.filterwarnings(action='ignore')

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--dataset', default='CheXpert' , help='Dataor Integral Object Attention githubor Integral Object Attention githubset to train')
parser.add_argument('--plus', default= True, type=str, 
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


# Stella added
parser.add_argument('--base_path', default = 'History', type=str, help='base path for Stella (you have to change)')
parser.add_argument('--wandb_key', default='108101f4b9c3e31a235aa58307d1c6b548cfb54a', type=str, help='wandb key for Stella (you have to change). You can get it from https://wandb.ai/authorize')
parser.add_argument('--wandb_mode', default='online', type=str, choices=['online', 'offline'], help='tracking with wandb or turn it off')
parser.add_argument('--wandb_user', default='stellasybae', type=str, help='your wandb username (you have to change)')
parser.add_argument('--experiment_name', default='231102_no_mask', type=str, help='your wandb experiment name (you have to change)')
parser.add_argument('--wandb_project', default='ICASC++', type=str, help='your wandb project name (you have to change)')

parser.add_argument('--model_weight', default='NIH_1_parallel', type=str)
                    
                    
parser.add_argument('--layer_depth', default=1, type=int, help='depth of last layer')
parser.add_argument('--bw_loss_setting', default='simple', type=str, choices=['simple', 'exponential', 'exponential_and_temperature'])
parser.add_argument('--temperature', default=5, type=int)

parser.add_argument('--test', action='store_true')

best_prec1 = 0

global result_dir

def main():
    global args, best_prec1
    global viz, train_lot, test_lot
    args = parser.parse_args()
    print ("args", args)
    kwargs = vars(args) # Namespace to Dict
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    
    train_dataset, val_dataset, num_classes, unorm = get_datasets(args.dataset)
    # create model
    #model = sfocus18(args.dataset, num_classes, pretrained=False, plus=args.plus, test = True)
    model = sfocus18(pretrained=False, **kwargs)
    model = torch.nn.DataParallel(model, device_ids=list(range(args.ngpu)))
    #model = torch.nn.DataParallel(model).cuda()
    model = model.cuda()

    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    # optionally resume from a checkpoint
    # model.load_state_dict(torch.load('./results/{0}/model.pth'.format(args.experiment_name)))
    model.load_state_dict(torch.load('./weights/{0}.pth'.format(args.model_weight)))
    cudnn.benchmark = True

    # Data loading code
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=args.workers, pin_memory=True)
    
    validate(val_loader, model)

def get_hscore(true,false):
    #print(true.min(), true.max())
    true = (true - true.min()) / (true.max() - true.min() + 0.0000001)
    false = (false - false.min()) / (false.max() - false.min() + 0.0000001)
    #print((torch.abs(2 * true - false) - false))
    h_score = ((torch.abs(2 * true - false) - false) / (2*150*150) * 100).sum().item()
    if math.isnan(h_score):
        h_score = 0
    return h_score

def create_binary_mask(heatmap, threshold=0.5):
    # Grad-CAM 히트맵을 이진 마스크로 변환
    binary_mask = np.where(heatmap >= threshold, 1, 0)
    return binary_mask

def calculate_iou(binary_mask, x, y, h, w):
    # BBOX를 이진 마스크로 변환
    bbox_mask = np.zeros_like(binary_mask)
    bbox_mask[y-h//2:y+h//2, x-w//2:x+w//2] = 1
    # 교차 영역과 합집합 영역 계산
    intersection = np.logical_and(binary_mask, bbox_mask).sum()
    union = np.logical_or(binary_mask, bbox_mask).sum()

    # IoU 계산
    iou = intersection / union
    return iou

def save_cam(name, torch_img, grad_cam_map, index, args, conf = False):
    args = parser.parse_args()
    result_dir = os.path.join('./results', args.experiment_name)
    os.makedirs(result_dir, exist_ok=True)
    
    if args.dataset == 'NIH':
        bbox_df = pd.read_csv('./data/NIH/BBox_List_2017.csv')
        file_name = './results/'+args.experiment_name+'/'+args.experiment_name+'_IOU.txt'
        content = ''
        with open(file_name, 'w') as file:
            file.write(content)
    # print(grad_cam_map)
    grad_cam_map = grad_cam_map[0].unsqueeze(dim=0)
    grad_cam_map = F.interpolate(grad_cam_map, size=(150, 150), mode='bilinear', align_corners=False) # (1, 1, W, H)
    map_min, map_max = grad_cam_map.min(), grad_cam_map.max()
    grad_cam_map = (grad_cam_map - map_min).div(map_max - map_min).data # (1, 1, W, H), min-max scaling
    
    #grad_cam_map = grad_cam_map.squeeze() # : (224, 224)
    grad_heatmap = cv2.applyColorMap(np.uint8(255 * grad_cam_map.squeeze().cpu()), cv2.COLORMAP_JET) # (W, H, 3), numpy 
    grad_heatmap = torch.from_numpy(grad_heatmap).permute(2, 0, 1).float().div(255) # (3, W, H)
    b, g, r = grad_heatmap.split(1)
    grad_heatmap = torch.cat([r, g, b]) # (3, 244, 244), opencv's default format is BGR, so we need to change it as RGB format.

    grad_result = grad_heatmap + torch_img.cpu() # (1, 3, W, H)
    grad_result = grad_result.div(grad_result.max()).squeeze() # (3, W, H)
    
    
    #### grad result & BBOX IOU computation ####
    if args.dataset == 'NIH':
        if name in list(bbox_df['Image Index']):
            x = bbox_df[bbox_df['Image Index'] == name]['Bbox [x'].values[0]
            y = bbox_df[bbox_df['Image Index'] == name]['y'].values[0]
            w = bbox_df[bbox_df['Image Index'] == name]['w'].values[0]
            h = bbox_df[bbox_df['Image Index'] == name]['h]'].values[0]

            binary_mask = create_binary_mask(grad_result)
            iou = calculate_iou(binary_mask, x, y, h, w)
            result = '{0} : {1} \n'.format(name.split('.')[0], iou)

            with open(file_name, 'a') as file:
                file.write(result)
    
    os.makedirs(result_dir+'/attention_map', exist_ok=True)
    if conf == False:
        save_image(grad_result, result_dir+'/attention_map/name_{}_result{}_true.png'.format(name.split('.')[0], index))
    else:
        save_image(grad_result,result_dir+'/attention_map/name_{}_result{}_false.png'.format(name.split('.')[0], index))

def validate(val_loader, model):
    batch_time = AverageMeter()

    # switch to evaluate mode
    model.eval() 
    end = time.time()
    h_score = 0
    args = parser.parse_args()
    for i, (name, inputs, target) in enumerate(val_loader):
        
        target = target.cuda()
        inputs = inputs.cuda()
        
        # compute output
        if args.plus == False:
            hmaps, hmaps_conf = model(inputs, target)
        else:
            hmaps, hmaps_conf = model(inputs, target)
        
        #save_image(inputs[0], 'C:/Users/rhtn9/OneDrive/바탕 화면/code/ICASC++/result/imgs/{}.jpg'.format(i+50))
        for j in range(len(hmaps)):
            # if i*len(inputs) + j < 5000:
            save_cam(name[j], inputs[j], hmaps[j], i*len(inputs) + j, args)
            save_cam(name[j], inputs[j], hmaps_conf[j], i*len(inputs) + j, args, conf=True)
            h_score += get_hscore(hmaps[j], hmaps_conf[j])


    print("H-score of {0}: ".format(args.experiment_name), h_score/len(val_loader))
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
