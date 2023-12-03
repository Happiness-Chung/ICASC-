import argparse
import os
import shutil
import time
import random
import warnings

import torch
import numpy as np
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
from sklearn.metrics import roc_auc_score
from models.sfocus import sfocus18
from PIL import Image
from datetime import datetime
import torch.nn.functional as F

warnings.filterwarnings("ignore")

# Stella added
import wandb

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--dataset', default='CheXpert' , help='ImageNet, CheXpert, MIMIC, ADNI')
parser.add_argument('--plus', default=True, type=str, 
                    help='(1) whether apply icasc++')
parser.add_argument('--mask', default= False, type=str, 
                    help='(2) whether apply icasc++')
parser.add_argument('--ngpu', default=1, type=int, metavar='G',
                    help='number of gpus to use')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default= 20, type=int, metavar='N',
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
                    help='path to latest checkpoint (default: none)') # 설명상 default가 None이라서 그렇게 바꿨습니다
parser.add_argument("--seed", type=int, default=1234, metavar='BS', help='input batch size for training (default: 64)')
parser.add_argument("--prefix", default="Result", type=str, required=False, metavar='PFX', help='prefix for logging & checkpoint saving')
parser.add_argument('--evaluate',default=False, dest='evaluate', action='store_true', help='evaluation only')
best_prec1 = 0

# Stella added
parser.add_argument('--base_path', default = 'History', type=str, help='base path for Stella (you have to change)')
parser.add_argument('--wandb_key', default='108101f4b9c3e31a235aa58307d1c6b548cfb54a', type=str, help='wandb key for Stella (you have to change). You can get it from https://wandb.ai/authorize')
parser.add_argument('--wandb_mode', default='online', type=str, choices=['online', 'offline'], help='tracking with wandb or turn it off')
parser.add_argument('--wandb_user', default='stellasybae', type=str, help='your wandb username (you have to change)')
parser.add_argument('--experiment_name', default='231102_no_mask', type=str, help='your wandb experiment name (you have to change)')
parser.add_argument('--wandb_project', default='ICASC++', type=str, help='your wandb project name (you have to change)')

parser.add_argument('--layer_depth', default=1, type=int, help='depth of last layer')
parser.add_argument('--bw_loss_setting', default='simple', type=str, choices=['simple', 'exponential', 'exponential_and_temperature'])
parser.add_argument('--temperature', default=5, type=int)

global result_dir
global probs 
global gt    
global k

def main():
    global args, best_prec1
    global viz, train_lot, test_lot
    args = parser.parse_args()
    print ("args", args)
    base_path = args.base_path
    os.environ["WANDB_API_KEY"] = args.wandb_key
    os.environ["WANDB_MODE"] = args.wandb_mode
    wandb.init(project=args.wandb_project, entity=args.wandb_user, reinit=True, name=args.experiment_name)

    now = datetime.now()
    result_dir = os.path.join(base_path, 'results', args.experiment_name) #"{}_{}H".format(now.date(), str(now.hour)))
    os.makedirs(result_dir, exist_ok=True)
    c = open(result_dir + "/config.txt", "w")
    c.write("plus: {}, dataset: {}, epochs: {}, lr: {}, momentum: {},  weight-decay: {}, seed: {}".format(args.plus, args.dataset, str(args.epochs), str(args.lr), str(args.momentum),str(args.weight_decay), str(args.seed)))
    open(result_dir + "/performance.txt", "w")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    
    train_dataset, val_dataset, num_classes, unorm = get_datasets(args.dataset)
    
    # create model
    kwargs = vars(args) # Namespace to Dict
    #model = sfocus18(args.dataset, num_classes, pretrained=False, plus=args.plus, **kwargs)
    model = sfocus18(pretrained=False, **kwargs)
    # define loss function (criterion) and optimizer
    if args.dataset == 'ImageNet':
        criterion = nn.CrossEntropyLoss().cuda()
    elif args.dataset == 'CheXpert' or args.dataset == 'MIMIC' or args.dataset == 'ADNI':
        criterion = torch.nn.BCEWithLogitsLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.milestones, 0.1)
    model = torch.nn.DataParallel(model, device_ids=list(range(args.ngpu)))
    #model = torch.nn.DataParallel(model).cuda()
    model = model.cuda()

    if args.mask == True:
        mask_model = sfocus18(pretrained=False, **kwargs)
        mask_model = torch.nn.DataParallel(mask_model, device_ids=list(range(args.ngpu)))
        mask_model = mask_model.cuda()
        # optionally resume from a checkpoint
        mask_model.load_state_dict(torch.load(os.path.join(base_path, 'Results/2023-10-31_10H/model.pth')))


    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    # optionally resume from a checkpoint
    if args.resume:
        model.load_state_dict(torch.load(os.path.join(base_path, '2023-11-06_9H/model.pth')))

    cudnn.benchmark = True

    # Data loading code
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=args.workers, pin_memory=True)
    if args.evaluate:
        validate(val_loader, model, criterion, unorm, -1, PATH)
        return
    PATH = os.path.join('./checkpoints/SF', args.dataset, args.prefix)
    os.makedirs(PATH, exist_ok=True)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    wandb.watch(model, log='all', log_freq=10)
    for epoch in range(args.start_epoch, args.epochs):
        
        # train for one epoch
        if args.mask == True:
            train(train_loader, model, criterion, optimizer, epoch, result_dir, mask_model)
        else:
            train(train_loader, model, criterion, optimizer, epoch, result_dir)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, unorm, epoch, PATH, result_dir)
        
        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, PATH)
        scheduler.step()
        ## wandb updates its log per single epoch ##
        

        torch.save(model.state_dict(), result_dir + "/model.pth" )

def train(train_loader, model, criterion, optimizer, epoch, dir, mask_model = None):

    global result_dir
    global probs 
    global gt
    global k

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    black_and_white_losses = AverageMeter()

    train_loader_examples_num = len(train_loader.dataset)
    if args.dataset == 'CheXpert':
        label_dim=10
    elif args.dataset=='ADNI': 
        label_dim=3
    probs = np.zeros((train_loader_examples_num, label_dim), dtype = np.float32)
    gt = np.zeros((train_loader_examples_num, label_dim), dtype = np.float32)
    k = 0

    # switch to train mode
    model.train()

    end = time.time()
    for i, (inputs, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        optimizer.zero_grad()
        
        target = target.cuda()
        inputs = inputs.cuda()

        if args.mask == True:
            output, l1, l2, l3, hmaps, _, bw = mask_model(inputs, target)
            inputs = mask_img(inputs, hmaps)
            inputs = torch.nan_to_num(inputs)
        
        # compute output
        if args.plus == False:
            output, l1, l2, l3, _, _ = model(inputs, target)
            loss = criterion(output, target)+l1+l2+l3
        else:
            output, l1, l2, l3, _, _, bw = model(inputs, target)
            loss = criterion(output, target) + l1 + l2 + l3 + bw

        # measure accuracy and record loss
        prec1, prec5 = accuracy(args.dataset, output.data, target, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))
        black_and_white_losses.update(bw.item(), inputs.size(0))

        
        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
    
    if args.dataset == 'ImageNet':
        wandb.log({
        "Epoch":epoch,
        "Train loss":losses.avg,
        "Train Top 1 ACC":top1.avg,
        "Train Top 5 ACC":top5.avg,
    }) 
    elif args.dataset == 'CheXpert' or args.dataset == 'MIMIC' or args.dataset=='ADNI': 
        auc = roc_auc_score(gt, probs)
        print("Training AUC: {}". format(auc))
        wandb.log({
        "Epoch":epoch,
        "Train loss":losses.avg,
        "train AUC":auc,
        "train BW loss": black_and_white_losses.avg
    })   
    
def mask_img(imgs, grad_cam_map):

    for i in range(len(imgs)):
        img = imgs[i]
        mask = get_mask(grad_cam_map, i)
        masked_img = img * mask
        imgs[i] = masked_img
    
    return imgs
    

def get_mask(grad_cam_map, idx):

    grad_cam_map = grad_cam_map[idx].unsqueeze(dim=0)
    grad_cam_map = F.interpolate(grad_cam_map, size=(32, 32), mode='bilinear', align_corners=False) # (1, 1, W, H)
    map_min, map_max = grad_cam_map.min(), grad_cam_map.max()
    grad_cam_map = (grad_cam_map - map_min).div(map_max - map_min).data # (1, 1, W, H), min-max scaling
    grad_cam_map = torch.cat([grad_cam_map.squeeze().unsqueeze(dim=0), grad_cam_map.squeeze().unsqueeze(dim=0), grad_cam_map.squeeze().unsqueeze(dim=0)])

    return grad_cam_map

def vis_heatmaps(hmaps, inputs, unnorm, epoch, path):
    f_shape = hmaps[0].shape
    i_shape = inputs[0].shape
    img_tensors = []
    for idx, image in enumerate(inputs):
        hmap = hmaps[idx]
        if f_shape[0] == 1:
            hmap = torch.cat((hmap, torch.zeros(2, f_shape[1], f_shape[2])))
        hmap = (transforms.ToPILImage()(hmap)).resize((i_shape[1], i_shape[2]))
        pil_image = transforms.ToPILImage()(torch.clamp(unnorm(image), 0, 1))
        res = Image.blend(pil_image, hmap, 0.5)
        img_tensors.append(transforms.ToTensor()(res))
    save_image(img_tensors, '{}/{}.png'.format(path, epoch), nrow=8)

def validate(val_loader, model, criterion, unorm, epoch, PATH, dir):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    black_and_white_losses = AverageMeter()

    global probs 
    global gt
    global k

    val_loader_examples_num = len(val_loader.dataset)
    if args.dataset == 'CheXpert':
        label_dim=10
    elif args.dataset=='ADNI': 
        label_dim=3
    probs = np.zeros((val_loader_examples_num, label_dim), dtype = np.float32)
    gt = np.zeros((val_loader_examples_num, label_dim), dtype = np.float32)
    k = 0

    # switch to evaluate mode
    model.eval()
    end = time.time()
    for i, (inputs, target) in enumerate(val_loader):
        target = target.cuda()
        inputs = inputs.cuda()
        
        # compute output
        if args.plus == False:
            output, l1, l2, l3, hmaps, _  = model(inputs, target)
            loss = criterion(output, target)+l1 + l2 + l3
        else:
            output, l1, l2, l3, hmaps, _, bw = model(inputs, target)
            loss = criterion(output, target)+l1+l2+l3+bw
        # measure accuracy and record loss
        prec1, prec5 = accuracy(args.dataset, output.data, target, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))
        black_and_white_losses.update(bw.item(), inputs.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    if args.dataset == 'ImageNet':
        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
                .format(top1=top1, top5=top5))
        wandb.log({
            "Valid loss":losses.avg,
            "Valid Top 1 ACC":top1.avg,
            "Valid Top 5 ACC":top5.avg,
        })   
        f = open(dir + "/performance.txt", "a")
        f.write(str(top1.avg.item()) + "\n")
        f.close()
    elif args.dataset == 'CheXpert' or args.dataset == 'MIMIC' or args.dataset=='ADNI': 
        auc = roc_auc_score(gt, probs)
        print("ValidAUC: {}". format(auc))
        wandb.log({
        "Epoch":epoch,
        "Valid loss":losses.avg,
        "valid AUC":auc,
        "valid BW loss": black_and_white_losses.avg
        })
        f = open(dir + "/performance.txt", "a")
        f.write(str(auc) + "\n")
        f.close()   

    return top1.avg


def save_checkpoint(state, is_best, path):
    filename='{}/checkpoint.pth.tar'.format(path)
    if is_best:
        torch.save(state, filename)


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



def accuracy(dataset, output, target, topk=(1,)):
    
    """Computes the precision@k for the specified values of k"""
    sigmoid = torch.nn.Sigmoid()
    res = []
    global probs 
    global gt
    global k
    
    if dataset == 'ImageNet':
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
    
    elif dataset == 'CheXpert' or dataset == 'MIMIC' or dataset == 'ADNI':
        
        # For AUC ROC
        probs[k: k + output.shape[0], :] = output.cpu()
        gt[   k: k + output.shape[0], :] = target.cpu()
        k += output.shape[0] 
        
        # For accuracy
        preds = np.round(sigmoid(output).cpu().detach().numpy())
        targets = target.cpu().detach().numpy()
        test_sample_number = len(targets)* len(output[0])
        test_correct = (preds == targets).sum()
        
        res.append([test_correct / test_sample_number * 100])
        res.append([0])
    
    return res


if __name__ == '__main__':
    main()
