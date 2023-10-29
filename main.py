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

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--dataset', default='cifar10' , help='Dataor Integral Object Attention githubor Integral Object Attention githubset to train')
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

    now = datetime.now()
    result_dir = "C:/Users/rhtn9/OneDrive/바탕 화면/code/ICASC++/result/{}_{}H".format(now.date(), str(now.hour))
    os.makedirs(result_dir)
    c = open(result_dir + "/config.txt", "w")
    c.write("dataset: {}, epochs: {}, lr: {}, momentum: {},  weight-decay: {}, seed: {}".format(args.dataset, str(args.epochs), str(args.lr), str(args.momentum),str(args.weight_decay), str(args.seed)))
    open(result_dir + "/performance.txt", "w")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    
    train_dataset, val_dataset, num_classes, unorm = get_datasets(args.dataset)
    # create model
    model = sfocus18(num_classes, pretrained=False, plus=True)
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.milestones, 0.1)
    model = torch.nn.DataParallel(model, device_ids=list(range(args.ngpu)))
    #model = torch.nn.DataParallel(model).cuda()
    model = model.cuda()

    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


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

    for epoch in range(args.start_epoch, args.epochs):
        
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, result_dir)
        
        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, unorm, epoch, PATH)
        
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

    torch.save(model.state_dict(), result_dir + "/model.pth" )

def train(train_loader, model, criterion, optimizer, epoch, dir):

    global result_dir

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (inputs, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        optimizer.zero_grad()
        
        target = target.cuda()
        inputs = inputs.cuda()
        
        # compute output
        output, l1, l2, l3, _ = model(inputs, target)
        loss = criterion(output, target)+l1+l2+l3
        
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))
        
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
            
    f = open(dir + "/performance.txt", "a")
    f.write(str(top1) + "\n")
    f.close()


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

def validate(val_loader, model, criterion, unorm, epoch, PATH):
      batch_time = AverageMeter()
      losses = AverageMeter()
      top1 = AverageMeter()
      top5 = AverageMeter()

      # switch to evaluate mode
      model.eval()
      end = time.time()
      for i, (inputs, target) in enumerate(val_loader):
        target = target.cuda()
        inputs = inputs.cuda()
        
        # compute output
        output, l1, l2, l3, hmaps  = model(inputs, target)
        loss = criterion(output, target)+l1 + l2 + l3
        if i == 0:    #SAVE HEATMAP
            vis_heatmaps(hmaps.cpu(), inputs.cpu(), unorm, epoch, PATH)
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))
        
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
    
      print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
            .format(top1=top1, top5=top5))

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



def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
