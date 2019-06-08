import argparse
import os, sys, json, math
import shutil
import time, copy
from tqdm import tqdm

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import Dataset, DataLoader

import numpy as np

import torchvision
from torchvision import datasets, models, transforms, utils
from inceptionresnetv2.pytorch_load import inceptionresnetv2
from inceptionv4.pytorch_load import inceptionv4
from PIL import Image
import cPickle as pkl

import warnings
warnings.filterwarnings("ignore")

from dataset import Dataset

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--data', metavar='DIR', default='../data',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50')
parser.add_argument('--optim', default='sgd', type=str)
parser.add_argument('--prefix', default='', type=str, metavar='P',
                    help='prefix of this training')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--ndf', default=64, type=int)
parser.add_argument('--max_mult', default=8, type=int)
parser.add_argument('--n_layers_D', default=3, type=int)
parser.add_argument('--feat_len', default=256, type=int)
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr-mult', '--learning-rate-mult', default=1.0, type=float,
                    metavar='LRM', help='learning rate mult')
parser.add_argument('--min-lr', '--min-learning-rate', default=0.000001, type=float,
                    metavar='MLR', help='minimum learning rate')
parser.add_argument('--decay-epoch', default=30, type=int,
                    metavar='DE', help='learning rate decay epoch')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 100)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', default=False, action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('-p', '--pred', dest='pred', default=False, action='store_true',
                    help='evaluate model on test set and output to arch-test.pkl')
parser.add_argument('--pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--no_eval', action='store_true')
parser.add_argument('-t', '--toy', dest='toy', default=False, action='store_true',
                    help='using a toy dataset of iMat')
parser.add_argument('--debug', dest='debug', default=False, action='store_true')
parser.add_argument('--all', dest='all', default=False, action='store_true',
                    help='using a train-val as train dataset of iMat')
parser.add_argument('--lr-decay-epoch', nargs='+', type=int)
parser.add_argument('--lr-decay', default='epoch', type=str)
parser.add_argument('--split-task', default='pass', type=str)
parser.add_argument('--test-crop', default=False, type=bool)
best_prec1 = 0


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'densenet121': 'https://download.pytorch.org/models/densenet121-241335ed.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-6f0f7f60.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-4c113574.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-17b70270.pth',
}

import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torchvision.models.resnet import Bottleneck, ResNet
from sklearn import metrics

def initialize_weights(model):
    if type(model) in [nn.Linear]:
        nn.init.xavier_normal(model.weight.data, gain=1)
    elif type(model) in [nn.BatchNorm2d]:
        nn.init.uniform(model.weight.data, 1.0, 0.02)
        nn.init.constant(model.bias.data, 0.0)
    elif type(model) in [nn.Conv2d]:
        nn.init.xavier_normal(model.weight.data, gain=1)
    for i in model.children():
            initialize_weights(i)

def print_network_shape(model, input_size):
    class TablePrinter(object):
        def __init__(self, fmt, sep=' ', ul=None):
            super(TablePrinter,self).__init__()
            self.fmt   = str(sep).join('{lb}{0}:{1}{rb}'.format(key, width, lb='{', rb='}') for heading,key,width in fmt)
            self.head  = {key:heading for heading,key,width in fmt}
            self.ul    = {key:str(ul)*width for heading,key,width in fmt} if ul else None
            self.width = {key:width for heading,key,width in fmt}

        def row(self, data):
            return self.fmt.format(**{ k:str(data.get(k,''))[:w] for k,w in self.width.iteritems() })

        def __call__(self, dataList):
            _r = self.row
            res = [_r(data) for data in dataList]
            res.insert(0, _r(self.head))
            if self.ul:
                res.insert(1, _r(self.ul))
            return '\n'.join(res)

    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split('.')[-1].split("'")[0]
            module_idx = len(summary)

            idx = module_idx+1
            m_key = '%i-%s' % (idx, class_name)
            tmp = {}
            tmp['name'] = m_key
            tmp['input_shape'] = list(input[0].size())
            tmp['output_shape'] = list(output.size())
            tmp['nb_params'] = 0
            for param in module.parameters():
                tmp['nb_params'] += param.numel()
            summary.append(tmp)

        if not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList) and not (module == model):
            hooks.append(module.register_forward_hook(hook))

    # check if there are multiple inputs to the network
    if isinstance(input_size[0], (list, tuple)):
        x = [Variable(torch.rand(1,*in_size)).cuda() for in_size in input_size]
    else:
        x = Variable(torch.rand(1,*input_size)).cuda()

    # create properties
    summary = []
    hooks = []
    # register hook
    model.apply(register_hook)
    # make a forward pass
    if isinstance(input_size[0], (list, tuple)):
        model(*x)
    else:
        model(x)
    # remove these hooks
    for h in hooks:
        h.remove()

    fmt = [('Name', 'name', 30), ('In Shape', 'input_shape', 25), ('Out Shape', 'output_shape', 25),('Params', 'nb_params', 15)]
    print( TablePrinter(fmt, ul='=')(summary) )

    return summary

def print_network(net, input_shape=None):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)
    print_network_shape(net, input_shape)

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1) 

class MyNet(nn.Module):
    def __init__(self, pretrained=False, **kwargs):
        super(MyNet, self).__init__()
        netname = args.arch.split('_')[0]
        if netname == 'resnet152':
            self.model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
        else:
            self.model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
        if pretrained:
            parameters =  model_zoo.load_url(model_urls[netname])
            self.model.load_state_dict(parameters)
        self.model.avgpool= nn.AvgPool2d(8)
        self.model.fc = nn.Linear(1024, 1)
        self.model.sig = nn.Sigmoid()

    def forward(self, x): 
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        # x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.model.fc(x)
        return self.model.sig(x)

class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, max_mult=8, norm_layer=nn.BatchNorm2d, feat_len=256):
        super(NLayerDiscriminator, self).__init__()
        use_bias = norm_layer == nn.InstanceNorm2d
        kw = 3

        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, max_mult)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult)]
            sequence += [nn.LeakyReLU(0.2, True)]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, max_mult)
        sequence += [
        nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                  kernel_size=kw, stride=1, padding=padw, bias=use_bias),
        norm_layer(ndf * nf_mult),
        nn.LeakyReLU(0.2, True)
        ]
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        sequence += [Flatten(), nn.Linear(feat_len, 1)]

        sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)
 
class NLayerDiscriminatorMod(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, max_mult=8, norm_layer=nn.BatchNorm2d, feat_len=256):
        super(NLayerDiscriminatorMod, self).__init__()
        use_bias = False
        kw = 3

        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=7, stride=2, padding=3),
            norm_layer(ndf),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, max_mult)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult)]
            sequence += [nn.LeakyReLU(0.2, True)]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, max_mult)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=3, stride=1, padding=padw)]
        # sequence += [nn.AvgPool2d(7, stride=1)]
        # sequence += [nn.Linear(feat_len, 1)]
        sequence += [Flatten(), nn.Linear(feat_len, 1)]

        sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)

def main():   
    global args, best_prec1

    args = parser.parse_args()


    if args.debug:
        import ipdb; ipdb.set_trace()

    if not args.pred and not args.evaluate and not os.path.exists('%s_%s'%(args.arch, args.prefix)):
        os.makedirs('%s_%s'%(args.arch,args.prefix))

    # create model
    train_part = ''
    print("=> creating model '{}'".format(args.arch))
    if 'dis' not in args.arch:
            model = MyNet()
    elif 'mod' not in args.arch:
            model = NLayerDiscriminator(3, ndf=args.ndf, n_layers = args.n_layers_D, max_mult=args.max_mult, feat_len=args.feat_len)
    else:
            model = NLayerDiscriminatorMod(3, ndf=args.ndf, n_layers = args.n_layers_D, max_mult=args.max_mult, feat_len=args.feat_len)
    initialize_weights(model)

    model = model.cuda()

    print_network(model, (3,128,128))
    # define loss function (criterion) and optimizer
    criterion = nn.BCELoss().cuda()
    # criterion = nn.MSELoss().cuda()

    if args.optim == 'sgd':
        print 'Use SGD'
        optimizer = optim.SGD(model.parameters(),lr=args.lr,momentum=args.momentum,weight_decay=args.weight_decay,nesterov=True)
    else:
        print 'Use Adam'
        optimizer = optim.Adam(model.parameters(),weight_decay=args.weight_decay)

   # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            if args.start_epoch == 0:
                args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    crop_size = 128
    resize_size = 128

    normalize = None
    if not args.pretrained:
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if not args.all:
        train_loader = torch.utils.data.DataLoader(
        Dataset(args.data, 'train', args.toy, transforms.Compose([
                                transforms.Scale(resize_size),
                                transforms.RandomSizedCrop(crop_size),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                normalize
        ]), args.split_task),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    else:
        train_loader = torch.utils.data.DataLoader(
        Dataset(args.data, 'train_val', args.toy, transforms.Compose([
                                transforms.Scale(resize_size),
                                transforms.RandomSizedCrop(crop_size),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                normalize
        ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        Dataset(args.data, 'val', args.toy, transforms.Compose([
            transforms.Scale(resize_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            normalize
        ]), args.split_task),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)



    if args.evaluate:
        print 'Do only eval'
        validate(val_loader, model, criterion)
        #validate(val_loader_old, model, criterion)
        return

    learning_rate_adjust = LearningRateAdjust(max_cool_down = 5, decay_method = args.lr_decay)
    force_decay = False
    # best_epoch = 0
    for epoch in range(args.start_epoch, args.epochs):
        early_stop = learning_rate_adjust.update(optimizer, epoch, force_decay)

        # if early_stop:
            # print 'Reached maximum cool down epoch, stopped.'
            # print 'Best Error %f at %d epoch'%(100.0 - best_prec1, best_epoch)
            # break

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        if not args.no_eval:
                validate(val_loader, model, criterion)
        else:
                print ''

        # remember best prec@1 and save checkpoint
        # is_best = prec1 > best_prec1
        # force_decay = prec1 < best_prec1
        # best_prec1 = max(prec1, best_prec1)
        # if is_best:
            # best_epoch = epoch+1
        if epoch % 100 == 0:
                print 'save epoch ', epoch
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_prec1,
                    'optimizer' : optimizer.state_dict(),
                }, False, filename='%s_%s/chkpt_%d.pth.tar'%(args.arch,args.prefix,epoch))

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    avgprec = AverageMeter()
    avgf1 = AverageMeter()
    avgacc = AverageMeter()
    avgrec = AverageMeter()

    # switch to train mode
    model.train()
    alpha = 0.5
    end = time.time()
    if args.debug:
        import ipdb; ipdb.set_trace()
    for i, (input, target) in enumerate(train_loader):
        target = target.float()
        target = target.view(-1,1)
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda()
        target = target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        output = model(input_var)
        loss = criterion(output, target_var)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        prec, rec, acc, f1= accuracy(output.data, target)
        losses.update(loss.data[0], input.size(0))
        avgacc.update(acc, input.size(0))
        avgprec.update(prec, input.size(0))
        avgf1.update(f1, input.size(0))
        avgrec.update(rec, input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i > 0 and i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f}\t'
                  'Data {data_time.val:.3f}\t'
                  'Loss {loss.avg:.4f}\t'
                  'Rec {avgrec.avg:.3f}\t'
                  'Acc {avgacc.avg:.3f}\t'
                  'F1 {avgf1.avg:.3f}\t'
                  'Prec {avgprec.avg:.3f}\t'.format(
                   epoch, i, len(train_loader),batch_time=batch_time,
                   data_time=data_time,loss=losses, avgacc=avgacc, avgrec=avgrec, avgprec=avgprec, avgf1=avgf1))

    print('Epoch {0} Train\t'
          'Rec {avgrec.avg:.3f}\t'
          'Acc {avgacc.avg:.3f}\t'
          'F1 {avgf1.avg:.3f}\t'
          'Prec {avgprec.avg:.3f}\t'.format(epoch, avgacc=avgacc, avgrec=avgrec, avgprec=avgprec, avgf1=avgf1)),

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    avgprec = AverageMeter()
    avgf1 = AverageMeter()
    avgacc = AverageMeter()
    avgrec = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    if args.debug:
            import ipdb;ipdb.set_trace()
    for i, (input, target) in (enumerate(val_loader)):
        target = target.float()
        target = target.view(-1,1)
        input = input.cuda()
        target = target.cuda()
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec, rec, acc, f1 = accuracy(output.data, target)
        losses.update(loss.data[0], input.size(0))
        avgacc.update(acc, input.size(0))
        avgprec.update(prec, input.size(0))
        avgf1.update(f1, input.size(0))
        avgrec.update(rec, input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        #if i % args.print_freq == 0:
        #    print('Test: [{0}/{1}]\t'
        #          'Loss {loss.avg:.4f}\t'
        #          'Prec@1 {top1.avg:.3f}\t'
        #          'Prec@5 {top5.avg:.3f}'.format(
        #           i, len(val_loader), loss=losses,
        #           top1=top1, top5=top5))

    print('\tVal\t'
          'Rec {avgrec.avg:.3f}\t'
          'Acc {avgacc.avg:.3f}\t'
          'F1 {avgf1.avg:.3f}\t'
          'Prec {avgprec.avg:.3f}\t'.format(
          avgacc=avgacc, avgrec=avgrec, avgprec=avgprec, avgf1=avgf1))

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, '%s_%s/best.pth.tar'%(args.arch,args.prefix))

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

class LearningRateAdjust(object):
    def __init__(self, decay_rate = 0.1, cool_down = 1, max_cool_down = 3, decay_method = 'epoch'):
        self.decay_rate  = decay_rate
        self.decay_method = decay_method
        self.cool_down = cool_down
        self.max_cool_down = max_cool_down
        self.cool_down_count = 0
        self.current_lr = 0.1
        self.early_stop = False

    def update(self, optimizer, epoch, force_decay=False):
        if self.decay_method == 'epoch':
            lr = max(args.lr * (self.decay_rate ** (epoch // args.decay_epoch)), args.min_lr)
            for param_group in optimizer.param_groups:
                if lr != param_group['lr']:
                    print 'Set lr to %f, previously %f'%(lr, param_group['lr'])
                    param_group['lr'] = lr
        elif self.decay_method == 'list':
            if epoch in args.lr_decay_epoch:
                for param_group in optimizer.param_groups:
                    new_lr = max(param_group['lr'] * self.decay_rate, args.min_lr)
                    print 'Set %s lr to %f, previously %f'%(param_group['group_name'], 
                                                            new_lr,
                                                            param_group['lr'])
                    param_group['lr'] = new_lr
            else:
                for param_group in optimizer.param_groups:
                    print 'Not in epoch list, remain lr as %f'%param_group['lr']
        elif self.decay_method == 'wait':
            if force_decay and self.cool_down_count >= self.cool_down and self.current_lr > args.min_lr:
                self.cool_down_count = 0
                for param_group in optimizer.param_groups:
                    print 'Set %s lr to %f, previously %f'%(param_group['group_name'],
                                                            param_group['lr'] * self.decay_rate,
                                                            param_group['lr'])
                    param_group['lr'] = max(param_group['lr'] * self.decay_rate, args.min_lr)
                    if param_group['lr_mult'] == 1:
                        self.current_lr = min(param_group['lr'], self.current_lr)
            else:
                if force_decay:
                    self.cool_down_count += 1
                    print 'Force decay cooled down %d time(s)'%self.cool_down_count
                else:
                    self.cool_down_count = 0
                if self.cool_down_count > self.max_cool_down:
                    self.early_stop = True
                for param_group in optimizer.param_groups:
                    pass#print 'No force-decay, remain %s lr to %f'%(param_group['group_name'],param_group['lr'])
        else:
                pass
        return self.early_stop

def accuracy(output, target):
    """Computes the precision@k for the specified values of k"""
    tmp = output
    tmp[tmp > 0.5] = 1
    tmp[tmp <= 0.5] = 0
    prec = metrics.precision_score(target, tmp)
    rec = metrics.recall_score(target, tmp)
    acc = metrics.accuracy_score(target, tmp)
    f1 = metrics.f1_score(target, tmp)

    return prec, rec, acc, f1

if __name__ == '__main__':
    main()

