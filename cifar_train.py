import argparse
import os
import random
import time
import warnings
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models
from tensorboardX import SummaryWriter
from sklearn.metrics import confusion_matrix
from utils import *
from imbalance_cifar import IMBALANCECIFAR10, IMBALANCECIFAR100
from losses import LDAMLoss, FocalLoss

from pyclustering.cluster.xmeans import xmeans
from pyclustering.utils import euclidean_distance


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Cifar Training')
parser.add_argument('--dataset', default='cifar10', help='dataset setting')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet32',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet32)')
parser.add_argument('--loss_type', default="CE", type=str, help='loss type')
parser.add_argument('--imb_type', default="exp", type=str, help='imbalance type')
parser.add_argument('--imb_factor', default=0.01, type=float, help='imbalance factor')
parser.add_argument('--train_rule', default='None', type=str, help='data sampling strategy for train loader')
parser.add_argument('--rand_number', default=0, type=int, help='fix random number for data sampling')
parser.add_argument('--exp_str', default='0', type=str, help='number to indicate which experiment it is')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=2e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--root_log',type=str, default='log')
parser.add_argument('--root_model', type=str, default='checkpoint')
parser.add_argument('--cutoff_epoch', type=int, default = 160)
best_acc1 = 0


def main():
    args = parser.parse_args()
    args.store_name = '_'.join([args.dataset, args.arch, args.loss_type, args.train_rule, args.imb_type, str(args.imb_factor), args.exp_str])
    prepare_folders(args)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    ngpus_per_node = torch.cuda.device_count()
    main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    print("=> creating model '{}'".format(args.arch))
    num_classes = 100 if args.dataset == 'cifar100' else 10
    use_norm = True if args.loss_type == 'LDAM' else False
    model = models.__dict__[args.arch](num_classes=num_classes, use_norm=use_norm)
    non_gpu_model = models.__dict__[args.arch](num_classes=num_classes, use_norm=use_norm)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cuda:0')
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if args.dataset == 'cifar10':
        train_dataset = IMBALANCECIFAR10(root='./data', imb_type=args.imb_type, imb_factor=args.imb_factor, rand_number=args.rand_number, train=True, download=True, transform=transform_train)
        val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_val)
    elif args.dataset == 'cifar100':
        train_dataset = IMBALANCECIFAR100(root='./data', imb_type=args.imb_type, imb_factor=args.imb_factor, rand_number=args.rand_number, train=True, download=True, transform=transform_train)
        val_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_val)
    else:
        warnings.warn('Dataset is not listed')
        return
    cls_num_list = train_dataset.get_cls_num_list()
    print('cls num list:')
    print(cls_num_list)
    args.cls_num_list = cls_num_list


    train_sampler = None
        
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=100, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # init log for training
    log_training = open(os.path.join(args.root_log, args.store_name, 'log_train.csv'), 'w')
    log_testing = open(os.path.join(args.root_log, args.store_name, 'log_test.csv'), 'w')
    with open(os.path.join(args.root_log, args.store_name, 'args.txt'), 'w') as f:
        f.write(str(args))
    tf_writer = SummaryWriter(log_dir=os.path.join(args.root_log, args.store_name))
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        if args.train_rule == 'None':
            train_sampler = None  
            per_cls_weights = None
        elif args.train_rule == 'Resample':
            train_sampler = ImbalancedDatasetSampler(train_dataset)
            per_cls_weights = None
        elif args.train_rule == 'Reweight':
            train_sampler = None
            beta = 0.9999
            effective_num = 1.0 - np.power(beta, cls_num_list)
            per_cls_weights = (1.0 - beta) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(args.gpu)
        elif args.train_rule == 'DRW':
            train_sampler = None
            idx = min(epoch // 160, 1)
            betas = [0, 0.9999]
            effective_num = 1.0 - np.power(betas[idx], cls_num_list)
            per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(args.gpu)
        elif args.train_rule == 'SWITCHING_DRW_AUTO_CLUSTER':
            train_sampler = None
            cutoff_epoch = args.cutoff_epoch
            idx = min(epoch // cutoff_epoch, 1)
            betas = [0, 0.9999]
            if epoch >= cutoff_epoch + 10 and (epoch - cutoff_epoch) % 20 == 0:
                max_real_ratio_number_of_labels = 20
                # todo: transform data batch by batch, then concatentate...
                temp_batch_size = int(train_dataset.data.shape[0])
                temp_train_loader = torch.utils.data.DataLoader(
                    train_dataset, batch_size=int(temp_batch_size), shuffle=(train_sampler is None),
                    num_workers=args.workers, pin_memory=True, sampler=train_sampler)

                transformed_data = None
                transformed_labels = None
                for i, (xs, labels) in enumerate(train_loader):
                    transformed_batch_data = model.forward_partial(xs.cuda(), num_layers=6)
                    transformed_batch_data = transformed_batch_data.cpu().detach()
                    transformed_batch_data = transformed_batch_data.numpy()
                    transformed_batch_data = np.reshape(transformed_batch_data, (transformed_batch_data.shape[0], -1))
                    labels = np.array(labels)[:, np.newaxis]
                    if transformed_data is None:
                        transformed_data = transformed_batch_data
                        transformed_labels = labels
                    else:
                        transformed_data = np.vstack((transformed_data, transformed_batch_data))
                        # print(labels.shape)
                        # print(transformed_labels.shape)
                        transformed_labels = np.vstack((transformed_labels, labels))

                xmean_model = xmeans(data=transformed_data, kmax=num_classes * max_real_ratio_number_of_labels)
                xmean_model.process()
                # Extract clustering results: clusters and their centers
                clusters = xmean_model.get_clusters()
                centers = xmean_model.get_centers()
                new_labels = []
                xs = transformed_data
                centers = np.array(centers)
                print("number of clusters: ", len(centers))
                squared_norm_dist = np.sum((xs - centers[:, np.newaxis]) ** 2, axis=2)
                data_centers = np.argmin(squared_norm_dist, axis=0)
                data_centers = np.expand_dims(data_centers, axis=1)

                new_labels = []
                for i in range(len(transformed_labels)):
                    new_labels.append(data_centers[i][0] + transformed_labels[i][0] * len(centers))

                new_label_counts = {}
                for label in new_labels:
                    if label in new_label_counts.keys():
                        new_label_counts[label] += 1
                    else:
                        new_label_counts[label] = 1

                # print(new_label_counts)

                per_cls_weights = []
                for i in range(len(cls_num_list)):
                    temp = []
                    for j in range(len(centers)):
                        new_label = j + i * len(centers)
                        if new_label in new_label_counts:
                            temp.append(new_label_counts[new_label])
                    effective_num_temp = 1.0 - np.power(betas[idx], temp)
                    per_cls_weights_temp = (1.0 - betas[idx]) / np.array(effective_num_temp)
                    per_cls_weights.append(np.average(per_cls_weights_temp, weights=temp))
                per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
                per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(args.gpu)
            elif epoch < cutoff_epoch or (epoch - cutoff_epoch) % 20 == 10:
                effective_num = 1.0 - np.power(betas[idx], cls_num_list)
                per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
                per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
                per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(args.gpu)

        elif args.train_rule == 'DRW_AUTO_CLUSTER':
            train_sampler = None
            cutoff_epoch = args.cutoff_epoch
            idx = min(epoch // cutoff_epoch, 1)
            betas = [0, 0.9999]
            if epoch >= cutoff_epoch and (epoch - cutoff_epoch) % 10 == 0:
                max_real_ratio_number_of_labels = 20
                # todo: transform data batch by batch, then concatentate...
                temp_batch_size = int(train_dataset.data.shape[0])
                temp_train_loader = torch.utils.data.DataLoader(
                    train_dataset, batch_size=int(temp_batch_size), shuffle=(train_sampler is None),
                    num_workers=args.workers, pin_memory=True, sampler=train_sampler)

                transformed_data = None
                transformed_labels = None
                for i, (xs, labels) in enumerate(train_loader):
                    transformed_batch_data = model.forward_partial(xs.cuda(), num_layers=6)
                    transformed_batch_data= transformed_batch_data.cpu().detach()
                    transformed_batch_data = transformed_batch_data.numpy()
                    transformed_batch_data= np.reshape(transformed_batch_data, (transformed_batch_data.shape[0], -1))
                    labels = np.array(labels)[:, np.newaxis]
                    if transformed_data is None:
                        transformed_data = transformed_batch_data
                        transformed_labels = labels
                    else:
                        transformed_data = np.vstack( (transformed_data, transformed_batch_data) )
                        # print(labels.shape)
                        # print(transformed_labels.shape)
                        transformed_labels = np.vstack((transformed_labels, labels))

                initial_centers = [np.zeros((transformed_data.shape[1],)) for i in range(num_classes)]
                center_counts = [0 for i in range(num_classes)]
                for i in range(transformed_data.shape[0]):
                    temp_idx = transformed_labels[i][0]
                    initial_centers[temp_idx] = initial_centers[temp_idx] + transformed_data[i,:]
                    center_counts[temp_idx] = center_counts[temp_idx] + 1

                for i in range(num_classes):
                    initial_centers[i] = initial_centers[i] / center_counts[i]

                xmean_model = xmeans(data=transformed_data, initial_centers=initial_centers, \
                                     kmax=num_classes * max_real_ratio_number_of_labels)
                xmean_model.process()
                # Extract clustering results: clusters and their centers
                clusters = xmean_model.get_clusters()
                centers = xmean_model.get_centers()
                new_labels = []
                xs = transformed_data
                centers = np.array(centers)
                print("number of clusters: ", len(centers))
                squared_norm_dist = np.sum((xs - centers[:, np.newaxis]) ** 2, axis = 2 )
                data_centers = np.argmin(squared_norm_dist, axis = 0)
                data_centers = np.expand_dims(data_centers, axis = 1)

                new_labels = []
                for i in range(len(transformed_labels)):
                    new_labels.append(data_centers[i][0] + transformed_labels[i][0] * len(centers))

                new_label_counts = {}
                for label in new_labels:
                    if label in new_label_counts.keys():
                        new_label_counts[label] += 1
                    else:
                        new_label_counts[label] = 1

                # print(new_label_counts)

                per_cls_weights = []
                for i in range(len(cls_num_list)):
                    temp = []
                    for j in range(len(centers)):
                        new_label = j + i * len(centers)
                        if new_label in new_label_counts:
                            temp.append(new_label_counts[new_label])
                    effective_num_temp = 1.0 - np.power(betas[idx], temp)
                    per_cls_weights_temp = (1.0 - betas[idx]) / np.array(effective_num_temp)
                    per_cls_weights.append(np.average(per_cls_weights_temp, weights=temp))
                per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
                per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(args.gpu)
            elif epoch < cutoff_epoch:
                effective_num = 1.0 - np.power(betas[idx], cls_num_list)
                per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
                per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
                per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(args.gpu)

        else:
            warnings.warn('Sample rule is not listed')

        if args.loss_type == 'CE':
            criterion = nn.CrossEntropyLoss(weight=per_cls_weights).cuda(args.gpu)
        elif args.loss_type == 'LDAM':
            criterion = LDAMLoss(cls_num_list=cls_num_list, max_m=0.5, s=30, weight=per_cls_weights).cuda(args.gpu)
            #temp = [cls_num_list[i] * per_cls_weights[i].item() for i in range(len(cls_num_list))]
            #criterion = LDAMLoss(cls_num_list=temp, max_m=0.5, s=30, weight=per_cls_weights).cuda(args.gpu)
        elif args.loss_type == 'Focal':
            criterion = FocalLoss(weight=per_cls_weights, gamma=1).cuda(args.gpu)
        else:
            warnings.warn('Loss type is not listed')
            return

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, log_training, tf_writer)
        
        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, epoch, args, log_testing, tf_writer)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        tf_writer.add_scalar('acc/test_top1_best', best_acc1, epoch)
        output_best = 'Best Prec@1: %.3f\n' % (best_acc1)
        print(output_best)
        log_testing.write(output_best + '\n')
        log_testing.flush()

        save_checkpoint(args, {
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)


def train(train_loader, model, criterion, optimizer, epoch, args, log, tf_writer):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr'] * 0.1))  # TODO
            print(output)
            log.write(output + '\n')
            log.flush()

    # tf_writer.add_scalar('loss/train', losses.avg, epoch)
    # tf_writer.add_scalar('acc/train_top1', top1.avg, epoch)
    # tf_writer.add_scalar('acc/train_top5', top5.avg, epoch)
    # tf_writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch)

def validate(val_loader, model, criterion, epoch, args, log=None, tf_writer=None, flag='val'):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    # switch to evaluate mode
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            _, pred = torch.max(output, 1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

            if i % args.print_freq == 0:
                output = ('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))
                print(output)
        cf = confusion_matrix(all_targets, all_preds).astype(float)
        cls_cnt = cf.sum(axis=1)
        cls_hit = np.diag(cf)
        cls_acc = cls_hit / cls_cnt
        output = ('{flag} Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
                .format(flag=flag, top1=top1, top5=top5, loss=losses))
        out_cls_acc = '%s Class Accuracy: %s'%(flag,(np.array2string(cls_acc, separator=',', formatter={'float_kind':lambda x: "%.3f" % x})))
        print(output)
        print(out_cls_acc)
        if log is not None:
            log.write(output + '\n')
            log.write(out_cls_acc + '\n')
            log.flush()

        # tf_writer.add_scalar('loss/test_'+ flag, losses.avg, epoch)
        # tf_writer.add_scalar('acc/test_' + flag + '_top1', top1.avg, epoch)
        # tf_writer.add_scalar('acc/test_' + flag + '_top5', top5.avg, epoch)
        # tf_writer.add_scalars('acc/test_' + flag + '_cls_acc', {str(i):x for i, x in enumerate(cls_acc)}, epoch)

    return top1.avg


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    epoch = epoch + 1
    if epoch <= 5:
        lr = args.lr * epoch / 5
    #elif epoch > 180:
    #    lr = args.lr * 0.0001
    elif epoch > 160:
        lr = args.lr * 0.01
    else:
        lr = args.lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    main()