import argparse
import json
import os
import pickle
import random
import sys
import time
import warnings

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from utils.data import EcgDataset
from utils import ecg_transforms
from utils.misc import SimpleLogger, EarlyStopCounter, AverageMeter, ProgressMeter, Summary, metrics, save_checkpoint, plot_history
from models import xresnet1d


parser = argparse.ArgumentParser(description='Supervised Model')

# model
parser.add_argument('--arch', default='xresnet1d50', type=str, metavar='S', help='model architecture')

# data
parser.add_argument('--hdf5-path', default='data/ptb_xl/processed/dataset.h5', type=str, metavar='PATH', help='path to HDF5 dataset file')
parser.add_argument('--label-type', default='subclass', type=str, metavar='S', help='label type')
parser.add_argument('--num-train-folds', type=int, default=8, metavar='N', help='number of train data folds (default: 8)')

# training setting
parser.add_argument('--epochs', type=int, default=1000, metavar='N', help='number of epochs to train (default: 100)')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', type=int, default=256, metavar='N', help='input batch size (default: 256)')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 0.0003)')
parser.add_argument('--wd', default=0.01, type=float, metavar='W', help='weight decay')
parser.add_argument('--early-stop', type=int, default=20, metavar='N', help='early stop patience (default: 20)')

# utils
parser.add_argument('--num-workers', type=int, default=8, metavar='N', help='number of workers (default: 4)')
parser.add_argument('--seed', default=0, type=int, help='seed for initializing training. ')
parser.add_argument('--print-freq', default=100, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--save', default='', type=str, metavar='S', help='save path')


def main():
    args = parser.parse_args()
    if args.save == '':
        args.save = f'./outputs/seed_{args.seed}/supervised/{args.arch}/{args.num_train_folds}_folds/{args.label_type}'

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if torch.cuda.is_available():
        args.device = "cuda"
    elif torch.backends.mps.is_available():
        args.device = "mps"
    else:
        args.device = "cpu"
        print('using CPU, this will be slow')
    device = torch.device(args.device)

    os.makedirs(args.save, exist_ok=True)
    flog = open(os.path.join(args.save, 'train_log.txt'), 'w')
    sys.stdout = SimpleLogger(sys.stdout, flog)

    print('===> Configuration')
    print(args)

    with open(os.path.join(args.save, "config.json"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    
    ###########
    # Dataset #
    ###########

    train_transform = transforms.Compose([
        ecg_transforms.AddGaussianNoise(mean=0.0, std=0.02),            # Add mild Gaussian noise
        ecg_transforms.AmplitudeScale1D(scale_range=(0.8, 1.2)),        # Apply amplitude scaling
        ecg_transforms.RandomRescale1D(scale_range=(0.8, 1.2)),         # Apply temporal rescaling
        ecg_transforms.RandomCrop1D(target_size=256),                   # Finally, crop to the fixed target size
        ecg_transforms.RandomZeroOut1D(zero_ratio_range=(0.0, 0.2)),   # Optionally, zero out segments
        ecg_transforms.Normalize1D(mean=ecg_transforms.ECG_CHANNEL_MEAN, std=ecg_transforms.ECG_CHANNEL_STD, transpose_back=False),
    ])
    
    test_transform = transforms.Compose([
        ecg_transforms.CenterCrop1D(target_size=256),
        ecg_transforms.Normalize1D(mean=ecg_transforms.ECG_CHANNEL_MEAN, std=ecg_transforms.ECG_CHANNEL_STD, transpose_back=False),
    ])

    # data prepare
    train_folds = [1, 2, 3, 4, 5, 6, 7, 8]
    if args.num_train_folds < 8:
        random.shuffle(train_folds)
        train_folds = train_folds[:args.num_train_folds]
    
    val_folds = [9,]
    test_folds = [10,]

    train_data = EcgDataset(args.hdf5_path, labeled_only=True, label_type=args.label_type, include_folds=train_folds, transform=train_transform)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    
    val_data = EcgDataset(args.hdf5_path, labeled_only=True, label_type=args.label_type, include_folds=val_folds, transform=test_transform)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    test_data = EcgDataset(args.hdf5_path, labeled_only=True, label_type=args.label_type, include_folds=test_folds, transform=test_transform)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    ## Computing class weights
    # Collect train labels
    train_labels = []
    for data, target in train_loader:
        train_labels.extend(target.numpy().tolist())
    train_labels = np.array(train_labels)
    # Calculate average likelihood per class
    class_likelihoods = np.mean(train_labels, axis=0)
    # Compute class weights as inverses (add epsilon to avoid divide-by-zero)
    epsilon = 1e-6
    class_weights = 1 / (class_likelihoods + epsilon)
    # Normalize weights to have an average of 1 (optional)
    class_weights = class_weights / np.mean(class_weights)
    print()
    print('Class Likelihoods')
    print(class_likelihoods)
    print('Class Weights')
    print(class_weights)
    print()
    #########
    # Model #
    #########

    model_arch = getattr(xresnet1d, args.arch)
    model = model_arch(num_classes=train_loader.dataset.num_classes)
    model[0][0] = torch.nn.Conv1d(12, 32, kernel_size=5, stride=2, padding=2)
    print(model)
    model = model.to(device)

    # define loss function (criterion), optimizer, and learning rate scheduler
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.from_numpy(class_weights)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10, min_lr=1e-7, verbose=True)
    early_stop = EarlyStopCounter(mode='max', patience=20, cooldown=20)

    best_epoch = 0
    best_aupr = 0.0
    best_lrap = 0.0
    best_auroc = 0.0
    best_lrap_epoch = 0
    best_auroc_epoch = 0

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))    
            checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
            args.start_epoch = checkpoint['epoch']
            best_epoch = checkpoint['best_epoch']
            best_aupr = checkpoint['best_aupr']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))    

    history = {'train_loss': [], 'val_loss': [], 'train_aupr': [], 'val_aupr': []}
    for epoch in range(args.start_epoch, args.epochs):
        
        # train for one epoch
        train_loss, train_aupr, train_lrap, train_auroc = train(train_loader, model, criterion, optimizer, epoch, device, args)
        history['train_loss'].append(train_loss)
        history['train_aupr'].append(train_aupr)
        
        # evaluate on validation set
        val_loss, val_aupr, val_lrap, val_auroc = evaluate(val_loader, model, criterion, device, 'Validation', args)
        history['val_loss'].append(val_loss)
        history['val_aupr'].append(val_aupr)

        # learning rate adjust
        scheduler.step(val_aupr)
        early_stop.step(val_aupr)

        # track best AUPR and save checkpoint
        is_best = val_aupr > best_aupr
        if is_best:
            best_epoch = epoch
            best_aupr = val_aupr
            test_loss, test_aupr, test_lrap, test_auroc = evaluate(test_loader, model, criterion, device, 'Test', args)
            with open(os.path.join(args.save, 'test_results.csv'), 'w') as fout:
                fout.write("Epoch,Loss,mAP,LRAP,mAUC\n")
                fout.write(f"{epoch},{test_loss},{test_aupr},{test_lrap},{test_auroc}\n")
        
        if val_lrap > best_lrap:
            best_lrap = val_lrap
            best_lrap_epoch = epoch
        
        if val_auroc > best_auroc:
            best_auroc = val_auroc
            best_auroc_epoch = epoch
        
        print(f'\t\t Best Val mAP: {best_aupr:.4f} (epoch {best_epoch})\t Best Val LRAP: {best_lrap:.4f} (epoch {best_lrap_epoch})\t Best Val mAUC: {best_auroc:.4f} (epoch {best_auroc_epoch})')
        print(f'\t\t Current LR: {scheduler.get_last_lr()}')
        print()

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_epoch': best_epoch,
            'best_aupr': best_aupr,
            'optimizer' : optimizer.state_dict(),
            'scheduler' : scheduler.state_dict()
        }, is_best, os.path.join(args.save, 'checkpoint.pth'))

        if early_stop.is_stop:
            print("Early Stop Called!")
            break
    
    with open(os.path.join(args.save, 'history.pkl'), 'wb') as f:
        pickle.dump(history, f, pickle.HIGHEST_PROTOCOL)

    plot_history(history, args)


def train(train_loader, model, criterion, optimizer, epoch, device, args):
    batch_time = AverageMeter('Time', ':6.3f', Summary.SUM)
    data_time = AverageMeter('Data', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.AVERAGE)
    auroc_meter = AverageMeter('mAUC', ':1.4f', Summary.VALUE)
    aupr_meter = AverageMeter('mAP', ':1.4f', Summary.VALUE)
    lrap_meter = AverageMeter('LRAP', ':1.4f', Summary.VALUE)
    
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, losses, aupr_meter, lrap_meter, auroc_meter],
        prefix="Epoch {}: [Train]".format(epoch)
    )

    # switch to train mode
    model.train()

    end = time.time()
    all_labels = []
    all_scores = []
    for i, (data, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # move data to the same device as model
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        output = model(data)
        loss = criterion(output, target)
        
        # record loss and collect targets and prediction scores
        losses.update(loss.item(), data.size(0))
        all_labels.append((target.detach() > 0.5).long().cpu().numpy())
        all_scores.append(torch.sigmoid(output.detach()).cpu().numpy())
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    
    y_true = np.vstack(all_labels)
    y_score = np.vstack(all_scores)
    macro_auc_roc, macro_auc_pr, lrap = metrics(y_true, y_score)
    auroc_meter.update(macro_auc_roc)
    aupr_meter.update(macro_auc_pr)
    lrap_meter.update(lrap)
    progress.display_summary()

    return losses.avg, aupr_meter.val, lrap_meter.val, auroc_meter.val


def evaluate(val_loader, model, criterion, device, prefix, args):
    batch_time = AverageMeter('Time', ':6.3f', Summary.SUM)
    losses = AverageMeter('Loss', ':.4e', Summary.AVERAGE)
    auroc_meter = AverageMeter('mAUC', ':1.4f', Summary.VALUE)
    aupr_meter = AverageMeter('mAP', ':1.4f', Summary.VALUE)
    lrap_meter = AverageMeter('LRAP', ':1.4f', Summary.VALUE)

    progress = ProgressMeter(
        len(val_loader),
        [losses, aupr_meter, lrap_meter, auroc_meter],
        prefix=f'\t[{prefix}]')
    
    # switch to evaluate mode
    model.eval()
    all_labels = []
    all_scores = []
    with torch.no_grad():
        end = time.time()
        for i, (data, target) in enumerate(val_loader):
            
            # move data to the same device as model
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # compute output
            output = model(data)
            loss = criterion(output, target)

            # record loss and collect targets and prediction scores
            losses.update(loss.item(), data.size(0))
            all_labels.append((target.detach() > 0.5).long().cpu().numpy())
            all_scores.append(torch.sigmoid(output.detach()).cpu().numpy())
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # if i % args.print_freq == 0:
            #     progress.display(i + 1)

    y_true = np.vstack(all_labels)
    y_score = np.vstack(all_scores)
    macro_auc_roc, macro_auc_pr, lrap = metrics(y_true, y_score)
    auroc_meter.update(macro_auc_roc)
    aupr_meter.update(macro_auc_pr)
    lrap_meter.update(lrap)

    progress.display_summary()
    
    return losses.avg, aupr_meter.val, lrap_meter.val, auroc_meter.val


if __name__ == '__main__':
    main()