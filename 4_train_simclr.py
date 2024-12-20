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
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.neighbors import KNeighborsClassifier

from utils.data import EcgDataset, ContrastiveTransformations
from utils import ecg_transforms
from utils.misc import SimpleLogger, AverageMeter, ProgressMeter, Summary, metrics, save_checkpoint, plot_history
from models.simclr import ModelSimCLR


parser = argparse.ArgumentParser(description='SimCLR Self-supervised Pre-training')

# data
parser.add_argument('--hdf5-path', default='data/ptb_xl/processed/dataset.h5', type=str, metavar='PATH', help='path to HDF5 dataset file')
parser.add_argument('--num-train-folds', type=int, default=8, metavar='N', help='number of train data folds (default: 8)')

# moco specific configs:
parser.add_argument('--arch', default='xresnet1d50', type=str, metavar='S', help='model architecture')
parser.add_argument('--softmax-t', default=0.07, type=float, help='softmax temperature')
parser.add_argument('--use-mlp', action='store_true', help='use a MLP projection head')

# knn monitor
parser.add_argument('--knn-k', default=50, type=int, help='k in kNN monitor')

# training setting
parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train (default: 100)')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', type=int, default=2048, metavar='N', help='input batch size (default: 256)')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 0.0003)')
parser.add_argument('--wd', default=0.01, type=float, metavar='W', help='weight decay')

# utils
parser.add_argument('--num-workers', type=int, default=8, metavar='N', help='number of workers (default: 4)')
parser.add_argument('--seed', default=0, type=int, help='seed for initializing training. ')
parser.add_argument('--print-freq', default=100, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--save', default='', type=str, metavar='S', help='save path')


def main():
    args = parser.parse_args()
    if args.save == '':
        args.save = f'./outputs/seed_{args.seed}/simclr/{args.arch}/pretrain'

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

    args.use_mlp = True

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
    train_folds = train_folds[:args.num_train_folds]
    val_folds = [9,]
    test_folds = [10,]
    exclude_folds = val_folds + test_folds

    all_train_data = EcgDataset(args.hdf5_path, labeled_only=False, exclude_folds=exclude_folds, transform=ContrastiveTransformations(train_transform))
    all_train_loader = DataLoader(all_train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    # labeled train data for KNN evaluation
    labeled_train_data = EcgDataset(args.hdf5_path, labeled_only=True, label_type='subclass', include_folds=train_folds, transform=test_transform)
    labeled_train_loader = DataLoader(labeled_train_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)
    
    val_data = EcgDataset(args.hdf5_path, labeled_only=True, label_type='subclass', include_folds=val_folds, transform=test_transform)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    test_data = EcgDataset(args.hdf5_path, labeled_only=True, label_type='subclass', include_folds=test_folds, transform=test_transform)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    #########
    # Model #
    #########

    # create model
    model = ModelSimCLR(
        arch=args.arch,
        T=args.softmax_t,
        use_mlp=args.use_mlp,
    )
    print(model.encoder)
    model = model.to(device)
    
    # define loss function (criterion), optimizer, and learning rate scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150], gamma=0.1)

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

    history = {'train_loss': [], 'val_aupr': []}
    for epoch in range(args.start_epoch, args.epochs):
        
        # self-supervised train
        train_loss = train(all_train_loader, model, optimizer, epoch, device, args)
        history['train_loss'].append(train_loss)
        # scheduler.step(train_loss)
        
        # evaluate on validation set
        val_aupr, val_lrap, val_auroc = evaluate(labeled_train_loader, val_loader, model.encoder, device, 'Validation', args)
        history['val_aupr'].append(val_aupr)
    
        # learning rate adjust
        scheduler.step()

        # track best AUPR and save checkpoint
        is_best = val_aupr > best_aupr
        if is_best:
            best_epoch = epoch
            best_aupr = val_aupr
            test_aupr, test_lrap, test_auroc = evaluate(labeled_train_loader, test_loader, model.encoder, device, 'Test', args)
            with open(os.path.join(args.save, f'test_knn{args.knn_k}.csv'), 'w') as fout:
                fout.write("Epoch,mAP,LRAP,mAUC\n")
                fout.write(f"{epoch},{test_aupr},{test_lrap},{test_auroc}\n")
        
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
    
    with open(os.path.join(args.save, 'history.pkl'), 'wb') as f:
        pickle.dump(history, f, pickle.HIGHEST_PROTOCOL)

    plot_history(history, args)


def train(train_loader, model, optimizer, epoch, device, args):
    batch_time = AverageMeter('Time', ':6.3f', Summary.SUM)
    data_time = AverageMeter('Data', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.AVERAGE)
        
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, losses],
        prefix="Epoch {}: [Train]".format(epoch)
    )

    # switch to train mode
    model.train()

    end = time.time()
    for i, (data_pair, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # move data to the same device as model
        data_1, data_2 = data_pair[0], data_pair[1]
        data_1 = data_1.to(device, non_blocking=True)
        data_2 = data_2.to(device, non_blocking=True)
        
        # SimCLR model computes loss directly
        loss = model(data_1, data_2)
        
        # record loss
        losses.update(loss.item(), data_1.size(0))
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # if i % args.print_freq == 0:
        #     progress.display(i + 1)
    
    progress.display_summary()

    return losses.avg


def evaluate(train_loader, val_loader, encoder, device, prefix, args):
    batch_time = AverageMeter('Time', ':6.3f', Summary.SUM)
    auroc_meter = AverageMeter('mAUC', ':1.4f', Summary.VALUE)
    aupr_meter = AverageMeter('mAP', ':1.4f', Summary.VALUE)
    lrap_meter = AverageMeter('LRAP', ':1.4f', Summary.VALUE)

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, aupr_meter, lrap_meter, auroc_meter],
        prefix=f'\t[{prefix}]')
    
    # switch to evaluate mode
    encoder.eval()
    end = time.time()

    X_train, y_train = [], []
    X_val, y_val = [], []
    with torch.no_grad():
        for i, (data, target) in enumerate(train_loader):
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            feature = encoder(data)
            feature = F.normalize(feature, dim=1)
        
            X_train.append(feature.detach().cpu().numpy())
            y_train.append((target.detach() > 0.5).long().cpu().numpy())
        
        for i, (data, target) in enumerate(val_loader):
            
            # move data to the same device as model
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            feature = encoder(data)
            feature = F.normalize(feature, dim=1)
            
            X_val.append(feature.detach().cpu().numpy())
            y_val.append((target.detach() > 0.5).long().cpu().numpy())
    
    X_train = np.vstack(X_train)
    y_train = np.vstack(y_train)
    X_val = np.vstack(X_val)
    y_val = np.vstack(y_val)

    clf = KNeighborsClassifier(n_neighbors=args.knn_k, weights='distance', metric='cosine', n_jobs=-1)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_val)
    y_score = clf.predict_proba(X_val)

    try:
        y_score = np.vstack([y[:, 1] for y in y_score]).T
    except:
        for i, y in enumerate(y_score):
            print(f'{i}-th y_pred')
            print(y_pred[:, i])
            print(np.unique(y_pred[:, i], return_counts=True))
            print(f'{i}-th y_score')
            print(y)
            print(y.shape)
            print()
        raise ValueError
    
    macro_auc_roc, macro_auc_pr, lrap = metrics(y_val, y_score)
    auroc_meter.update(macro_auc_roc)
    aupr_meter.update(macro_auc_pr)
    lrap_meter.update(lrap)

    batch_time.update(time.time() - end)
    progress.display_summary()
    
    return aupr_meter.val, lrap_meter.val, auroc_meter.val


if __name__ == '__main__':
    main()