import math
import os
import shutil
from enum import Enum
from typing import Literal

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
sns.set_theme(style='darkgrid')

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, label_ranking_average_precision_score
import torch


class SimpleLogger(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush() # If you want the output to be visible immediately
    def flush(self) :
        for f in self.files:
            f.flush()


# Early stopping class
class EarlyStopCounter:
    def __init__(
            self,
            mode: Literal["min", "max"] = "min",
            patience=10,
            threshold=1e-4,
            threshold_mode: Literal["rel", "abs"] = "rel",
            cooldown=0
            ):
        self.patience = patience
        self.cooldown = cooldown
        self.best = None
        self.cooldown_counter = 0
        self.num_bad_epochs = 0
        self.is_stop = False
        self.last_epoch = 0
        self._init_is_better(mode=mode, threshold=threshold, threshold_mode=threshold_mode)
        self._reset()

    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0
        self.is_stop = False
    
    def step(self, metric):  # type: ignore[override]
        current = float(metric)
        self.last_epoch += 1

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            self.is_stop = True

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0
    
    def is_better(self, a, best):
        if self.mode == "min" and self.threshold_mode == "rel":
            rel_epsilon = 1.0 - self.threshold
            return a < best * rel_epsilon
        elif self.mode == "min" and self.threshold_mode == "abs":
            return a < best - self.threshold
        elif self.mode == "max" and self.threshold_mode == "rel":
            rel_epsilon = self.threshold + 1.0
            return a > best * rel_epsilon
        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + self.threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {"min", "max"}:
            raise ValueError("mode " + mode + " is unknown!")
        if threshold_mode not in {"rel", "abs"}:
            raise ValueError("threshold mode " + threshold_mode + " is unknown!")
        
        if mode == "min":
            self.mode_worse = math.inf
        else:  # mode == 'max':
            self.mode_worse = -math.inf

        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3
    VALUE = 4

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            # fmtstr = '{name} {avg:.3f}'
            fmtstr = '{name} {avg' + self.fmt + '}'
        elif self.summary_type is Summary.SUM:
            # fmtstr = '{name} {sum:.3f}'
            fmtstr = '{name} {sum' + self.fmt + '}'
        elif self.summary_type is Summary.COUNT:
            # fmtstr = '{name} {count:.3f}'
            fmtstr = '{name} {count' + self.fmt + '}'
        elif self.summary_type is Summary.VALUE:
            # fmtstr = '{name} {val:.3f}'
            fmtstr = '{name} {val' + self.fmt + '}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        
    def display_summary(self):
        # entries = [" *"]
        entries = [self.prefix + "[Summary]"]
        entries += [meter.summary() for meter in self.meters]
        print(' \t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


# Calculate macro-averaged AUC-ROC, AUC-PR, and LRAP
def metrics(y_true, y_score):
    
    # Calculate macro-averaged AUC-ROC
    try:
        macro_auc_roc = roc_auc_score(y_true, y_score, average="macro")
    except ValueError:
        # macro_auc_roc = float("nan")  # Handle case with undefined AUC
        macro_auc_roc = 0.0

    # Calculate macro-averaged AUC-PR
    try:
        pr_auc = [average_precision_score(y_true[:, i], y_score[:, i]) for i in range(y_true.shape[1])]
        macro_auc_pr = np.mean(pr_auc)
    except ValueError:
        # macro_auc_pr = float("nan")  # Handle case with undefined AUC-PR
        macro_auc_pr = 0.0

    # Calculate Label Ranking Average Precision (LRAP)
    try:
        lrap = label_ranking_average_precision_score(y_true, y_score)
    except ValueError:
        # lrap = float("nan")  # Handle case with undefined LRAP
        lrap = 0.0

    return macro_auc_roc, macro_auc_pr, lrap


def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    torch.save(state, filename)
    if is_best:
        directory = os.path.dirname(filename)
        shutil.copyfile(filename, os.path.join(directory, 'best_checkpoint.pth'))


def plot_history(history, args):
    metric_names = {
        'loss': 'Loss',
        'aupr': 'mAP',
    }
    # Retrieve the default color palette (deep)
    palette = sns.color_palette("deep")  # "deep" is Seaborn's default palette
    
    history_keys = set([k.split('_')[-1] for k in history.keys()])
    for k in history_keys:
        train_history_key = f'train_{k}'
        val_history_key = f'val_{k}'
        
        if train_history_key in history:
            train_history = history[train_history_key]
            num_train_epochs = len(train_history)
        else:
            train_history = None
            num_train_epochs = 0

        if val_history_key in history:
            val_history = history[val_history_key]
            num_val_epochs = len(val_history)
        else:
            val_history = None
            num_val_epochs = 0
        
        fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8))
        if train_history is not None:
            ax.plot(range(1, num_train_epochs + 1), train_history, linewidth=3, color=palette[0], label='Train')
        if val_history is not None:
            ax.plot(range(1, num_val_epochs + 1), val_history, linewidth=3, color=palette[1], label='Validation')
        
        ax.set_xlabel('Epochs', fontsize=18)
        ax.set_ylabel(metric_names[k], fontsize=18)
        plt.setp(ax.spines.values(), linewidth=2)
        ax.tick_params(axis='both', which='major', width=2, labelsize=16)
        ax.legend(fontsize=18)
        fig.tight_layout()

        fig.savefig(os.path.join(args.save, f'{k}_curve.pdf'), format='pdf')
    