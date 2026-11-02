"""Required original helpers recovered from the verified model_source_snapshot.

Only unrelated functions and unused imports were removed. See AUDIT.md.
"""

import math
import random
import time
import datetime
from collections import defaultdict, deque

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist
from loguru import logger


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = '{median:.4f} ({global_avg:.4f})'
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(median=self.median, avg=self.avg, global_avg=self.global_avg, max=self.max, value=self.value)


class MetricLogger(object):

    def __init__(self, delimiter='\t'):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append('{}: {}'.format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [header, '[{0' + space_fmt + '}/{1}]', 'eta: {eta}', '{meters}', 'time: {time}', 'data: {data}']
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        if len(iterable) > 0:
            logger.info('{} Total time: {} ({:.4f} s / it)', header, total_time_str, total_time / len(iterable))
        else:
            logger.info('{} Total time: {} (0.0000 s / it)', header, total_time_str)


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Struct:

    def __init__(self, **entries):
        self.__dict__.update(entries)


def adjust_learning_rate(optimizer, epoch, cfg):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < cfg.warmup_epoch:
        lr = cfg.lr * epoch / cfg.warmup_epoch
    else:
        lr = cfg.min_lr + (cfg.lr - cfg.min_lr) * 0.5 * (1.0 + math.cos(math.pi * (epoch - cfg.warmup_epoch) / (cfg.train_epoch - cfg.warmup_epoch)))
    for param_group in optimizer.param_groups:
        if 'lr_scale' in param_group:
            param_group['lr'] = lr * param_group['lr_scale']
        else:
            param_group['lr'] = lr
    return lr


def save_model(conf, epoch, model, optimizer, save_path):
    to_save = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'config': conf}
    torch.save(to_save, save_path)


class FocalLoss(nn.Module):

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma
        focal_loss = focal_weight * ce_loss
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Top-k accuracy as a percentage (the original timm.utils.accuracy convention)."""
    maxk = min(max(topk), output.size(1))
    _, prediction = output.topk(maxk, 1, True, True)
    correct = prediction.t().eq(target.reshape(1, -1).expand_as(prediction.t()))
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100.0 / target.size(0) for k in topk]
