"""Training and evaluation extracted from S4_WSI_classification__MoE_fstkim.py."""

import torch
import torchmetrics
from loguru import logger
from utils.utils import MetricLogger, SmoothedValue, adjust_learning_rate, accuracy


def train_one_epoch(model, criterion, data_loader, optimizer0, device, epoch, conf, writer):
    """
    Trains the given network for one epoch according to given criterions (loss functions)
    """
    model.train()
    metric_logger = MetricLogger(delimiter='  ')
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 80
    logger.info(header)
    for data_it, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image_patches = data['input'].to(device, dtype=torch.float32)
        labels = data['label'].to(device)
        adjust_learning_rate(optimizer0, epoch + data_it / len(data_loader), conf)
        sub_preds, slide_preds, gate_weights = model(image_patches)
        labels_processed = labels.view(-1).long()
        if labels_processed.max() >= conf.n_class or labels_processed.min() < 0:
            logger.warning(f'Warning: Label values {labels_processed} out of range for {conf.n_class} classes')
            labels_processed = torch.clamp(labels_processed, 0, conf.n_class - 1)
        loss0 = criterion(sub_preds, labels_processed)
        loss1 = criterion(slide_preds, labels_processed)
        gate_entropy = -(gate_weights * torch.log(gate_weights + 1e-08)).sum(dim=-1).mean()
        diff_loss = gate_entropy
        loss = diff_loss + loss0 + loss1
        optimizer0.zero_grad()
        loss.backward()
        optimizer0.step()
        metric_logger.update(lr=optimizer0.param_groups[0]['lr'])
        metric_logger.update(sub_loss=loss0.item())
        metric_logger.update(diff_loss=diff_loss.item())
        metric_logger.update(slide_loss=loss1.item())
        writer.add_scalar('train_losses/sub_loss', loss0.item(), epoch * len(data_loader) + data_it)
        writer.add_scalar('train_losses/diff_loss', diff_loss.item(), epoch * len(data_loader) + data_it)
        writer.add_scalar('train_losses/slide_loss', loss1.item(), epoch * len(data_loader) + data_it)


@torch.no_grad()
def evaluate(net, criterion, data_loader, device, conf, header):
    net.eval()
    y_pred = []
    y_true = []
    metric_logger = MetricLogger(delimiter='  ')
    for data in metric_logger.log_every(data_loader, 100, header):
        image_patches = data['input'].to(device, dtype=torch.float32)
        labels = data['label'].to(device)
        sub_preds, slide_preds, gate_weights = net(image_patches)
        pred = torch.softmax(slide_preds, dim=-1)
        labels_processed = labels.view(-1).long()
        if labels_processed.max() >= conf.n_class or labels_processed.min() < 0:
            print(f'Warning: Label values {labels_processed} out of range for {conf.n_class} classes')
            labels_processed = torch.clamp(labels_processed, 0, conf.n_class - 1)
        loss = criterion(slide_preds, labels_processed)
        acc1 = accuracy(pred, labels, topk=(1,))[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=labels.shape[0])
        y_pred.append(pred)
        y_true.append(labels)
    y_pred = torch.cat(y_pred, dim=0)
    y_true = torch.cat(y_true, dim=0)
    AUROC_metric = torchmetrics.AUROC(num_classes=conf.n_class, task='multiclass').to(device)
    AUROC_metric(y_pred, y_true)
    auroc = AUROC_metric.compute().item()
    F1_metric = torchmetrics.F1Score(num_classes=conf.n_class, task='multiclass').to(device)
    F1_metric(y_pred, y_true)
    f1_score = F1_metric.compute().item()
    logger.info('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f} auroc {AUROC:.3f} f1_score {F1:.3f}'.format(top1=metric_logger.acc1, losses=metric_logger.loss, AUROC=auroc, F1=f1_score))
    return (auroc, metric_logger.acc1.global_avg, f1_score, metric_logger.loss.global_avg)
