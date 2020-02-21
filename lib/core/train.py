import torch.nn as nn

import time

from lib.utils import AvgrageMeter, adjust_learning_rate, accuracy


def train(model, train_loader, optimizer, criterion, epoch, cfg, logger, writer):
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()
    losses = AvgrageMeter()

    model.train()

    step = 0
    sta_time = time.time()
    for i, (data, target) in enumerate(train_loader):
        adjust_learning_rate(optimizer, epoch, cfg.TRAIN.OPTIM.LR)
        if cfg.CUDA:
            data, target = data.cuda(), target.cuda()

        N = data.size(0)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        prec1, prec5 = accuracy(output, target, topk=(1, 5))

        losses.update(loss.item(), N)
        top1.update(prec1.item(), N)
        top5.update(prec5.item(), N)

        step += 1
        if step % cfg.TRAIN.DISP == 0:
            logger.info("Train: Epoch {}/{}  Time: {:.3f} Loss {losses.avg:.3f} LR {:.3f} "
                        "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                        epoch, cfg.TRAIN.EPOCH, (time.time()-sta_time)/cfg.TRAIN.DISP,
                        optimizer.param_groups[0]['lr'],
                        losses=losses, top1=top1, top5=top5))

            writer.add_scalar('Loss/train', losses.avg, epoch * len(train_loader) + i)
            writer.add_scalar('Accuracy/train', top1.avg, epoch * len(train_loader) + i)