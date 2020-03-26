import torch.nn as nn

import time

from lib.utils import AvgrageMeter, accuracy
from lib.config import cfg


def train(model, train_loader, optimizer, criterion, scheduler, epoch, logger, writer):
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()
    losses = AvgrageMeter()

    model.train()

    step = 0
    for i, (data, target) in enumerate(train_loader):
        sta_time = time.time()
        if cfg.cuda:
            data, target = data.cuda(), target.cuda()

        N = data.size(0)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if scheduler is not None:
            scheduler.step(epoch * len(train_loader) + i)

        prec1, prec5 = accuracy(output, target, topk=(1, 5))

        losses.update(loss.item(), N)
        top1.update(prec1.item(), N)
        top5.update(prec5.item(), N)

        step += 1
        if step % cfg.train.disp == 0 and cfg.ddp.local_rank == 0:
            if cfg.finetune.is_finetune:
                logger.info("Finetune: Epoch {}/{}  Time: {:.3f} Loss {losses.avg:.3f} LR {:.3f} "
                            "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                            epoch, cfg.finetune.epoch, (time.time()-sta_time)/cfg.train.disp,
                            optimizer.param_groups[0]['lr'],
                            losses=losses, top1=top1, top5=top5))
            else:
                logger.info("Train: Epoch {}/{}  Time: {:.3f} Loss {losses.avg:.3f} LR {:.3f} "
                            "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                            epoch, cfg.train.epoch, (time.time()-sta_time)/cfg.train.disp,
                            optimizer.param_groups[0]['lr'],
                            losses=losses, top1=top1, top5=top5))

            writer.add_scalar('Loss/train', losses.avg, epoch * len(train_loader) + i)
            writer.add_scalar('Accuracy/train', top1.avg, epoch * len(train_loader) + i)
            writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch * len(train_loader) + i)
