import torch

import time

from lib.utils import AvgrageMeter, accuracy


def validate(model, test_loader, criterion, epoch, cfg, logger, attention_logger, writer):
    print('evaluation ...')
    model.eval()

    top1 = AvgrageMeter()
    top5 = AvgrageMeter()
    losses = AvgrageMeter()


    step = 0
    attention_logger.info("Epoch {}".format(epoch))
    with torch.no_grad():
        sta_time = time.time()
        for i, (data, target) in enumerate(test_loader):
            N = data.size(0)
            if cfg.CUDA:
                data, target = data.cuda(), target.cuda()
            output = model(data)

            loss = criterion(output, target)
            prec1, prec5 = accuracy(output, target, topk=(1, 5))

            losses.update(loss.item(), N)
            top1.update(prec1.item(), N)
            top5.update(prec5.item(), N)

            step += 1
            if step % cfg.TRAIN.DISP == 0 and epoch > 80:
                cfg.DISP_ATTENTION = True
                logger.info("Test: Epoch {}/{}  Time: {:.3f} Loss {losses.avg:.3f} "
                            "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                    epoch, cfg.TRAIN.EPOCH, (time.time() - sta_time) / cfg.TRAIN.DISP,
                    losses=losses, top1=top1, top5=top5))

                writer.add_scalar('Loss/vaild', losses.avg, epoch * len(test_loader) + i)
                writer.add_scalar('Accuracy/vaild', top1.avg, epoch * len(test_loader) + i)
            else:
                cfg.DISP_ATTENTION = False

    return prec1.item()
