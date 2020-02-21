import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import os
import time

from config import get_args
from models.saResnet import SAResNet50, SAResNet38, SAResNet26
from data.preprocess import load_data
from lib.utils import  AvgrageMeter, accuracy, save_checkpoint


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(model, train_loader, optimizer, criterion, epoch, args, logger, writer):
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()
    losses = AvgrageMeter()

    model.train()

    step = 0
    sta_time = time.time()
    for i, (data, target) in enumerate(train_loader):
        adjust_learning_rate(optimizer, epoch, args)
        if args.cuda:
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
        if step % args.print_interval == 0:
            logger.info("Train: Epoch {}/{}  Time: {:.3f} Loss {losses.avg:.3f} LR {:.3f} "
                        "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                        epoch, args.epochs, (time.time()-sta_time)/args.display,
                        optimizer.param_groups[0]['lr'],
                        losses=losses, top1=top1, top5=top5))

            writer.add_scalar('Loss/train', losses.avg, epoch * len(train_loader) + i)
            writer.add_scalar('Accuracy/train', top1.avg, epoch * len(train_loader) + i)


def eval(model, test_loader, criterion, epoch, args, writer):
    print('evaluation ...')
    model.eval()

    top1 = AvgrageMeter()
    top5 = AvgrageMeter()
    losses = AvgrageMeter()


    step = 0
    sta_time = time.time()
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            N = data.size(0)
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)

            loss = criterion(output, target)
            prec1, prec5 = accuracy(output, target, topk=(1, 5))

            losses.update(loss.item(), N)
            top1.update(prec1.item(), N)
            top5.update(prec5.item(), N)

            step += 1
            if step % args.print_interval == 0:
                logger.info("Train: Epoch {}/{}  Time: {:.3f} Loss {losses.avg:.3f} "
                            "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                    epoch, args.epochs, (time.time() - sta_time) / args.display,
                    losses=losses, top1=top1, top5=top5))

                writer.add_scalar('Loss/vaild', losses.avg, epoch * len(test_loader) + i)
                writer.add_scalar('Accuracy/vaild', top1.avg, epoch * len(test_loader) + i)

    return top1


def get_model_parameters(model):
    total_parameters = 0
    for layer in list(model.parameters()):
        layer_parameter = 1
        for l in list(layer.size()):
            layer_parameter *= l
        total_parameters += layer_parameter
    return total_parameters


def main(args, logger):
    writer = SummaryWriter()

    train_loader, test_loader = load_data(args)
    if args.dataset == 'CIFAR10':
        num_classes = 10
    elif args.dataset == 'CIFAR100':
        num_classes = 100
    elif args.dataset == 'IMAGENET':
        num_classes = 1000

    print('img_size: {}, num_classes: {}, stem: {}'.format(args.img_size, num_classes, args.stem))
    if args.model_name == 'SAResNet26':
        print('Model Name: {0}'.format(args.model_name))
        model = SAResNet26(num_classes=num_classes, stem=args.stem, num_sablock=args.num_sablock)
    elif args.model_name == 'SAResNet38':
        print('Model Name: {0}'.format(args.model_name))
        model = SAResNet38(num_classes=num_classes, stem=args.stem, num_sablock=args.num_sablock)
    elif args.model_name == 'SAResNet50':
        print('Model Name: {0}'.format(args.model_name))
        model = SAResNet50(num_classes=num_classes, stem=args.stem, num_sablock=args.num_sablock)

    if args.pretrained_model:
        filename = 'best_model_' + str(args.dataset) + '_' + str(args.model_name) + '_' + str(args.stem) + '_ckpt.tar'
        print('filename :: ', filename)
        file_path = os.path.join('./checkpoint', filename)
        checkpoint = torch.load(file_path)

        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        model_parameters = checkpoint['parameters']
        print('Load model, Parameters: {0}, Start_epoch: {1}, Acc: {2}'.format(model_parameters, start_epoch, best_acc))
        logger.info('Load model, Parameters: {0}, Start_epoch: {1}, Acc: {2}'.format(model_parameters, start_epoch, best_acc))
    else:
        start_epoch = 1
        best_acc = 0.0

    if args.cuda:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model = model.cuda()

    print("Number of model parameters: ", get_model_parameters(model))
    logger.info("Number of model parameters: {0}".format(get_model_parameters(model)))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    for epoch in range(start_epoch, args.epochs + 1):
        train(model, train_loader, optimizer, criterion, epoch, args, logger, writer)
        eval_acc = eval(model, test_loader, criterion, epoch, args, writer)

        is_best = eval_acc > best_acc
        best_acc = max(eval_acc, best_acc)

        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        filename = 'model_' + str(args.dataset) + '_' + str(args.model_name) + '_' + str(args.stem) + '_ckpt.tar'
        print('filename :: ', filename)

        parameters = get_model_parameters(model)

        if torch.cuda.device_count() > 1:
            save_checkpoint({
                'epoch': epoch,
                'arch': args.model_name,
                'state_dict': model.module.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'parameters': parameters,
            }, is_best, filename)
        else:
            save_checkpoint({
                'epoch': epoch,
                'arch': args.model_name,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'parameters': parameters,
            }, is_best, filename)

if __name__ == '__main__':
    args, logger = get_args()
    main(args, logger)
