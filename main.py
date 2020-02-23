import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import os
import argparse
import shutil

from lib.data import *
from lib.models.saResnet import *
from lib.core.train import train
from lib.core.vaild import validate
from lib.config import cfg
from lib.utils import save_checkpoint, get_model_parameters, get_logger, CrossEntropyLabelSmooth


parser = argparse.ArgumentParser('parameters')
parser.add_argument('name', type=str)
parser.add_argument('--cfg', type=str, default='./experiments/yaml/baseline.yaml')


def main(cfg):
    logger = get_logger(os.path.join(cfg.SAVE_PATH, cfg.JOB_NAME, 'train.log'))
    # writer = SummaryWriter(os.path.join(cfg.SAVE_PATH, cfg.JOB_NAME, 'runs'))

    train_loader, test_loader, num_classes = eval(cfg.TRAIN.DATASET.NAME)(cfg)

    print('img_size: {}, num_classes: {}, stem: {}'.format(cfg.TRAIN.DATASET.IMAGE_SIZE,
                                                           num_classes,
                                                           cfg.TRAIN.MODEL.STEM))

    print('Model Name: {0}'.format(cfg.TRAIN.MODEL.NAME))
    model = eval(cfg.TRAIN.MODEL.NAME)(num_classes=num_classes,
                                       stem=cfg.TRAIN.MODEL.STEM,
                                       num_sablock=cfg.TRAIN.MODEL.NUM_SABLOCK)

    if cfg.TRAIN.MODEL.PRE_TRAINED:
        filename = 'best_model_' + str(cfg.TRAIN.DATASET.NAME) + '_' + \
                   str(cfg.TRAIN.MODEL.NAME) + '_' + str(cfg.TRAIN.MODEL.STEM) + '_ckpt.tar'
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

    if cfg.TRAIN.CRIT.SMOOTH > 0:
        criterion = CrossEntropyLabelSmooth(num_classes=num_classes, epsilon=cfg.TRAIN.CRIT.SMOOTH)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=cfg.TRAIN.OPTIM.LR,
                          momentum=cfg.TRAIN.OPTIM.MOMENTUM, weight_decay=cfg.TRAIN.OPTIM.WD)

    if cfg.CUDA:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model = model.cuda()
        criterion = criterion.cuda()

    print("Number of model parameters: ", get_model_parameters(model))
    logger.info("Number of model parameters: {0}".format(get_model_parameters(model)))

    for epoch in range(start_epoch, cfg.TRAIN.EPOCH + 1):
        train(model, train_loader, optimizer, criterion, epoch, cfg, logger, writer)
        eval_acc = validate(model, test_loader, criterion, epoch, cfg, logger, writer)

        is_best = eval_acc > best_acc
        best_acc = max(eval_acc, best_acc)

        save_path = os.path.join(cfg.SAVE_PATH, cfg.JOB_NAME, 'checkpoints')
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        filename = 'model_' + str(cfg.TRAIN.DATASET.NAME) + '_' +\
                   str(cfg.TRAIN.MODEL.NAME) + '_' + str(cfg.TRAIN.MODEL.STEM) + '_ckpt.tar'
        print('filename :: ', filename)

        parameters = get_model_parameters(model)

        if torch.cuda.device_count() > 1:
            save_checkpoint({
                'epoch': epoch,
                'arch': cfg.TRAIN.MODEL.NAME,
                'state_dict': model.module.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'parameters': parameters,
            }, is_best, save_path, filename)
        else:
            save_checkpoint({
                'epoch': epoch,
                'arch': cfg.TRAIN.MODEL.NAME,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'parameters': parameters,
            }, is_best, save_path, filename)


if __name__ == '__main__':
    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    cfg.SAVE_PATH = os.path.join(cfg.SAVE_PATH, 'train')
    if not os.path.isdir(cfg.SAVE_PATH):
        os.mkdir(cfg.SAVE_PATH)

    cfg.JOB_NAME = args.name
    save_path = os.path.join(cfg.SAVE_PATH, cfg.JOB_NAME)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    else:
        key = input('Delete Existing Directory [y/n]: ')
        if key == 'n':
            if cfg.AUTO_RESUME:
                pass
            else:
                raise ValueError("Save directory already exists")
        elif key == 'y':
            shutil.rmtree(save_path)
            os.mkdir(save_path)
        else:
            raise ValueError("Input Not Supported!")

    main(cfg)
