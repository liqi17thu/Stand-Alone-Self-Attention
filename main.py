import os

import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from lib.config import cfg
from lib.core.train import train
from lib.core.vaild import validate
from lib.models.saMobilenet import *
from lib.models.saResnet import *
from lib.utils import CrossEntropyLabelSmooth, get_scheduler, get_net_info
from lib.utils import save_checkpoint, get_model_parameters, get_logger, get_attention_logger


def main():
    if cfg.test:
        logger = get_logger(os.path.join(cfg.save_path, 'test.log'))
    else:
        logger = get_logger(os.path.join(cfg.save_path, 'train.log'))
    attention_logger = get_attention_logger(os.path.join(cfg.save_path, 'attention.log'))
    writer = SummaryWriter(cfg.log_dir)

    train_loader, test_loader, num_classes = eval(cfg.dataset.name)()

    print('img_size: {}, num_classes: {}, stem: {}'.format(cfg.dataset.image_size,
                                                           num_classes,
                                                           cfg.model.stem))

    print('Model Name: {0}'.format(cfg.model.name))
    if cfg.model.name == "MobileNetV3":
        model = MobileNetV3(n_class=num_classes, input_size=cfg.dataset.image_size, mode='large',
                            logger=attention_logger)
    else:
        model = eval(cfg.model.name)(num_classes=num_classes,
                                     heads=cfg.model.heads,
                                     kernel_size=cfg.model.kernel,
                                     stem=cfg.model.stem,
                                     num_resblock=cfg.model.num_resblock,
                                     attention_logger=attention_logger)

    if cfg.model.pre_trained:
        filename = 'best_model_' + str(cfg.dataset.name) + '_' + \
                   str(cfg.model.name) + '_' + str(cfg.model.stem) + '_ckpt.tar'
        print('filename :: ', filename)
        file_path = os.path.join('./checkpoint', filename)
        checkpoint = torch.load(file_path)

        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        model_parameters = checkpoint['parameters']
        print('Load model, Parameters: {0}, Start_epoch: {1}, Acc: {2}'.format(model_parameters, start_epoch, best_acc))
        logger.info(
            'Load model, Parameters: {0}, Start_epoch: {1}, Acc: {2}'.format(model_parameters, start_epoch, best_acc))
    else:
        start_epoch = cfg.train.start_epoch
        best_acc = 0.0

    get_net_info(model, (3, cfg.dataset.image_size, cfg.dataset.image_size), logger=logger)

    if cfg.crit.smooth > 0:
        criterion = CrossEntropyLabelSmooth(num_classes=num_classes, epsilon=cfg.crit.smooth)
    else:
        criterion = nn.CrossEntropyLoss()

    if cfg.optim.method == 'SGD':
        optimizer = optim.SGD(model.parameters(), **cfg.optim.sgd_params)
    elif cfg.optim.method == 'Adam':
        optimizer = optim.Adam(model.parameters(), **cfg.optim.adam_params)
    else:
        raise ValueError(f'Unsupported optimization: {cfg.optim.method}')

    scheduler = get_scheduler(optimizer, len(train_loader), cfg)

    if cfg.cuda:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model = model.cuda()
        criterion = criterion.cuda()

    if cfg.test:
        _ = validate(model, test_loader, criterion, start_epoch, logger, attention_logger, writer)
        return

    for epoch in range(start_epoch, cfg.train.epoch + 1):
        train(model, train_loader, optimizer, criterion, scheduler, epoch, logger, writer)
        eval_acc = validate(model, test_loader, criterion, epoch, logger, attention_logger, writer)

        is_best = eval_acc > best_acc
        best_acc = max(eval_acc, best_acc)

        filename = 'model_' + str(cfg.dataset.name) + '_' + \
                   str(cfg.model.name) + '_' + str(cfg.model.stem) + '_ckpt.tar'
        print('filename :: ', filename)

        parameters = get_model_parameters(model)

        if torch.cuda.device_count() > 1:
            save_checkpoint({
                'epoch': epoch,
                'arch': cfg.model.name,
                'state_dict': model.module.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'parameters': parameters,
            }, is_best, cfg.ckp_dir, filename)
        else:
            save_checkpoint({
                'epoch': epoch,
                'arch': cfg.model.name,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'parameters': parameters,
            }, is_best, cfg.ckp_dir, filename)


if __name__ == '__main__':
    main()
