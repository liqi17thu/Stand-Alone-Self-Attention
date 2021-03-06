import os
import numpy as np

import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist

from lib.config import cfg
from lib.core.train import train
from lib.core.vaild import validate
from lib.data import *
from lib.models.saMobilenet import *
from lib.models.saResnet import *
from lib.utils import CrossEntropyLabelSmooth, get_scheduler, get_net_info
from lib.utils import save_checkpoint, get_model_parameters, get_logger, get_attention_logger

if cfg.ddp.distributed:
    from torch.nn.parallel import DistributedDataParallel as DDP
    # try:
    #     import apex
    #     from apex.parallel import DistributedDataParallel as DDP
    #     from apex import amp, optimizers
    #     from apex.fp16_utils import *
    # except ImportError:
    #     raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")


def main():

    # DistributedDataParallel
    if cfg.ddp.distributed:
        # np.random.seed(cfg.ddp.seed)
        torch.manual_seed(cfg.ddp.seed)
        torch.cuda.manual_seed_all(cfg.ddp.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        # torch.cuda.set_device(cfg.ddp.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method=cfg.ddp.dist_url,
                                             world_size=cfg.ddp.gpus, rank=cfg.ddp.local_rank,
                                             group_name='mtorch')
        gpu_id = dist.get_rank() % torch.cuda.device_count()
        torch.cuda.set_device(gpu_id)

    if cfg.ddp.local_rank == 0:
        if cfg.test:
            logger = get_logger(os.path.join(cfg.save_path, 'test.log'))
        elif cfg.finetune.is_finetune:
            logger = get_logger(os.path.join(cfg.save_path, 'finetune.log'))
        else:
            logger = get_logger(os.path.join(cfg.save_path, 'train.log'))
        attention_logger = get_attention_logger(os.path.join(cfg.save_path, 'attention.log'))
        writer = SummaryWriter(cfg.log_dir)
    else:
        logger = None
        attention_logger = None
        writer = None

    loaders, samplers, num_classes = eval(cfg.dataset.name)()
    train_loader, test_loader = loaders
    train_sampler, test_sampler = samplers

    if cfg.ddp.local_rank == 0:
        print('img_size: {}, num_classes: {}, stem: {}'.format(cfg.dataset.image_size,
                                                           num_classes,
                                                           cfg.model.stem))
    if cfg.ddp.local_rank == 0:
        print('Model Name: {0}'.format(cfg.model.name))
    if cfg.model.name == "MobileNetV3":
        model = MobileNetV3(num_classes=num_classes)
    else:
        model = eval(cfg.model.name)(num_classes=num_classes,
                                     heads=cfg.model.heads,
                                     kernel_size=cfg.model.kernel,
                                     stem=cfg.model.stem,
                                     num_resblock=cfg.model.num_resblock,
                                     attention_logger=attention_logger)

    if cfg.optim.method == 'SGD':
        optimizer = optim.SGD(model.parameters(), **cfg.optim.sgd_params)
    elif cfg.optim.method == 'Adam':
        optimizer = optim.Adam(model.parameters(), **cfg.optim.adam_params)
    elif cfg.optim.method == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), **cfg.optim.rmsprop_params)
    else:
        raise ValueError(f'Unsupported optimization: {cfg.optim.method}')

    scheduler = get_scheduler(optimizer, len(train_loader), cfg)

    if cfg.auto_resume or cfg.test or cfg.finetune.is_finetune:
        if 'finetune' in cfg.dataset.name:
            cfg.dataset.name = cfg.dataset.name.split('_')[1]
        filename = 'best_model_' + cfg.dataset.name + '_' + \
                str(cfg.model.name) + '_' + str(cfg.model.stem) + '_ckpt.tar'
        if cfg.ddp.local_rank == 0:
            print('filename :: ', filename)
        file_path = os.path.join(cfg.test_path, 'checkpoints', filename)
        checkpoint = torch.load(file_path)

        state_dict = checkpoint['state_dict']

        keys = state_dict.copy().keys()
        for key in keys:
            if 'layer1' in key or 'layer2' in key or 'layer3' in key or 'layer4' in key or 'total_ops' in key or 'total_params' in key:
                del state_dict[key]
            # if 'total_ops' in key or 'total_params' in key:
            #     del state_dict[key]

        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']

        if cfg.auto_resume:
            optimizer.load_state_dict(checkpoint['optimizer'])
            # scheduler.load_state_dict(checkpoint['scheduler'])
        if cfg.ddp.local_rank == 0:
            logger.info('Best Epoch: {0}, Best Acc: {1:.1%}'.format(start_epoch, best_acc))
    else:
        start_epoch = cfg.train.start_epoch
        best_acc = 0.0

    get_net_info(model, (3, cfg.dataset.image_size, cfg.dataset.image_size),
                 logger=logger, local_rank=cfg.ddp.local_rank)

    if cfg.crit.smooth > 0:
        criterion = CrossEntropyLabelSmooth(num_classes=num_classes, epsilon=cfg.crit.smooth)
    else:
        criterion = nn.CrossEntropyLoss()

    if cfg.cuda:
        if cfg.ddp.distributed:
            model = model.to(torch.cuda.current_device())
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DDP(model, device_ids=[gpu_id], find_unused_parameters=True)
        elif torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            model = model.cuda()
        else:
            model = model.cuda()
        criterion = criterion.cuda()

    if cfg.test:
        _ = validate(model, test_loader, criterion, start_epoch, logger, attention_logger, writer)
        return

    if cfg.finetune:
        # freeze early layers
        for i, layer in enumerate(model.module.layers):
            if i < 3:
                for param in layer.parameters():
                    param.require_grad = False

        for epoch in range(cfg.finetune.start_epoch, cfg.finetune.epoch + 1):
            if cfg.ddp.distributed:
                train_sampler.set_epoch(epoch)
                test_sampler.set_epoch(epoch)

            train(model, train_loader, optimizer, criterion, scheduler, epoch, logger, writer)
            eval_acc = validate(model, test_loader, criterion, epoch, logger, attention_logger, writer)

            is_best = eval_acc > best_acc
            best_acc = max(eval_acc, best_acc)

            filename = 'model_' + str(cfg.dataset.name) + '_' + \
                       str(cfg.model.name) + '_' + str(cfg.model.stem) + '_finetune_ckpt.tar'
            if cfg.ddp.local_rank == 0:
                print('filename :: ', filename)

            if cfg.ddp.local_rank == 0:
                if torch.cuda.device_count() > 1:
                    save_checkpoint({
                        'epoch': epoch,
                        'arch': cfg.model.name,
                        'state_dict': model.module.state_dict(),
                        'best_acc': best_acc,
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                    }, is_best, cfg.ckp_dir, filename)
                else:
                    save_checkpoint({
                        'epoch': epoch,
                        'arch': cfg.model.name,
                        'state_dict': model.state_dict(),
                        'best_acc': best_acc,
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                    }, is_best, cfg.ckp_dir, filename)

        return

    for epoch in range(start_epoch, cfg.train.epoch + 1):
        if cfg.ddp.distributed:
            train_sampler.set_epoch(epoch)
            test_sampler.set_epoch(epoch)

        train(model, train_loader, optimizer, criterion, scheduler, epoch, logger, writer)
        eval_acc = validate(model, test_loader, criterion, epoch, logger, attention_logger, writer)

        is_best = eval_acc > best_acc
        best_acc = max(eval_acc, best_acc)

        filename = 'model_' + str(cfg.dataset.name) + '_' + \
                   str(cfg.model.name) + '_' + str(cfg.model.stem) + '_ckpt.tar'
        if cfg.ddp.local_rank == 0:
            print('filename :: ', filename)

        if cfg.ddp.local_rank == 0:
            if torch.cuda.device_count() > 1:
                save_checkpoint({
                    'epoch': epoch,
                    'arch': cfg.model.name,
                    'state_dict': model.module.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }, is_best, cfg.ckp_dir, filename)
            else:
                save_checkpoint({
                    'epoch': epoch,
                    'arch': cfg.model.name,
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }, is_best, cfg.ckp_dir, filename)


if __name__ == '__main__':
    main()
