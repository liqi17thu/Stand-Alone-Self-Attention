import argparse
from os.path import join

import os
import shutil

from yacs.config import CfgNode

from lib.utils import check_dir

cfg = CfgNode(dict(
    save_path='experiments',
    cuda=True,
    auto_resume=False,
    test=False,
    test_name='default',
    disp_attention=True,

    ddp=dict(
        distributed=True,
        gpus=8,
        local_rank=0,
        dist_url='tcp://127.0.0.1:26443',
        seed=772002,
    ),
    model=dict(
        name='SAResNet26',
        stem='cifar_conv',
        heads=8,
        kernel=7,
        expansion=4,
        num_resblock=2,
        with_conv=False,
        encoding='learnable',
        temperature=1.0,
        r_dim=256,
        rezero=True,
    ),
    train=dict(
        epoch=200,
        attention_epoch=130,
        start_epoch=0,
        disp=1,
    ),
    dataset=dict(
        name="cifar10",
        split_ratio=0.5,
        train_dir="/data/home/v-had/data_local/imagenet/train",
        test_dir="/data/home/v-had/data_local/imagenet/val",
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        batch_size=512,
        image_size=32,
        test_resize=256,
        workers=8,
        use_aa=True,
        bgr=False,
    ),
    crit=dict(
        smooth=0.1,
    ),
    optim=dict(
        method="SGD",
        sgd_params=dict(
            lr=0.5,
            momentum=0.9,
            weight_decay=0.0007,
        ),
        adam_params=dict(
            lr=3e-3,
            weight_decay=0.0001,
        ),
        warmup_epoch=20,
        warmup_multiplier=16,
    ),
))

# load from file and overrided by command line arguments

parser = argparse.ArgumentParser('Stand-Alone Self-Attention')
parser.add_argument('name', type=str)
parser.add_argument('--cfg', type=str, default='./experiments/yaml/baseline.yaml')
parser.add_argument('--local_rank', type=int, default=0, help='local_rank')

args, unknown = parser.parse_known_args()
cfg.merge_from_file(args.cfg)
cfg.merge_from_list(unknown)

if cfg.ddp.distributed:
    cfg.ddp.local_rank = args.local_rank
    cfg.ddp.local_rank = cfg.ddp.local_rank % cfg.ddp.gpus
    # cfg.ddp.dist_url = 'tcp://' + os.environ['MASTER_ADDR'] + ":" + os.environ['MASTER_PORT']
    # cfg.ddp.local_rank = int(os.environ.get('OMPI_COMM_WORLD_RANK') or 0)
    # cfg.ddp.word_size = int(os.environ.get('OMPI_COMM_WORLD_SIZE') or 1)

if cfg.ddp.local_rank == 0:
    # inference some folder dir
    cfg.save_path = join(cfg.save_path, 'train')
    check_dir(cfg.save_path)
    if cfg.test_name == 'default':
        cfg.test_path = join(cfg.save_path, args.name)
    else:
        cfg.test_path = check_dir(join(cfg.save_path, cfg.test_name))

    cfg.save_path = join(cfg.save_path, args.name)

    if not os.path.exists(cfg.save_path):
        os.mkdir(cfg.save_path)
    elif not cfg.test and not cfg.auto_resume:
        key = input('Delete Existing Directory [y/n]: ')
        if key == 'n':
            raise ValueError("Save directory already exists")
        elif key == 'y':
            shutil.rmtree(cfg.save_path)
            os.mkdir(cfg.save_path)
        else:
            raise ValueError("Input Not Supported!")
    cfg.ckp_dir = check_dir(join(cfg.save_path, 'checkpoints'))
    cfg.log_dir = check_dir(join(cfg.save_path, 'runs'))
else:
    if cfg.test_name == 'default':
        cfg.test_path = join(cfg.save_path, 'train', args.name)
    else:
        cfg.test_path = join(cfg.save_path, 'train', cfg.test_name)
    cfg.save_path = join(cfg.save_path, 'train', args.name)
    cfg.ckp_dir = join(cfg.save_path, 'checkpoints')
    cfg.log_dir = join(cfg.save_path, 'runs')
