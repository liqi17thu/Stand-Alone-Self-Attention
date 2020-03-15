import argparse
from os.path import join

import os
import shutil

from yacs.config import CfgNode

from lib.utils import check_dir

cfg = CfgNode(dict(
    save_path='./experiments',
    cuda=True,
    auto_resume=False,
    test=False,
    disp_attention=False,

    ddp=dict(
        distributed=True,
        gpus=8,
        dist_url='tcp://127.0.0.1:26443',
        seed=772002,
    ),
    model=dict(
        name='SAResNet26',
        stem='cifar_conv',
        heads=8,
        kernel=7,
        expansion=4,
        pre_trained=False,
        num_resblock=2,
        with_conv=False,
        encoding='learnable',
        temperature=1.0,
        r_dim=256,
    ),
    train=dict(
        epoch=100,
        start_epoch=0,
        disp=10,
    ),
    dataset=dict(
        name="cifar10",
        split_ratio=0.5,
        train_dir="/data/home/v-had/data_local/imagenet/train",
        test_dir="/data/home/v-had/data_local/imagenet/val",
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        batch_size=256,
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
            weight_decay=0.0001,
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

# inference some folder dir
cfg.save_path = join(cfg.save_path, 'train')
check_dir(cfg.save_path)

cfg.save_path = join(cfg.save_path, args.name)

if not os.path.exists(cfg.save_path):
    os.mkdir(cfg.save_path)
elif not cfg.test:
    key = input('Delete Existing Directory [y/n]: ')
    if key == 'n':
        if cfg.auto_resume:
            pass
        else:
            raise ValueError("Save directory already exists")
    elif key == 'y':
        shutil.rmtree(cfg.save_path)
        os.mkdir(cfg.save_path)
    else:
        raise ValueError("Input Not Supported!")

cfg.ckp_dir = check_dir(join(cfg.save_path, 'checkpoints'))
cfg.log_dir = check_dir(join(cfg.save_path, 'runs'))

print(cfg.local_rank)


