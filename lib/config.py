import argparse
from os.path import join

import os
import shutil

from yacs.config import CfgNode

from .utils import check_dir

cfg = CfgNode(dict(
    save_path='./experiments',
    cuda=True,
    distributed=True,
    gpus=8,
    auto_resume=False,
    test=False,
    disp_attention=False,

    model=dict(
        name='SAResNet26',
        stem='cifar_conv',
        heads=8,
        kernel=7,
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
        disp=100,
    ),
    dataset=dict(
        name="cifar10",
        split_ratio=0.5,
        train_dir="/data/home/v-had/data_local/imagenet/train",
        test_dir="/data/home/v-had/data_local/imagenet/val",
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        batch_size=50,
        image_size=32,
        workers=8,
        use_aa=True,
        bgr=False,
    ),
    crit=dict(
        smooth=0.1,
    ),
    optim=dict(
        method="sgd",
        lr=0.05,
        momentum=0.9,
        warmup_epoch=20,
        warmup_multiplier=16,
        wd=0.0001,
    ),
))

# load from file and overrided by command line arguments

parser = argparse.ArgumentParser('Stand-Alone Self-Attention')
parser.add_argument('name', type=str)
parser.add_argument('--cfg', type=str, default='./experiments/yaml/baseline.yaml')

args, unknown = parser.parse_known_args()
cfg.merge_from_file(args.config)
cfg.merge_from_list(unknown)

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


