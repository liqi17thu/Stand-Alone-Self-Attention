from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from yacs.config import CfgNode as CN

__C = CN()

cfg = __C

__C.SAVE_PATH = './experiments'

__C.CUDA = True
__C.DISTRIBUTED = True
__C.GPUS = 8

__C.TRAIN = CN()

__C.TRAIN.EPOCH = 100
__C.TRAIN.START_EPOCH = 0
__C.TRAIN.WARMUP_EPOCH = 20
__C.TRAIN.WARMUP_MULTIPLIER = 16
__C.TRAIN.DISP = 100

__C.TRAIN.MODEL = CN()
__C.TRAIN.MODEL.NAME = 'SAResNet26'
__C.TRAIN.MODEL.STEM = 'cifar_conv'
__C.TRAIN.MODEL.HEADS = 8
__C.TRAIN.MODEL.KERNEL = 7
__C.TRAIN.MODEL.PRE_TRAINED = False
__C.TRAIN.MODEL.NUM_SABLOCK = 2

__C.TRAIN.DATASET = CN()
__C.TRAIN.DATASET.NAME = 'cifar10'
__C.TRAIN.DATASET.SPLIT_RATIO = 0.5
__C.TRAIN.DATASET.TRAIN_DIR = '/data/home/v-had/data_local/imagenet/train'
__C.TRAIN.DATASET.TEST_DIR = '/data/home/v-had/data_local/imagenet/val'
__C.TRAIN.DATASET.MEAN = [0.485, 0.456, 0.406]
__C.TRAIN.DATASET.STD = [0.229, 0.224, 0.225]
__C.TRAIN.DATASET.BATCH_SIZE = 25
__C.TRAIN.DATASET.IMAGE_SIZE = 32
__C.TRAIN.DATASET.TEST_RESIZE = 256
__C.TRAIN.DATASET.TEST_SIZE = 224
__C.TRAIN.DATASET.WORKERS = 8
__C.TRAIN.DATASET.USE_AA = True
__C.TRAIN.DATASET.BGR = False

__C.TRAIN.CRIT = CN()
__C.TRAIN.CRIT.SMOOTH=0.1

__C.TRAIN.OPTIM = CN()
__C.TRAIN.OPTIM.METHOD = 'sgd'
__C.TRAIN.OPTIM.LR = 1e-1
__C.TRAIN.OPTIM.MOMENTUM = 0.9
__C.TRAIN.OPTIM.WD = 1e-4