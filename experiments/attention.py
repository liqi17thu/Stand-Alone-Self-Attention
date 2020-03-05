import numpy as np
import argparse

parser = argparse.ArgumentParser('parameters')
parser.add_argument('path', type=str)
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--layer', type=int, default=3)
parser.add_argument('--block', type=int, default=0)
parser.add_argument('--head', type=int, default=0)
parser.add_argument('--height', type=int, default=3)
parser.add_argument('--width', type=int, default=3)
parser.add_argument('--kernel', type=int, default=7)

args = parser.parse_args()

epoch = args.epoch
layer = args.layer
block = args.block
head = args.head
height = args.height
width = args.width
kernel = args.kernel


print(get_attention(path, block, head, height, width, kernel))

