import numpy as np
import argparse

from .lib import get_attention

parser = argparse.ArgumentParser('parameters')
parser.add_argument('path', type=str)
parser.add_argument('--layer', type=int, default=3)
parser.add_argument('--block', type=int, default=0)
parser.add_argument('--head', type=int, default=0)
parser.add_argument('--height', type=int, default=3)
parser.add_argument('--width', type=int, default=3)
parser.add_argument('--kernel', type=int, default=7)

args = parser.parse_args()

path = args.path
layer = args.layer
block = args.block
head = args.head
height = args.height
width = args.width
kernel = args.kernel


print(get_attention(path, block, head, height, width, kernel))

