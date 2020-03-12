import os
import random
from os import listdir
from os.path import isfile, isdir, join

import numpy as np

data_path = join(os.getcwd(), 'CUB_200_2011/images')
savedir = './'
dataset_list = ['train', 'test', 'all', 'base', 'val', 'novel']

folder_list = [f for f in listdir(data_path) if isdir(join(data_path, f))]
folder_list.sort()
label_dict = dict(zip(folder_list, range(0, len(folder_list))))

classfile_list_all = []

for i, folder in enumerate(folder_list):
    folder_path = join(data_path, folder)
    classfile_list_all.append([join(folder_path, cf) for cf in listdir(folder_path)
                               if (isfile(join(folder_path, cf)) and cf[0] != '.')])
    random.shuffle(classfile_list_all[i])

for dataset in dataset_list:
    file_list = []
    label_list = []
    for i, classfile_list in enumerate(classfile_list_all):
        sep = int(len(classfile_list)*0.8)
        if dataset == 'train':
            file_list = file_list + classfile_list[:sep]
            label_list = label_list + np.repeat(i, sep).tolist()
        elif dataset == 'test':
            file_list = file_list + classfile_list[sep:]
            label_list = label_list + np.repeat(i, len(classfile_list)-sep).tolist()
        elif dataset == 'all':
            file_list = file_list + classfile_list
            label_list = label_list + np.repeat(i, len(classfile_list)).tolist()
        elif dataset == 'base':
            if i % 2 == 0:
                file_list = file_list + classfile_list
                label_list = label_list + np.repeat(i, len(classfile_list)).tolist()
        elif dataset == 'val':
            if i % 4 == 1:
                file_list = file_list + classfile_list
                label_list = label_list + np.repeat(i, len(classfile_list)).tolist()
        elif dataset == 'novel':
            if i % 4 == 3:
                file_list = file_list + classfile_list
                label_list = label_list + np.repeat(i, len(classfile_list)).tolist()


    fo = open(savedir + dataset + ".json", "w")
    fo.write('{"label_names": [')
    fo.writelines(['"%s",' % item for item in folder_list])
    fo.seek(0, os.SEEK_END)
    fo.seek(fo.tell() - 1, os.SEEK_SET)
    fo.write('],')

    fo.write('"image_names": [')
    fo.writelines(['"%s",' % item for item in file_list])
    fo.seek(0, os.SEEK_END)
    fo.seek(fo.tell() - 1, os.SEEK_SET)
    fo.write('],')

    fo.write('"image_labels": [')
    fo.writelines(['%d,' % item for item in label_list])
    fo.seek(0, os.SEEK_END)
    fo.seek(fo.tell() - 1, os.SEEK_SET)
    fo.write(']}')

    fo.close()
    print("%s -OK" % dataset)
