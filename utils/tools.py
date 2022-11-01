import os
import time
import sys
import numpy as np
from PIL import Image
import cv2

def create_backup_dir(dir):
    backup_dir = dir +'/'+time.strftime('%Y%m%d-%H:%M:%S')
    # backup_dir = dir
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
    return backup_dir

def backup(src, dest):
    dest = create_backup_dir(dest)
    for file in src:
        if not os.path.exists(file):
            print('backup error! {} src dir not exist'.format(file))
            sys.exit(1)
        srcfile = file
        destfile = os.path.join(dest, file.split("/")[-1])
        if os.path.isfile(srcfile):
            open(destfile, "wb").write(open(srcfile, "rb").read())

def backup_1(src, dest):
    if not os.path.exists(dest):
        os.makedirs(dest)
    for file in src:
        if not os.path.exists(file):
            print('backup error! {} src dir not exist'.format(file))
            sys.exit(1)
        srcfile = file
        destfile = os.path.join(dest, file.split("/")[-1])
        if os.path.isfile(srcfile):
            open(destfile, "wb").write(open(srcfile, "rb").read())


def saveRGB(img, path):
    """
    img: 3,256,256
    """
    mean = np.array([[[0.485]], [[0.456]], [[0.406]]])
    std = np.array([[[0.229]], [[0.224]], [[0.225]]])

    img_trans = img * std + mean
    img = np.clip(img_trans, 0, 1)
    img0 = img.transpose((1, 2, 0))
    Image.fromarray((img0 * 255).astype('uint8')).save(path)


def visualize(dic, root, epoch=0, id=0):
    out_path = os.path.join(root, '{}/{}'.format(epoch, id))
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for k, v in dic.items():
        if v.shape[0] == 3:
            v = v.data.cpu().numpy()
            saveRGB(v, out_path + "/{}.png".format(k))
        elif v.shape[0] == 1:
            v = v[0].data.cpu().numpy()
            cv2.imwrite(out_path + "/{}.png".format(k),
                np.array((v * 255), dtype=np.uint8))
        elif len(v.shape) == 2:
            v = v.data.cpu().numpy()
            cv2.imwrite(out_path + "/{}.png".format(k),
                        np.array((v * 255), dtype=np.uint8))
        else:
            print(k, v.shape)
