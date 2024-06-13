import argparse
import sys
import os
import numpy as np
import cv2
from glob import glob
from pathlib import Path
from PIL import Image

root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

from wildscenes.tools.utils2d import METAINFO, class_2_cidx, cidx_2_rgb
from wildscenes.configs.benchmark_palette_remap import custom_label_map


'''
view_image.py

This script allows for viewing 2D labeled images. Input argument options provided are:
--loaddir
    Set to full path to a Wildscenes2d subfolder, e.g. V-01
--loadidx
    Specify which specific image to load, or from which image to begin viewing a sequence of images
--sequential
    Default is to load a single image then exit. Sequential loads images one after another - press a key to change 
    the image
--video
    Plays the images sequentially in a continuous video
--videospeed
    Defines the video playback speed, lower is faster. Setting to zero defaults to sequential mode
--raw
    Default is to show the benchmark class set. Raw displays the full class set instead, including rare classes 
'''


def load_image(imgfile, remap):
    img = Image.open(imgfile).convert("P")
    base_palette = [254] * 256*3
    for k, v in cidx_2_rgb.items():
        if remap is not None:
            v = cidx_2_rgb[remap[k]] # remap to benchmark colors
        kk = k*3
        base_palette[kk] = v[0]
        base_palette[kk+1] = v[1]
        base_palette[kk+2] = v[2]

    img.putpalette(base_palette)
    outimg = img.convert("RGB")

    return outimg


def view_image(pil_image, sequential=False, videospeed=0):
    open_cv_image = np.array(pil_image)
    open_cv_image = open_cv_image[:, :, ::-1].copy()

    cv2.imshow('labelled image', open_cv_image)
    cv2.waitKey(int(videospeed*1000))
    if not sequential:
        cv2.destroyAllWindows()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--loaddir', default=str(root_dir.parent / 'Wildscenes2d' / 'K-01'),
                        help="Path to directory in WildScenes to read data, for example K-01")
    parser.add_argument('--loadidx', default=-1, type=int,
                        help="Specify which cloud index you want to view. Defaults to a random image from the traverse")
    parser.add_argument('--sequential', default=False, action='store_true',
                        help="Iteratively view all images in a traverse, starting from 0 or loadidx")
    parser.add_argument('--video', default=False, action='store_true',
                        help="View the images as a continuous video, starting from 0 or loadidx")
    parser.add_argument('--videospeed', default=0.5, type=float,
                        help='Video playback speed, lower is faster')
    parser.add_argument('--raw', action='store_true',
                        help="View the raw labels, the full set where pole, asphalt, vehicle are included")
    args = parser.parse_args()

    if not os.path.exists(args.loaddir):
        raise ValueError('Please set the loaddir argument to a subfolder (e.g. K-01) inside the WildScenes dataset. '
                         'This path (full path) should be set to whereever you downloaded the WildScenes dataset to.')

    images = sorted(glob(os.path.join(args.loaddir, 'image', '*')))
    labels = sorted(glob(os.path.join(args.loaddir, 'indexLabel', '*'))) # label / indexLabel

    if args.loadidx >= len(labels):
        raise ValueError('Your loadidx is greater than the number of images in this traverse')

    if args.raw:
        remap = None
    else:
        remaplist = []
        for classname, color, index in zip(METAINFO['classes'], METAINFO['palette'], METAINFO['cidx']):
            remapped_name = custom_label_map[classname]
            remapped_index = class_2_cidx[remapped_name]
            remaplist.append(remapped_index)

        remap = {i:r for i, r in zip(METAINFO['cidx'], remaplist)}

    if args.sequential:
        if args.loadidx == -1:
            args.loadidx = 0
        for idx in range(args.loadidx, len(labels)):
            labelimg = load_image(labels[idx], remap)
            view_image(labelimg, sequential=True)
    elif args.video:
        if args.loadidx == -1:
            args.loadidx = 0
        for idx in range(args.loadidx, len(labels)):
            labelimg = load_image(labels[idx], remap)
            view_image(labelimg, sequential=True, videospeed=args.videospeed)
    else:
        if args.loadidx == -1:
            args.loadidx = np.random.randint(len(labels))
        labelimg = load_image(labels[args.loadidx], remap)
        view_image(labelimg)

    cv2.destroyAllWindows()
    print('EXITING')