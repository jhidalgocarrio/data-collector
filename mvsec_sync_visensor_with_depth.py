from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import math
import argparse
import numpy as np
import cv2
from numpy import genfromtxt 

def parse_args():
    parser = argparse.ArgumentParser(
        description='MVSEC Sync Images to Depth.')

    parser.add_argument('--image_path', type=str,
                        help='path to a test image or folder of images', required=True)

    parser.add_argument('--depth_path', type=str,
                        help='path to a test image or folder of images', required=True)

    parser.add_argument('--output_directory', type=str,
                        help='path to a output directory', required=True)

    parser.add_argument('--ext', type=str,
                        help='image extension to search for in folder', default="png")

    return parser.parse_args()

def find_nearest(array,value): 
    idx = np.searchsorted(array, value, side="left") 
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])): 
        return idx-1, array[idx-1] 
    else: 
        return idx, array[idx] 

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def run(args):

    image_paths = sorted(glob.glob(os.path.join(args.image_path, '*.{}'.format(args.ext))))
    depth_paths = sorted(glob.glob(os.path.join(args.depth_path, '*.npy')))
    ensure_dir(args.output_directory)
    f=open("%s/timestamps.txt"%args.output_directory, "a+")

    image = genfromtxt(os.path.join(args.image_path, 'timestamps.txt'), delimiter=' ')
    image = image[:, 1] # only get the time stamps
    depth = genfromtxt(os.path.join(args.depth_path, 'timestamps.txt'), delimiter=' ')
    depth = depth[:, 1] # only get the time stamps

    for idx_depth, t_depth in enumerate(depth):

        # find the nearest imag to d
        idx_image, t_image = find_nearest(image, t_depth)

        # write the idx and time in the new timestamp file 
        #f.write("%d (%d) %s->%s\n" % (idx_depth, idx_image, t_depth, t_image))
        f.write("%d %s\n" % (idx_depth, t_image))

        image_name = '%s/frame_%010d.%s' % (args.image_path, idx_image, args.ext)
        image_new_name = '%s/frame_%010d.%s' % (args.output_directory, idx_depth, args.ext)
        os.system("cp "+ image_name+" "+image_new_name)
        print('\r%d / %s' % (idx_image, t_depth), end='')

    f.close()
    print ('done')

if __name__ == '__main__':
    args = parse_args()
    run(args)
