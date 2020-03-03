from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import math
import argparse
import numpy as np
import cv2
from numpy import genfromtxt 
import tqdm
from carla.image_converter import *
from carla.sensor import *

def parse_args():
    parser = argparse.ArgumentParser(
        description='MVSEC Sync Images to Depth.')

    parser.add_argument('--input_directory', type=str,
                        help='path to a test image or folder of images', required=True)

    parser.add_argument('--output_directory', type=str,
                        help='path to a output directory', required=True)

    parser.add_argument('--random', action='store_true', default=False)
    
    parser.add_argument('--convert_to_frames', action='store_true', default=False)

    return parser.parse_args()

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def convert_depth_to_image(i, data):
    depth = np.nan_to_num(data, nan=1000.00)
    depth = depth / np.amax(depth)
    img = np.ones((depth.shape[0], depth.shape[1]), dtype=np.float32) + np.log(depth)/5.70378
    return img * 255

def run(args):
    voxel_directory = os.path.join(args.input_directory, 'events/voxels') 
    depth_directory = os.path.join(args.input_directory, 'depth/data') 
    input_voxels = sorted(glob.glob(os.path.join(voxel_directory, '*.npy')))
    input_depth = sorted(glob.glob(os.path.join(depth_directory, '*.npy')))

    ensure_dir(args.output_directory)
    voxel_train_directory = os.path.join(args.output_directory, "train/sequence_0000000000/events/voxels")
    voxel_valid_directory = os.path.join(args.output_directory, "validation/sequence_0000000000/events/voxels")
    voxel_test_directory = os.path.join(args.output_directory, "test/sequence_0000000000/events/voxels")

    depth_train_directory = os.path.join(args.output_directory, "train/sequence_0000000000/depth/data")
    depth_valid_directory = os.path.join(args.output_directory, "validation/sequence_0000000000/depth/data")
    depth_test_directory = os.path.join(args.output_directory, "test/sequence_0000000000/depth/data")
 
    ensure_dir(voxel_train_directory)
    ensure_dir(voxel_valid_directory)
    ensure_dir(voxel_test_directory)

    ensure_dir(depth_train_directory)
    ensure_dir(depth_valid_directory)
    ensure_dir(depth_test_directory)

    voxel_timestamps = genfromtxt(os.path.join(voxel_directory, 'timestamps.txt'), delimiter=' ')
    voxel_dic = dict(zip(voxel_timestamps[:,0], np.sort(voxel_timestamps[:,1])))
    voxel_idx = voxel_timestamps[:,0]
    assert len(voxel_idx) == len(input_voxels)
    
    voxel_boundary = genfromtxt(os.path.join(voxel_directory, 'boundary_timestamps.txt'), delimiter=' ')
    boundary_dic = dict(zip(voxel_boundary[:,0], zip(voxel_boundary[:,1], voxel_boundary[:,2])))

    train_samples = int(0.7 * len(voxel_idx))
    if args.random:
        train_idx = np.sort(np.random.choice(np.arange(len(voxel_idx)), size=train_samples, replace=False))
        valid_idx, test_idx = np.split(np.sort(voxel_idx[sorted(list(set(range(len(voxel_idx))) - set(train_idx)))]), 2)
        assert len(valid_idx) + len(test_idx) + len(train_idx) == len(voxel_idx)
    else:
        rest_samples = int((len(voxel_idx) - train_samples)/2)
        train_idx, valid_idx, test_idx, _ = np.split(voxel_idx, [train_samples, train_samples+rest_samples, train_samples+(2*rest_samples)])

    print ("train samples %s, valid samples %s and test samples %s (Total %s)" % (len(train_idx), len(valid_idx), len(test_idx), len(voxel_idx)))


    depth_timestamps = genfromtxt(os.path.join(depth_directory, 'timestamps.txt'), delimiter=' ')
    depth_dic = dict(zip(depth_timestamps[:,0], np.sort(depth_timestamps[:,1])))
    depth_idx = depth_timestamps[:,0]
    assert len(depth_idx) == len(input_depth)

    # Copy train dataset to folder
    f=open("%s/timestamps.txt"%voxel_train_directory, "a+")
    d=open("%s/timestamps.txt"%depth_train_directory, "a+")
    b=open("%s/boundary_timestamps.txt"%voxel_train_directory, "a+")
    for i, idx in enumerate(train_idx):
        voxel_name = '%s/event_tensor_%010d.npy' % (voxel_directory, idx)
        new_voxel_name = '%s/event_tensor_%010d.npy' % (voxel_train_directory, i)
        os.system("cp " + voxel_name + " " + new_voxel_name)
        f.write("%d %f\n" % (i, voxel_dic[idx]))
        b.write("%d %f %f\n" % (i, boundary_dic[idx][0], boundary_dic[idx][1]))
        depth_name = '%s/depth_%010d.npy' % (depth_directory, idx+1)
        d.write("%d %f\n" % (i, depth_dic[idx+1]))
        if args.convert_to_frames:
            data = np.load(depth_name)
            img = convert_depth_to_image(i, data)
            new_depth_name = '%s/frame_%010d.png' % (depth_train_directory, i)
            cv2.imwrite(new_depth_name, img)
        else:
            new_depth_name = '%s/depth_%010d.npy' % (depth_train_directory, i)
            os.system("cp " + depth_name + " " + new_depth_name)


    f.close()
    b.close()
    d.close()

    # Copy validaton dataset to folder
    f=open("%s/timestamps.txt"%voxel_valid_directory, "a+")
    d=open("%s/timestamps.txt"%depth_valid_directory, "a+")
    b=open("%s/boundary_timestamps.txt"%voxel_valid_directory, "a+")
    for i, idx in enumerate(valid_idx):
        voxel_name = '%s/event_tensor_%010d.npy' % (voxel_directory, idx)
        new_voxel_name = '%s/event_tensor_%010d.npy' % (voxel_valid_directory, i)
        os.system("cp " + voxel_name + " " + new_voxel_name)
        f.write("%d %f\n" % (i, voxel_dic[idx]))
        b.write("%d %f %f\n" % (i, boundary_dic[idx][0], boundary_dic[idx][1]))
        depth_name = '%s/depth_%010d.npy' % (depth_directory, idx+1)
        d.write("%d %f\n" % (i, depth_dic[idx+1]))
        if args.convert_to_frames:
            data = np.load(depth_name)
            img = convert_depth_to_image(i, data)
            new_depth_name = '%s/frame_%010d.png' % (depth_valid_directory, i)
            cv2.imwrite(new_depth_name, img)
        else:
            new_depth_name = '%s/depth_%010d.npy' % (depth_valid_directory, i)
            os.system("cp " + depth_name + " " + new_depth_name)

    f.close()
    b.close()
    d.close()

    # Copy validaton dataset to folder
    f=open("%s/timestamps.txt"%voxel_test_directory, "a+")
    d=open("%s/timestamps.txt"%depth_test_directory, "a+")
    b=open("%s/boundary_timestamps.txt"%voxel_test_directory, "a+")
    for i, idx in enumerate(test_idx):
        voxel_name = '%s/event_tensor_%010d.npy' % (voxel_directory, idx)
        new_voxel_name = '%s/event_tensor_%010d.npy' % (voxel_test_directory, i)
        os.system("cp " + voxel_name + " " + new_voxel_name)
        f.write("%d %f\n" % (i, voxel_dic[idx]))
        b.write("%d %f %f\n" % (i, boundary_dic[idx][0], boundary_dic[idx][1]))
        depth_name = '%s/depth_%010d.npy' % (depth_directory, idx+1)
        d.write("%d %f\n" % (i, depth_dic[idx+1]))
        if args.convert_to_frames:
            data = np.load(depth_name)
            img = convert_depth_to_image(i, data)
            new_depth_name = '%s/frame_%010d.png' % (depth_test_directory, i)
            cv2.imwrite(new_depth_name, img)
        else:
            new_depth_name = '%s/depth_%010d.npy' % (depth_test_directory, i)
            os.system("cp " + depth_name + " " + new_depth_name)

    f.close()
    b.close()
    d.close()

    print ('done')

if __name__ == '__main__':
    args = parse_args()
    run(args)

