import numpy as np
import argparse
import glob
from os.path import join
import tqdm
import cv2
import os

from pprint import pprint

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def FLAGS():
    parser = argparse.ArgumentParser("""Clean noise in voxel grids with a depth map.""")

    # training / validation dataset
    parser.add_argument("--voxel_dataset", default="", required=True)
    parser.add_argument("--depth_dataset", default="", required=True)
    parser.add_argument("--output_folder", default=".", required=True)
    parser.add_argument("--voxel_offset", type=int, default=0)
    parser.add_argument("--depth_offset", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--idx", type=int, default=-1)

    flags = parser.parse_args()

    return flags


if __name__ == "__main__":
    flags = FLAGS()
    voxel_files = sorted(glob.glob(join(flags.voxel_dataset, '*.npy')))
    voxel_files = voxel_files[flags.voxel_offset:]

    depth_files = sorted(glob.glob(join(flags.depth_dataset, '*npy')))
    depth_files = depth_files[flags.depth_offset:]

    ensure_dir(flags.output_folder)

    num_it = min(len(voxel_files), len(depth_files))
    for idx in tqdm.tqdm(range(num_it)):
        v_file, d_file = voxel_files[idx], depth_files[idx]
        voxel = np.load(v_file)
        depth = np.load(d_file)

        mask = np.tile(depth>0, (voxel.shape[0],1,1))
        voxel_clean = np.where(mask, voxel, 0)
        
        np.save('%s/event_tensor_%010d.npy' % (flags.output_folder, idx), voxel_clean)

    pass