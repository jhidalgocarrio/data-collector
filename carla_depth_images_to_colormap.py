#!/usr/bin/env python

from __future__ import print_function
from absl import app, flags
import cProfile
import cv2
import numpy as np
import os
import re
import sys
import matplotlib as mpl
import matplotlib.cm as cm
from carla.image_converter import *
from carla.sensor import *

FLAGS = flags.FLAGS

flags.DEFINE_string('image_folder', '', 'Input image folder')
flags.DEFINE_bool('visualize', False, 'Visualize tracking?')
flags.DEFINE_bool('inv', False, 'Perform inverse of the input depth?')
flags.DEFINE_string('filter', '', 'Filter to apply to file names')
flags.DEFINE_string('name', 'track', 'Name of output hickle')
flags.DEFINE_integer('maxn', 0, 'Process only first n frames')
flags.DEFINE_string('output_folder', '.', 'Output image folder')

def readImages(image_folder):
    contents = sorted(os.listdir(image_folder))
    if FLAGS.filter is not '':
        contents = [i for i in contents if re.match(FLAGS.filter, i) is not None]
    return [os.path.join(image_folder, i) for i in contents if
            i.endswith('.png') or i.endswith('.jpg')]


def main(_):
    pr = cProfile.Profile()
    pr.enable()
    run()
    pr.disable()
    pr.dump_stats('depth_gen.profile')


def run():
    if FLAGS.image_folder is not '':
        print('Taking depth from %s...' % FLAGS.image_folder)
        images = readImages(FLAGS.image_folder)
        if FLAGS.maxn == 0:
            maxn = len(images)
        else:
            maxn = FLAGS.maxn

    os.mkdir("%s/depth_colormap"%FLAGS.output_folder)
    i = 0
    for image in images:
        if i == maxn:
            break
        if FLAGS.image_folder is not '':
            curr_img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

        curr_img = np.nan_to_num(curr_img)
        depth = np.ones((curr_img.shape[0], curr_img.shape[1], 4), dtype=np.uint8)
        depth[:,:,0] = np.nan_to_num(curr_img)
        depth[:,:,1] = np.nan_to_num(curr_img)
        depth[:,:,2] = np.nan_to_num(curr_img)
        depth[:,:,3] *= 255
        carla_img = Image(i, depth.shape[1], depth.shape[0], image_type='Depth', fov=85, raw_data=depth.ravel())
        # Convet to array (log depth)
        depth = depth_to_array(carla_img)
        if FLAGS.inv:
            # Convert to normalized depth [0 - 1]
            depth = np.exp(5.70378 * (depth - np.ones((depth.shape[0], depth.shape[1]), dtype=np.float32)))
            # Perform inverse depth
            inv_depth = 1/depth
            inv_depth = inv_depth/np.amax(inv_depth)
            #Convert back to log (inverse log depth)
            depth = np.ones((inv_depth.shape[0], inv_depth.shape[1]), dtype=np.float32) + np.log(inv_depth)/5.70378
            
        # Convert to color map
        vmax = np.percentile(depth, 95)
        normalizer = mpl.colors.Normalize(vmin=depth.min(), vmax=vmax)
        mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
        img = mapper.to_rgba(depth)
        img[:,:,0:3] = img[:,:,0:3][...,::-1]

        if FLAGS.visualize:
            img_viz = img
            cv2.imshow(FLAGS.name, img_viz)
            cv2.waitKey(1)

        if FLAGS.output_folder:
            # Save to image file
            cv2.imwrite('%s/depth_colormap/frame_%010d.png' % (FLAGS.output_folder, i), img*255.0)

        print('\r%d / %d' % (i, maxn), end='')
        sys.stdout.flush()

        i = i + 1
    print('')

if __name__ == '__main__':
    app.run(main)
