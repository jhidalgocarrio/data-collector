#!/usr/bin/env python

from __future__ import print_function
from absl import app, flags
import cProfile
import cv2
import cv_bridge
import numpy as np
import rosbag
import os
import re
import sys
import matplotlib as mpl
import matplotlib.cm as cm
from carla.image_converter import *
from carla.sensor import *

FLAGS = flags.FLAGS

flags.DEFINE_string('bag', '', 'Path to input bag')
flags.DEFINE_string('topic', '', 'image topic in bag')
flags.DEFINE_string('name', 'mvsec_depth', 'Name of output hickle')
flags.DEFINE_bool('visualize', False, 'Visualize tracking?')
flags.DEFINE_integer('maxn', 0, 'Process only first n frames')
flags.DEFINE_string('output_folder', '.', 'Output image folder')

def main(_):
    pr = cProfile.Profile()
    pr.enable()
    run()
    pr.disable()


def run():
    os.mkdir("%s/data"%FLAGS.output_folder)
    f=open("%s/data/timestamps.txt"%FLAGS.output_folder, "a+")

    assert FLAGS.bag is not ''
    assert FLAGS.topic is not ''
    print('Converting mvsec depth images from %s...' % FLAGS.bag)
    bag = rosbag.Bag(FLAGS.bag)
    images = bag.read_messages(topics=[FLAGS.topic])
    if FLAGS.maxn == 0:
        maxn = sum([1 for _ in images])
        images = bag.read_messages(topics=[FLAGS.topic])
    else:
        maxn = FLAGS.maxn
    bridge = cv_bridge.CvBridge()

    i = 0
    for topic, image, t in bag.read_messages(topics=[FLAGS.topic]):
        if i == maxn:
            break

        f.write("%d %f\n" % (i, image.header.stamp.to_sec()))
        curr_img = bridge.imgmsg_to_cv2(image)

        if FLAGS.output_folder:
            # Save to numpy
            np.save('%s/data/depth_%010d.npy' % (FLAGS.output_folder, i), curr_img)

        if FLAGS.visualize:
            #curr_img = np.nan_to_num(curr_img, nan=150.00) # only works in numpy >=1.17
            where_are_nan = np.isnan(curr_img)
            other_img = curr_img.copy()
            other_img[where_are_nan] = np.max(np.nan_to_num(curr_img))
            depth = np.ones((other_img.shape[0], other_img.shape[1], 4), dtype=np.uint8)
            depth[:,:,0] = np.nan_to_num(other_img)
            depth[:,:,1] = np.nan_to_num(other_img)
            depth[:,:,2] = np.nan_to_num(other_img)
            depth[:,:,3] *= 255
            img = Image(i, depth.shape[1], depth.shape[0], image_type='Depth', fov=85, raw_data=depth.ravel())
            # Convet to array
            depth = depth_to_logarithmic_grayscale(img)
            depth = depth[:, :, 0]
            depth = depth / 255.0
            # Convert to color map
            vmax = np.percentile(depth, 95)
            normalizer = mpl.colors.Normalize(vmin=depth.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma_r')
            img = mapper.to_rgba(depth)
            img[:,:,0:3] = img[:,:,0:3][...,::-1]
            depthviz = img
            cv2.imshow(FLAGS.name, depthviz)
            cv2.waitKey(1)
            cv2.imwrite('%s/data/frame_%010d.png' % (FLAGS.output_folder, i), img*255.0)

        print('\r%d / %d' % (i, maxn), end='')
        sys.stdout.flush()

        i = i + 1

    f.close()
    print('')

if __name__ == '__main__':
    app.run(main)
