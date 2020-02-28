#!/usr/bin/env python

from __future__ import print_function
from absl import app, flags
import cProfile
import cv2
import cv_bridge
import hickle as hkl
import numpy as np
import os
import re
import rosbag
import sys
from carla.image_converter import *
from carla.sensor import *

FLAGS = flags.FLAGS

flags.DEFINE_string('image_folder', '', 'Input image folder')
flags.DEFINE_string('bag', '', 'Path to input bag')
flags.DEFINE_string('topic', '', 'image topic in bag')
flags.DEFINE_bool('visualize', False, 'Visualize tracking?')
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
        print(' depth from %s...' % FLAGS.image_folder)
        images = readImages(FLAGS.image_folder)
        if FLAGS.maxn == 0:
            maxn = len(images)
        else:
            maxn = FLAGS.maxn
    else:
        assert FLAGS.bag is not ''
        assert FLAGS.topic is not ''
        print('Converting to normalized depth from %s...' % FLAGS.bag)
        bag = rosbag.Bag(FLAGS.bag)
        images = bag.read_messages(topics=[FLAGS.topic])
        if FLAGS.maxn == 0:
            maxn = sum([1 for _ in images])
            images = bag.read_messages(topics=[FLAGS.topic])
        else:
            maxn = FLAGS.maxn
        bridge = cv_bridge.CvBridge()

    i = 0
    mags = []
    for image in images:
        if i == maxn:
            break
        if FLAGS.image_folder is not '':
            curr_img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        else:
            print("\n", image.message.header.stamp)
            curr_img = bridge.imgmsg_to_cv2(image.message)

        curr_img = np.nan_to_num(curr_img)
        #depth = np.ones((curr_img.shape[0], curr_img.shape[1], 4), dtype=np.uint8)
        #depth[:,:,0] = np.nan_to_num(curr_img)
        #depth[:,:,1] = np.nan_to_num(curr_img)
        #depth[:,:,2] = np.nan_to_num(curr_img)
        #depth[:,:,3] *= 255
        #img = Image(i, depth.shape[1], depth.shape[0], image_type='Depth', fov=85, raw_data=depth.ravel())
        #log_depth = depth_to_logarithmic_grayscale(img)
        print("shape", curr_img.shape)
        print("max", np.max(curr_img))
        print("min", np.min(curr_img))
        if FLAGS.visualize:
            depthviz = curr_img
            cv2.imshow(FLAGS.name, depthviz)
            cv2.waitKey(1)
            cv2.imwrite(
                '%s/frame_%010d.png' % (FLAGS.output_folder, i), depthviz)

        print('\r%d / %d' % (i, maxn), end='')
        sys.stdout.flush()

        i = i + 1
    print('')

    #hkl.dump(mags, open(FLAGS.name + '_mags.hkl', 'w'))


if __name__ == '__main__':
    app.run(main)
