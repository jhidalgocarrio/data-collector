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
    os.mkdir("%s/visensor_left"%FLAGS.output_folder)
    f=open("%s/visensor_left/timestamps.txt"%FLAGS.output_folder, "a+")

    assert FLAGS.bag is not ''
    assert FLAGS.topic is not ''
    print('Saving mvsec vi-sensor images from %s...' % FLAGS.bag)
    bag = rosbag.Bag(FLAGS.bag)
    images = bag.read_messages(topics=[FLAGS.topic])
    if FLAGS.maxn == 0:
        maxn = sum([1 for _ in images])
        images = bag.read_messages(topics=[FLAGS.topic])
    else:
        maxn = FLAGS.maxn
    bridge = cv_bridge.CvBridge()

    i = 0
    for image in images:
        if i == maxn:
            break

        f.write("%d %f\n" % (i, image.message.header.stamp.to_sec()))
        curr_img = bridge.imgmsg_to_cv2(image.message)
        # 180 degrees
        rot = cv2.getRotationMatrix2D((curr_img.shape[1]/2, curr_img.shape[0]/2), 180, 1.0)
        curr_img = cv2.warpAffine(curr_img, rot, (curr_img.shape[1], curr_img.shape[0]))

        if FLAGS.output_folder:
            # Save to images
            cv2.imwrite('%s/visensor_left/frame_%010d.png' % (FLAGS.output_folder, i), curr_img)

        if FLAGS.visualize:
            curr_img = np.nan_to_num(curr_img)
            cv2.imshow(FLAGS.name, curr_img)
            cv2.waitKey(1)

        print('\r%d / %d' % (i, maxn), end='')
        sys.stdout.flush()

        i = i + 1

    f.close()
    print('')

if __name__ == '__main__':
    app.run(main)
