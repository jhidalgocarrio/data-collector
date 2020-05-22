#!/usr/bin/env python

from __future__ import print_function
from absl import app, flags
import cProfile
import cv2
import cv_bridge
import numpy as np
import rospy
from dvs_msgs.msg import Event, EventArray
import rosbag
import os
import re
import sys
from carla.image_converter import *
from carla.sensor import *

FLAGS = flags.FLAGS

flags.DEFINE_string('in_bag', '', 'Path to input bag')
flags.DEFINE_string('topic', '', 'image topic in bag')
flags.DEFINE_string('name', 'mvsec_event_frame', 'Name')
flags.DEFINE_string('rect_x_map', '', 'Path to the x-coord rectification matrix')
flags.DEFINE_string('rect_y_map', '', 'Path to the y-coord rectification matrix')
flags.DEFINE_bool('visualize', False, 'Visualize tracking?')
flags.DEFINE_string('out_bag', '', 'Path for the output bag')

K = np.array([[223.9940010790056, 0, 170.7684322973841],
              [0, 223.61783486959376, 223.61783486959376],
              [0.0,0.0, 1.0]], dtype = np.float64)

dist_coef = np.array([-0.033904378348448685, -0.01537260902537579, -0.022284741346941413, 0.0069204143687187645], dtype = np.float32)


def main(_):
    pr = cProfile.Profile()
    pr.enable()
    run()
    pr.disable()


def run():
    in_bag = rosbag.Bag(FLAGS.in_bag, 'r')
    out_bag = rosbag.Bag(FLAGS.out_bag, 'w')
    bridge = cv_bridge.CvBridge()
    rectification_map = False

    assert FLAGS.in_bag is not ''
    assert FLAGS.topic is not ''
    assert FLAGS.out_bag is not ''

    if FLAGS.rect_x_map is not '' and FLAGS.rect_y_map is not '':
        rectification_map = True

    print (FLAGS.rect_x_map)
    print (FLAGS.rect_y_map)
    print (rectification_map)
    if rectification_map:
        rect_x_map = np.loadtxt(FLAGS.rect_x_map, dtype=np.float32)
        rect_y_map = np.loadtxt(FLAGS.rect_y_map, dtype=np.float32)
        print ("Loading rectification maps... ", (rect_x_map.shape, rect_y_map.shape))

    print('Undistorting events from %s...' % FLAGS.in_bag)

    maxn = in_bag.get_message_count(FLAGS.topic)

    i = 0
    for topic, msg, t in in_bag.read_messages(topics=[FLAGS.topic]):
        #print (msg.height)
        #print (msg.width)
        #print (len(msg.events))
        for e in range(len(msg.events)):
            #print ("old: ", (msg.events[e].x, msg.events[e].y))
            # x_rect = outdoor_day_left_x(y, x)
            msg.events[e].x = min(msg.width-1, max(0, int(rect_x_map[int(msg.events[e].y), int(msg.events[e].x)])))
            # y_rect = outdoor_day_left_y(y, x)
            msg.events[e].y = min(msg.height-1, max(0, int(rect_y_map[int(msg.events[e].y), int(msg.events[e].x)])))
            #print ("new: ", (msg.events[e].x, msg.events[e].y))
 

        # Do it in this way in case we use th opencv function in the future
        #points = np.array(coord)
        #points = np.expand_dims(points, axis=0).astype(float)

        #new_points = cv2.undistortPoints(src=points, cameraMatrix=K, distCoeffs=dist_coef)
        #new_points = cv2.fisheye.undistortPoints(distorted=points, K=K, D=dist_coef)

        if FLAGS.visualize:
            curr_img = bridge.imgmsg_to_cv2(msg)
            curr_img = np.nan_to_num(curr_img)
            if rectification_map:
                new_img = cv2.remap(src= curr_img, map1=rect_x_map, map2=rect_y_map, interpolation=cv2.INTER_LINEAR)
            else:
                #new_img = cv2.fisheye.undistortImage(distorted=curr_img, K=K, D=dist_coef)
                new_img = cv2.undistort(src=curr_img, cameraMatrix=K, distCoeffs=dist_coef)

            cv2.imshow(FLAGS.name, curr_img)
            cv2.imshow(FLAGS.name+"_undist", new_img)
            cv2.waitKey(1)

        # Write in the new msg in the out bag
        out_bag.write(topic+"_rect", msg, msg.header.stamp)

        print('\r%d / %d' % (i, maxn), end='')
        sys.stdout.flush()
        #sys.stdin.read(1)

        i = i + 1

    print('')

    in_bag.close()
    out_bag.close()

if __name__ == '__main__':
    app.run(main)
