  
#!/usr/bin/env python
import sys
import roslib;
import rospy
import rosbag
from rospy import rostime
import argparse
import os


out = rosbag.Bag("/home/javi/ros/datasets/mvsec/outdoor_night1_davis_left_data_and_gt.bag", 'w')
bag = rosbag.Bag('/home/javi/ros/datasets/mvsec/2020-02-28-18-33-29.bag', 'r')

for topic, msg, t in bag.read_messages():
    print ("t: %s new t: %s" %(t, msg.header.stamp))
    out.write(topic, msg, msg.header.stamp)

bag.close()
out.close()
