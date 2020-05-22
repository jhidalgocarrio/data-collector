  
#!/usr/bin/env python
import sys
import argparse
import roslib;
import rospy
import rosbag
from rospy import rostime
import argparse
import os

def FLAGS():
    parser = argparse.ArgumentParser("""Datatype timestamp to rosbag timestamp""")

    # training / validation dataset
    parser.add_argument("--bag", default="", required=True)
    parser.add_argument("--out", default="", required=True)

    flags = parser.parse_args()

    return flags


flags = FLAGS()

bag = rosbag.Bag(flags.bag, 'r')
out = rosbag.Bag(flags.out, 'w')

for topic, msg, t in bag.read_messages():
    print ("t: %s new t: %s" %(t, msg.header.stamp))
    out.write(topic, msg, msg.header.stamp)

bag.close()
out.close()
