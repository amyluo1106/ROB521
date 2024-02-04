#!/usr/bin/env python3
import rospy
import math
from geometry_msgs.msg import Twist
from std_msgs.msg import String

def publisher_node():
    """TODO: initialize the publisher node here, \
            and publish wheel command to the cmd_vel topic')"""
    cmd_pub = rospy.Publisher("cmd_pub", Twist, queue_size = 1)
    
    twist = Twist()
    rate = rospy.Rate(10)
    cmd_pub.publish(twist)
    
    while not rospy.is_shutdown():
    cmd_pub.publish(twist)
    rospy.loginfo("The linear vel is: %s", twist.linear.x)

def main():
    try:
        rospy.init_node('motor')
        publisher_node()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
