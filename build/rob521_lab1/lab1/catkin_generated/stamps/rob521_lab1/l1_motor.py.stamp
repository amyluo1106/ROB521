#!/usr/bin/env python3
import rospy
import math
from geometry_msgs.msg import Twist
from std_msgs.msg import String

def publisher_node():
    """TODO: initialize the publisher node here, \
            and publish wheel command to the cmd_vel topic')"""
    cmd_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
    rospy.sleep(1)
    rate = rospy.Rate(10)

    counter = 0

    while counter != 100:
        # go forward 1 meter, rotate clockwise 360 degress, stop
        rate.sleep()
        twist = Twist()
        twist.linear.x = 0.1
        cmd_pub.publish(twist)
        counter +=1

    counter = 0
    while counter != 100:
        rate.sleep()
        twist.angular.z = 2*math.pi/10
        cmd_pub.publish(twist)
        counter +=1


def main():
    try:
        rospy.init_node('motor')
        publisher_node()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
