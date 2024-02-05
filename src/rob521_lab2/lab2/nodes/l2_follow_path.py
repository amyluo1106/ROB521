#!/usr/bin/env python3
from __future__ import division, print_function
import os

import numpy as np
from scipy.linalg import block_diag
from scipy.spatial.distance import cityblock
import rospy
import tf2_ros
import math

# msgs
from geometry_msgs.msg import TransformStamped, Twist, PoseStamped
from nav_msgs.msg import Path, Odometry, OccupancyGrid
from visualization_msgs.msg import Marker

# ros and se2 conversion utils
import utils

TRANS_GOAL_TOL = .1  # m, tolerance to consider a goal complete
ROT_GOAL_TOL = .3  # rad, tolerance to consider a goal complete
TRANS_VEL_OPTS = [0, 0.025, 0.13, 0.26]  # m/s, max of real robot is .26
ROT_VEL_OPTS = np.linspace(-1.82, 1.82, 11)  # rad/s, max of real robot is 1.82
CONTROL_RATE = 5  # Hz, how frequently control signals are sent
CONTROL_HORIZON = 5  # seconds. if this is set too high and INTEGRATION_DT is too low, code will take a long time to run!
INTEGRATION_DT = .025  # s, delta t to propagate trajectories forward by
COLLISION_RADIUS = 0.225  # m, radius from base_link to use for collisions, min of 0.2077 based on dimensions of .281 x .306
ROT_DIST_MULT = .1  # multiplier to change effect of rotational distance in choosing correct control
OBS_DIST_MULT = .1  # multiplier to change the effect of low distance to obstacles on a path
MIN_TRANS_DIST_TO_USE_ROT = TRANS_GOAL_TOL  # m, robot has to be within this distance to use rot distance in cost
PATH_NAME = 'shortest_path_RRT.npy'  # saved path from l2_planning.py, should be in the same directory as this file

# here are some hardcoded paths to use if you want to develop l2_planning and this file in parallel
# TEMP_HARDCODE_PATH = [[2, 0, 0], [2.75, -1, -np.pi/2], [2.75, -4, -np.pi/2], [2, -4.4, np.pi]]  # almost collision-free
TEMP_HARDCODE_PATH = [[2, -.5, 0], [2.4, -1, -np.pi/2], [2.45, -3.5, -np.pi/2], [1.5, -4.4, np.pi]]  # some possible collisions


class PathFollower():
    def __init__(self):
        # time full path
        self.path_follow_start_time = rospy.Time.now()

        # use tf2 buffer to access transforms between existing frames in tf tree
        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)
        rospy.sleep(1.0)  # time to get buffer running

        # constant transforms
        self.map_odom_tf = self.tf_buffer.lookup_transform('map', 'odom', rospy.Time(0), rospy.Duration(2.0)).transform

        # subscribers and publishers
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.global_path_pub = rospy.Publisher('~global_path', Path, queue_size=1, latch=True)
        self.local_path_pub = rospy.Publisher('~local_path', Path, queue_size=1)
        #self.local_path_candidates_pub = rospy.Publisher('~local_path_candidates', Path, queue_size=1)
        self.collision_marker_pub = rospy.Publisher('~collision_marker', Marker, queue_size=1)

        # map
        map = rospy.wait_for_message('/map', OccupancyGrid)
        self.map_np = np.array(map.data).reshape(map.info.height, map.info.width)
        self.map_resolution = round(map.info.resolution, 5)
        self.map_origin = -utils.se2_pose_from_pose(map.info.origin)  # negative because of weird way origin is stored
        self.map_nonzero_idxes = np.argwhere(self.map_np)

        # collisions
        self.collision_radius_pix = COLLISION_RADIUS / self.map_resolution
        self.collision_marker = Marker()
        self.collision_marker.header.frame_id = '/map'
        self.collision_marker.ns = '/collision_radius'
        self.collision_marker.id = 0
        self.collision_marker.type = Marker.CYLINDER
        self.collision_marker.action = Marker.ADD
        self.collision_marker.scale.x = COLLISION_RADIUS * 2
        self.collision_marker.scale.y = COLLISION_RADIUS * 2
        self.collision_marker.scale.z = 1.0
        self.collision_marker.color.g = 1.0
        self.collision_marker.color.a = 0.5

        # transforms
        self.map_baselink_tf = self.tf_buffer.lookup_transform('map', 'base_link', rospy.Time(0), rospy.Duration(2.0))
        self.pose_in_map_np = np.zeros(3)
        self.pos_in_map_pix = np.zeros(2)
        self.update_pose()

        # path variables
        cur_dir = os.path.dirname(os.path.realpath(__file__))

        # to use the temp hardcoded paths above, switch the comment on the following two lines
        self.path_tuples = np.load(os.path.join(cur_dir, PATH_NAME)).T # (N, 3)
        # self.path_tuples = np.array(TEMP_HARDCODE_PATH)

        self.path = utils.se2_pose_list_to_path(self.path_tuples, 'map')
        self.global_path_pub.publish(self.path)

        # goal
        self.cur_path_index = 1
        self.cur_goal = np.array(self.path_tuples[self.cur_path_index])

        # trajectory rollout tools
        # self.all_opts is a Nx2 array with all N possible combinations of the t and v vels, scaled by integration dt
        self.all_opts = np.array(np.meshgrid(TRANS_VEL_OPTS, ROT_VEL_OPTS)).T.reshape(-1, 2)

        # if there is a [0, 0] option, remove it
        all_zeros_index = (np.abs(self.all_opts) < [0.001, 0.001]).all(axis=1).nonzero()[0]
        if all_zeros_index.size > 0:
            self.all_opts = np.delete(self.all_opts, all_zeros_index, axis=0)
        self.all_opts_scaled = self.all_opts * INTEGRATION_DT

        self.num_opts = self.all_opts_scaled.shape[0]
        self.horizon_timesteps = int(np.ceil(CONTROL_HORIZON / INTEGRATION_DT))

        self.rate = rospy.Rate(CONTROL_RATE)

        rospy.on_shutdown(self.stop_robot_on_shutdown)

        self.prev_control = None
        self.prev_path = None

        self.follow_path()

    def follow_path(self):
        while not rospy.is_shutdown():
            # timing for debugging...loop time should be less than 1/CONTROL_RATE
            tic = rospy.Time.now()

            self.update_pose()
            self.check_and_update_goal()

            dist_to_goal = np.linalg.norm(self.pose_in_map_np[:2] - self.cur_goal[:2])
            
            updating_control_horizon = np.clip(dist_to_goal/0.26, 1, 5)

            # start trajectory rollout algorithm
            local_paths = self.trajectory_rollout(self.all_opts[:, 0:1], self.all_opts[:, 1:2], self.pose_in_map_np[:, None], fix_col = False, timestep = updating_control_horizon, num_substeps = int(np.ceil(CONTROL_HORIZON / INTEGRATION_DT)))
            col_traj_mask = np.any(np.any(np.isnan(local_paths), axis = -1), axis = -1)
            valid_opts = np.array(range(self.num_opts))
            # remove collision trajs
            local_paths = local_paths[~col_traj_mask] 
            valid_opts = valid_opts[~col_traj_mask]

            trans_cost = np.linalg.norm(local_paths[:, -1, :2] - self.cur_goal[:2], axis = -1)
            err = local_paths[:, -1, 2] - self.cur_goal[2]
            rot_cost = np.abs(np.arctan2(np.sin(err), np.cos(err)))
            rot_cost = np.where(rot_cost > np.pi, rot_cost - 2*np.pi,  rot_cost)

            rot_mult = ROT_DIST_MULT if np.linalg.norm(self.pose_in_map_np[:2] - self.cur_goal[:2]) < MIN_TRANS_DIST_TO_USE_ROT else 0

            sim_cost = np.zeros_like(trans_cost)
            
            final_cost = trans_cost + rot_mult * rot_cost

            if final_cost.size == 0:
                control = [-.1, 0]
            else:
                best_opt = valid_opts[final_cost.argmin()]
                control = self.all_opts[best_opt]
                traj = local_paths[final_cost.argmin()]
                self.prev_path = traj
                traj = utils.se2_pose_list_to_path(traj, 'map')
                self.local_path_pub.publish(traj)  

            self.prev_control = control

            # send command to robot
            self.cmd_pub.publish(utils.unicyle_vel_to_twist(control))

            # uncomment out for debugging if necessary
            #print("Selected control: {control}, Loop time: {time}, Max time: {max_time}".format(
            #    control=control, time=(rospy.Time.now() - tic).to_sec(), max_time=1/CONTROL_RATE))

            self.rate.sleep()

    def trajectory_rollout(self, vel, rot_vel, pos, fix_col, num_substeps, timestep):
        # Generate time steps for the trajectory
        timesteps = np.linspace(0, timestep, num_substeps)[None, :]

        # Extract initial positions (x0, y0, theta0) from the 'pos' array
        x0, y0, theta0 = pos[0], pos[1], pos[2]

        # Initialize arrays to store x and y coordinates of the trajectory
        x = np.zeros((rot_vel.shape[0], timesteps.shape[1]))
        y = np.zeros((rot_vel.shape[0], timesteps.shape[1]))
        theta = np.zeros((rot_vel.shape[0], timesteps.shape[1]))

        # Loop over each element of rot_vel
        for i in range(rot_vel.shape[0]):
            for j in range(timesteps.shape[1]):
                # Check if rot_vel is close to 0
                if abs(rot_vel[i]) >= 1:
                    temp_rot_vel = np.sign(rot_vel[i]) * 1
                else:
                    temp_rot_vel = rot_vel[i]

                if abs(temp_rot_vel) <= 0.20:
                    # Calculate x coordinates
                    x[i, j] = vel[i] * timesteps[0, j] * np.cos(theta0) + x0
                    # Calculate y coordinates
                    y[i, j] = vel[i] * timesteps[0, j] * np.sin(theta0) + y0
                    # Set theta values
                    theta[i, j] = theta0 * np.ones_like(temp_rot_vel * timesteps[0, j])
                else:
                    # Calculate x coordinates
                    x[i, j] = x0 + (vel[i] / temp_rot_vel) * (np.sin(temp_rot_vel * timesteps[0, j] + theta0) - np.sin(theta0))
                    # Calculate y coordinates
                    y[i, j] = y0 - (vel[i] / temp_rot_vel) * (np.cos(temp_rot_vel * timesteps[0, j] + theta0) - np.cos(theta0))
                    # Set theta values
                    theta[i, j] = theta0 + temp_rot_vel * timesteps[0, j]

        # Stack x, y, and theta arrays to form the trajectory array
        trajectory = np.stack((x, y, theta), axis=-1)

        # Return the generated trajectory
        return trajectory
    def update_pose(self):
        # Update numpy poses with current pose using the tf_buffer
        self.map_baselink_tf = self.tf_buffer.lookup_transform('map', 'base_link', rospy.Time(0)).transform
        self.pose_in_map_np[:] = [self.map_baselink_tf.translation.x, self.map_baselink_tf.translation.y,
                                  utils.euler_from_ros_quat(self.map_baselink_tf.rotation)[2]]
        self.pos_in_map_pix = (self.map_origin[:2] + self.pose_in_map_np[:2]) / self.map_resolution
        self.collision_marker.header.stamp = rospy.Time.now()
        self.collision_marker.pose = utils.pose_from_se2_pose(self.pose_in_map_np)
        self.collision_marker_pub.publish(self.collision_marker)

    def check_and_update_goal(self):
        # iterate the goal if necessary
        dist_from_goal = np.linalg.norm(self.pose_in_map_np[:2] - self.cur_goal[:2])
        #abs_angle_diff = np.abs(self.pose_in_map_np[2] - self.cur_goal[2])
        dangle = self.pose_in_map_np[2] - self.cur_goal[2]
        rot_dist_from_goal = np.abs(np.arctan2(np.sin(dangle), np.cos(dangle)))
        #rot_dist_from_goal = min(np.pi * 2 - abs_angle_diff, abs_angle_diff)
        if dist_from_goal < TRANS_GOAL_TOL and rot_dist_from_goal < ROT_GOAL_TOL:
            rospy.loginfo("Goal {goal} at {pose} complete.".format(
                    goal=self.cur_path_index, pose=self.cur_goal))
            if self.cur_path_index == len(self.path_tuples) - 1:
                rospy.loginfo("Full path complete in {time}s! Path Follower node shutting down.".format(
                    time=(rospy.Time.now() - self.path_follow_start_time).to_sec()))
                rospy.signal_shutdown("Full path complete! Path Follower node shutting down.")
            else:
                self.cur_path_index += 1
                self.cur_goal = np.array(self.path_tuples[self.cur_path_index])
        else:
            rospy.logdebug("Goal {goal} at {pose}, trans error: {t_err}, rot error: {r_err}.".format(
                goal=self.cur_path_index, pose=self.cur_goal, t_err=dist_from_goal, r_err=rot_dist_from_goal
            ))

    def stop_robot_on_shutdown(self):
        self.cmd_pub.publish(Twist())
        rospy.loginfo("Published zero vel on shutdown.")


if __name__ == '__main__':
    try:
        rospy.init_node('path_follower', log_level=rospy.DEBUG)
        pf = PathFollower()
    except rospy.ROSInterruptException:
        pass