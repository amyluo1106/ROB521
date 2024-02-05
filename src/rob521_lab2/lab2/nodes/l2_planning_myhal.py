#!/usr/bin/env python3
# Standard Libraries
import numpy as np
import yaml
import pygame
import time
import pygame_utils
import matplotlib.image as mpimg
from skimage.draw import circle_perimeter
from skimage.draw import disk
from scipy.linalg import block_diag
from scipy.spatial.distance import euclidean
import math
import scipy.spatial as sp
import os

np.random.seed(66)

def load_map(filename):
    im = mpimg.imread("../maps/" + filename)
    if len(im.shape) > 2:
        im = im[:, :, 0]
    im_np = np.array(im)  # Whitespace is true, black is false
    # im_np = np.logical_not(im_np)
    return im_np


def load_map_yaml(filename):
    with open("../maps/" + filename, "r") as stream:
        map_settings_dict = yaml.safe_load(stream)
    return map_settings_dict

# Node for building a graph


class Node:
    def __init__(self, point, parent_id, cost, tag):
        self.point = point  # A 3 by 1 vector [x, y, theta]
        # The parent node id that leads to this node (There should only every be one parent in RRT)
        self.parent_id = parent_id
        self.cost = cost  # The cost to come to this node
        self.children_ids = []  # The children node ids of this node
        self.tag = tag
        return

# Path Planner


class PathPlanner:
    # A path planner capable of perfomring RRT and RRT*
    def __init__(self, map_filename, map_setings_filename, goal_point, stopping_dist):
        # Get map information
        self.occupancy_map = load_map(map_filename)
        self.map_shape = self.occupancy_map.shape
        self.map_settings_dict = load_map_yaml(map_setings_filename)

        # Get the metric bounds of the map
        self.bounds = np.zeros([2, 2])  # m
        self.bounds[0, 0] = self.map_settings_dict["origin"][0]
        self.bounds[1, 0] = self.map_settings_dict["origin"][1]
        self.bounds[0, 1] = self.map_settings_dict["origin"][0] + \
            self.map_shape[1] * self.map_settings_dict["resolution"]
        self.bounds[1, 1] = self.map_settings_dict["origin"][1] + \
            self.map_shape[0] * self.map_settings_dict["resolution"]

        # Robot information
        self.robot_radius = 0.22  # m
        self.vel_max = 0.25 # 0.5  # m/s (Feel free to change!)
        self.rot_vel_max = 0.2 # 0.2  # rad/s (Feel free to change!)

        # Goal Parameters
        self.goal_point = goal_point  # m
        self.stopping_dist = stopping_dist  # m
        self.flag = 0

        # Trajectory Simulation Parameters
        self.timestep = 1.0  # s
        self.num_substeps = 10

        # Planning storage
        self.nodes = [Node(np.zeros((3, 1)), -1, 0, 0)]
        self.initial_pts = np.zeros((3, 1))
        self.nodes2 = self.initial_pts[:2][None]

        # RRT* Specific Parameters
        self.lebesgue_free = np.sum(
            self.occupancy_map) * self.map_settings_dict["resolution"] ** 2
        self.zeta_d = np.pi
        self.gamma_RRT_star = 2 * (1 + 1/2) ** (1/2) * \
            (self.lebesgue_free / self.zeta_d) ** (1/2)
        self.gamma_RRT = self.gamma_RRT_star + .1
        self.epsilon = 2.5

        # Pygame window for visualization
        # self.window = pygame_utils.PygameWindow(
        #     "Path Planner", (1000, 1000), self.occupancy_map.shape, self.map_settings_dict, self.goal_point, self.stopping_dist)

        self.window = pygame_utils.PygameWindow(
                "Path Planner", (795, 245), self.occupancy_map.T.shape, self.map_settings_dict, self.goal_point, self.stopping_dist)
        
        return

    # Functions required for RRT

    def sample_map_space(self):
        # Return an [x,y] coordinate to drive the robot towards
        # print("TO DO: Sample point to drive towards")

        sample_goal = np.random.rand() < 0.1
        # better_bounds = np.array([[0.0, 44], [-47, 11]])
        better_bounds = self.bounds
        # far_bounds = np.array([[38, 39.5], [-45.8, 43.5]])
        # near_bounds = np.array([[39.5, 42.5], [-45.8, 43.5]])

        # if not sample_goal:
            # x = np.clip(np.random.rand() * (better_bounds[0, 1] - better_bounds[0, 0]
            #                                 ) + better_bounds[0, 0], better_bounds[0, 0], better_bounds[0, 1])
            # y = np.clip(np.random.rand() * (better_bounds[1, 1] - better_bounds[1, 0]
            #                                 ) + better_bounds[1, 0], better_bounds[1, 0], better_bounds[1, 1])
            # point = np.array([[x], [y]])
        # else:
        #     sample_goal_far = np.random.rand() < 0.35
        #     if not sample_goal_far:
        #         x = np.clip(np.random.rand() * (far_bounds[0, 1] - far_bounds[0, 0]
        #                                     ) + far_bounds[0, 0], far_bounds[0, 0], far_bounds[0, 1])
        #         y = np.clip(np.random.rand() * (far_bounds[1, 1] - far_bounds[1, 0]
        #                                     ) + far_bounds[1, 0], far_bounds[1, 0], far_bounds[1, 1])
        #         point = np.array([[x], [y]])
        #     else:
        #         x = np.clip(np.random.rand() * (near_bounds[0, 1] - near_bounds[0, 0]
        #                                     ) + near_bounds[0, 0], near_bounds[0, 0], near_bounds[0, 1])
        #         y = np.clip(np.random.rand() * (near_bounds[1, 1] - near_bounds[1, 0]
        #                                     ) + near_bounds[1, 0], near_bounds[1, 0], near_bounds[1, 1])
        #         point = np.array([[x], [y]])

        x = np.random.rand() * (better_bounds[0, 1] - better_bounds[0, 0]
                                        ) + better_bounds[0, 0]
        y = np.random.rand() * (better_bounds[1, 1] - better_bounds[1, 0]
                                        ) + better_bounds[1, 0]
        point = np.array([[x], [y]])

        return point

    def check_if_duplicate(self, point):
        # Check if point is a duplicate of an already existing node
        # print("TO DO: Check that nodes are not duplicates")

        threshold = 0.15
        for node in self.nodes:
            if np.linalg.norm(node.point.reshape(3, ) - point) <= threshold:
                return True
        return False

    def closest_node(self, point, n):
        # print(point)
        # Returns the index of the closest node

        kdtree = sp.cKDTree(np.stack([node.point[:2, :]
                            for node in self.nodes], axis=0).squeeze(-1))
        d, i = kdtree.query(point.T, k=n)

        i = np.array(i)
        if len(i.shape) > 1:
            i = list(filter(lambda a: a != len(self.nodes), i[0]))

            if len(i) < n:
                return i

        return i[:n]

    def check_collision(self, robot_traj):

        # r, c = self.points_to_robot_circle(robot_traj[:2])
        # fp = np.clip(np.moveaxis(
        #     np.array([r, c]), 0, 2), 0, np.array(self.map_shape) - 1)

        # return not np.all(self.occupancy_map[fp[..., 1], fp[..., 0]])

        traj_rr, traj_cc = self.points_to_robot_circle(
            robot_traj[0:2, :])  # center and radius of trajectory in occupacy map
        footprint = np.moveaxis(np.array([traj_rr, traj_cc]), 0, 2)
        temp_x = np.clip(footprint[..., 1], 0, self.map_shape[0] - 1)
        temp_y = np.clip(footprint[..., 0], 0, self.map_shape[1] - 1)

        return np.any(np.any(self.occupancy_map[temp_x, temp_y] == 0, axis=-1))

    def simulate_trajectory(self, node_i, point_s):
        # Simulates the non-holonomic motion of the robot.
        # This function drives the robot from node_i towards point_s. This function does has many solutions!
        # node_i is a 3 by 1 vector [x;y;theta] this can be used to construct the SE(2) matrix T_{OI} in course notation
        # point_s is the sampled point vector [x; y]
        # print("TO DO: Implment a method to simulate a trajectory given a sampled point")

        vel, rot_vel = self.robot_controller(node_i, point_s)
        # 10 for sim
        if abs(euclidean(np.ravel(self.goal_point), np.ravel(node_i[:2])) > 1):
            robot_traj = self.trajectory_rollout(
                vel, rot_vel, node_i[0], node_i[1], node_i[2])
        else:
            # confused?
            min_dist_goal = float("inf")
            robot_traj = None
            for i in range(1, 11):
                if i <= 5:
                    robot_traj = self.trajectory_rollout(
                        vel, rot_vel + (i * 0.025), node_i[0], node_i[1], node_i[2])
                else:
                    robot_traj = self.trajectory_rollout(
                        vel, rot_vel - ((i - 5) * 0.025), node_i[0], node_i[1], node_i[2])
                dist_to_goal = euclidean(
                    np.ravel(self.goal_point), np.ravel(robot_traj[:, -1])[:2])
                if robot_traj is None:
                    min_dist_goal = dist_to_goal
                    robot_traj = robot_traj
                else:
                    collision = self.check_collision(robot_traj)
                    duplicate = self.check_if_duplicate(robot_traj[:, -1])
                    if collision or not duplicate:
                        if dist_to_goal < min_dist_goal:
                            min_dist_goal = dist_to_goal
                            robot_traj = robot_traj
        return robot_traj

    def robot_controller(self, node_i, point_s):
        # This controller determines the velocities that will nominally move the robot from node i to node s
        # Max velocities should be enforced
        # print("TO DO: Implement a control scheme to drive you towards the sampled point")

        theta_d = np.arctan2(
            (point_s[1] - node_i[1]), (point_s[0] - node_i[0]))
        theta = node_i[2]
        heading_error = theta_d - theta
        # normalize
        heading_error = math.atan2(
            math.sin(heading_error), math.cos(heading_error))
        # tune parameter?
        rot_vel = -0.4 * np.tan(heading_error)

        # max rotational velocity constraint
        if rot_vel > self.rot_vel_max:
            rot_vel = self.rot_vel_max
        if rot_vel < -self.rot_vel_max:
            rot_vel = -self.rot_vel_max

        vel = self.vel_max / (abs(rot_vel) + 1)  # not sure
        # max velocity constraint
        if vel > self.vel_max:
            vel = self.vel_max
        if vel < -self.vel_max:
            vel = -self.vel_max

        # turning radius/maximum curvature constraint
        if rot_vel > vel / self.robot_radius:
            rot_vel = vel / self.robot_radius

        return vel, -rot_vel

    def trajectory_rollout(self, vel, rot_vel, x0, y0, theta0):
        # Given your chosen velocities determine the trajectory of the robot for your given timestep
        # The returned trajectory should be a series of points to check for collisions
        # print("TO DO: Implement a way to rollout the controls chosen")

        t = np.linspace(0, self.timestep, self.num_substeps)
        x0 = np.ones((1, self.num_substeps)) * x0
        y0 = np.ones((1, self.num_substeps)) * y0
        theta0 = np.ones((1, self.num_substeps)) * theta0
        if rot_vel == 0:
            x = vel * t * np.cos(theta0) + x0
            y = vel * t * np.sin(theta0) + y0
            theta = rot_vel * t
        else:
            # don't understand?
            x = (vel / rot_vel) * \
                (np.sin(rot_vel * t + theta0) - np.sin(theta0)) + x0
            y = -(vel / rot_vel) * \
                (np.cos(rot_vel * t + theta0) - np.cos(theta0)) + y0
            theta = (rot_vel * t + theta0) % (2 * math.pi)
        return np.vstack((x, y, theta))

    def point_to_cell(self, point):
        # Convert a series of [x,y] points in the map to the indices for the corresponding cell in the occupancy map
        # point is a 2 by N matrix of points of interest

        # print("TO DO: Implement a method to get the map cell the robot is currently occupying")

        new_point = point.copy()
        new_point[0] = (point[0] - self.bounds[0, 0]
                        ) // self.map_settings_dict["resolution"]
        new_point[1] = self.map_shape[0] - \
            (point[1] - self.bounds[1, 0]
             ) // self.map_settings_dict["resolution"]
        new_point = (np.floor(new_point)).astype(int)

        return new_point

    def points_to_robot_circle(self, points):
        # Convert a series of [x,y] points to robot map footprints for collision detection
        # Hint: The disk function is included to help you with this function

        # print("TO DO: Implement a method to get the pixel locations of the robot path")

        r_result = []
        c_result = []
        r_robot = np.floor(self.robot_radius //
                           self.map_settings_dict["resolution"]).astype(int)

        for point in np.transpose(points):
            occupied_cell = self.point_to_cell(point)

            r, c = disk((occupied_cell[0], occupied_cell[1]), r_robot)

            r_result.append(r)
            c_result.append(c)

        return np.array(r_result), np.array(c_result)
    # Note: If you have correctly completed all previous functions, then you should be able to create a working RRT function

    # RRT* specific functions
    def ball_radius(self):
        # Close neighbor distance
        card_V = len(self.nodes)
        return min((self.gamma_RRT * (np.log(card_V) / card_V) ** (1.0/2.0))/2, self.epsilon/2)

    def connect_node_to_point(self, node_i, point_f):
        # Given two nodes find the non-holonomic path that connects them
        # Settings
        # node is a 3 by 1 node
        # point is a 2 by 1 point
        # line 1 through start tangent to theta
        # line 2 normal to line 1
        # line 3 through start and final
        # line 4 through midpoint prependicular to line 3

        start = node_i.reshape((3,))[0:2]
        theta_d = np.arctan2((point_f[1] - node_i[1]), (point_f[0] - node_i[0]))
        theta = node_i.reshape((3,))[2]
        heading_error = theta_d - theta

        final = point_f.reshape((2,))
        dist = np.linalg.norm(start - final)
        string_mid_point = (final - start) / 2 + start
        s1 = np.tan(theta)
        if s1 == 0:
            s2 = 9998
        else:
            s2 = -1 / s1
        s3d = (final[0] - start[0]) + 0.001
        s3 = (final[1] - start[1]) / s3d
        if s3 == 0:
            s4 = 9999
        else:
            s4 = -1 / s3
        y1 = start[1] - s1 * start[0]
        y2 = start[1] - s2 * start[0]
        y3 = start[1] - s3 * start[0]
        y4 = string_mid_point[1] - s4 * string_mid_point[0]

        cx = (y4 - y2) / (s2 - s4)
        cy = cx * s2 + y2
        center = np.array([cx, cy])
        r = np.linalg.norm(start - center)

        c = start
        b = center
        a = final

        ba = a - b
        bc = c - b

        angle = -(np.arctan2(start[1] - center[1], start[0] - center[0]) - (
            np.arctan2(final[1] - center[1], final[0] - center[0])))
        st = np.sin(-theta)
        ct = np.cos(-theta)
        R = np.array([[ct, -st], [st, ct]])
        c_R = R @ center.reshape((2, 1)) + start.reshape((2, 1))
        if c_R[1, 0] < 0:
            if angle > 0:
                angle = -2 * math.pi + angle
            else:
                angle = angle

        else:
            if angle < 0:
                angle = 2 * math.pi + angle
            else:
                angle = angle

        length = abs(angle) * r
        ignore_vel_max = True
        if ignore_vel_max:
            num_step = self.num_substeps
            rot_vel = angle / self.timestep
            vel = length / self.timestep

        else:
            num_step = self.num_substeps
            rot_vel = angle / self.timestep
            vel = length / self.timestep

            if rot_vel > self.rot_vel_max:
                rot_vel = self.rot_vel_max
            if rot_vel < -self.rot_vel_max:
                rot_vel = -self.rot_vel_max
            if vel > self.vel_max:
                vel = self.vel_max
            if vel < -self.vel_max:
                vel = -self.vel_max

            if rot_vel > vel / self.robot_radius:
                rot_vel = vel / self.robot_radius

        traj = self.trajectory_rollout(vel, rot_vel, x0=start[0], y0=start[1], theta0=theta)

        return traj

    def cost_to_come(self, trajectory_o, node):
        # The cost to get to a node from lavalle
        # print("TO DO: Implement a cost to come metric")

        cost = 0.0
        # Path distance
        for i in range(1, len(trajectory_o[0])):
            x_d = trajectory_o[0, i] - trajectory_o[0, i - 1]
            y_d = trajectory_o[1, i] - trajectory_o[1, i - 1]

            theta_s = math.atan2(y_d, x_d)
            theta_d = theta_s - node[2]
            theta_d_norm = math.atan2(math.sin(theta_d), math.cos(theta_d))
            cost += np.sqrt(x_d ** 2 + y_d ** 2  )#+ (0.1* theta_d_norm)**2)

        return cost

        # dist = np.linalg.norm(trajectory_o[1:, :2] - trajectory_o[:-1, :2], axis = -1).sum()
        # return dist

    def update_children(self, node_id):
        # Given a node_id with a changed cost, update all connected nodes with the new cost
        # print("TO DO: Update the costs of connected nodes after rewiring.")

        curr_cost = self.nodes[node_id].cost

        for i, child in enumerate(self.nodes[node_id].children_ids):
            traj = self.connect_node_to_point(self.nodes[node_id].point,
                                              np.array(self.nodes[child].point[:2, -1].reshape((2, 1))))
            self.nodes[child].cost = self.cost_to_come(traj, self.nodes[child].point) + curr_cost

    # Planner Functions
    def rrt_planning(self):
        # This function performs RRT on the given map and robot
        # You do not need to demonstrate this function to the TAs, but it is left in for you to check your work
        counter = 0
        while True:
            # for i in range(100000): #Most likely need more iterations than this to complete the map!
            counter += 1
            # Sample map space
            point = self.sample_map_space()

            # Get the closest point
            closest_node_id = self.closest_node(point, 1)[0]

            # Simulate driving the robot towards the closest point
            trajectory_o = self.simulate_trajectory(
                self.nodes[closest_node_id].point, point)

            # Check for collisions
            # print("TO DO: Check for collisions and add safe points to list of nodes.")
            collision = self.check_collision(trajectory_o)
            duplicate = self.check_if_duplicate(trajectory_o[:, -1])

            # Check if goal has been reached
            # print("TO DO: Check if at goal point.")
            if not (collision or duplicate):
                # print("here")
                # Add node to list
                cost = 0
                self.nodes[closest_node_id].children_ids.append(self.nodes[-1].tag + 1)
                self.nodes.append(Node(np.array(trajectory_o[:, -1].reshape((3, 1))), closest_node_id, cost, self.nodes[-1].tag + 1))

                # visualize
                temp_pt = np.array(trajectory_o[0:2, :]).copy().T
                self.window.add_se2_pose(
                    np.array(trajectory_o[:, -1].reshape((3,))))

                for i in temp_pt:
                    self.window.add_point(i)

                coords = [trajectory_o[0, -1], trajectory_o[1, -1]]
                if abs(euclidean(np.ravel(self.goal_point), coords)) < self.stopping_dist:
                    print("rrt success")
                    print("iterations: ", counter)
                    return self.recover_path()
        return self.nodes

    def rrt_star_planning(self):
        # This function performs RRT* for the given map and robot
        counter = 0
        while True:  # Most likely need more iterations than this to complete the map!
            # Sample
            point = self.sample_map_space()

            # Closest Node
            closest_node_id = self.closest_node(point, 1)[0]

            # Simulate trajectory
            trajectory_o = self.simulate_trajectory(self.nodes[closest_node_id].point, point)

            # Check for Collision
            collision = self.check_collision(trajectory_o)
            duplicate = self.check_if_duplicate(trajectory_o[:, -1])

            if not (collision or duplicate):
                # Calculate cost
                if counter > 0:
                    cost = self.cost_to_come(trajectory_o, self.nodes[closest_node_id].point) + self.nodes[
                        closest_node_id].cost

                else:
                    cost = 0

                pt = np.array(trajectory_o[:2, -1].reshape((2, 1)))

                # add node
                self.nodes[closest_node_id].children_ids.append(self.nodes[-1].tag + 1)
                self.nodes.append(Node(np.array(trajectory_o[:, -1].reshape((3, 1))), closest_node_id, cost, self.nodes[-1].tag + 1))
                self.nodes[-1].traj = trajectory_o[:3, :].reshape((3, 10))
                N = 10
                R = 3.5
                if counter > N:
                    n_closest_nodes = self.closest_node(pt, N)
                    closest_nodes = []
                    for nodes in n_closest_nodes:
                        if np.linalg.norm(self.nodes[nodes].point[:2] - point) <= R:
                            closest_nodes.append(nodes)
                    cost_min = cost
                    cost_min_id = closest_node_id
                    for node in closest_nodes[1:]:  # ignore closest node
                        # Simulate trajectory from near nodes to pt
                        trajectory_node = self.connect_node_to_point(self.nodes[node].point, pt)

                        # Check for Collision
                        collision = self.check_collision(trajectory_node)
                        if not collision:
                            # Calculate cost
                            if counter > 0:
                                temp_cost = self.cost_to_come(trajectory_node, self.nodes[node].point) + self.nodes[node].cost
                            else:
                                temp_cost = 0

                            if temp_cost < cost_min:  # get min cost
                                cost_min = temp_cost
                                cost_min_id = node
                                trajectory_o = trajectory_node

                    # Rewire pt to another node
                    if cost_min_id != closest_node_id:
                        # remove parent connection
                        self.nodes[closest_node_id].children_ids.remove(self.nodes[-1].tag)

                        # add node to min cost node
                        self.nodes[cost_min_id].children_ids.append(self.nodes[-1].tag)
                        self.nodes[-1].cost = cost_min
                        self.nodes[-1].parent_id = cost_min_id
                        self.nodes[-1].traj = trajectory_o[:3, :].reshape((3, 10))
                        self.update_children(self.nodes[-1].tag)

                    # Rewire other nodes to pt
                    for node in closest_nodes:
                        # Skip if parent node
                        if self.nodes[node].tag == cost_min_id:
                            continue

                        # Simulate trajectory from pt to near nodes
                        trajectory_node = self.connect_node_to_point(self.nodes[-1].point, np.array(
                            self.nodes[node].point[:2, -1].reshape((2, 1))))

                        # Check for Collision
                        collision = self.check_collision(trajectory_node)
                        if not collision:
                            # Calculate cost
                            if counter > 0:
                                cost = self.cost_to_come(trajectory_node, self.nodes[-1].point) + self.nodes[-1].cost
                            else:
                                cost = 0
                            # New path has less cost. Rewire
                            if cost < self.nodes[node].cost:
                                # remove parent connection
                                self.nodes[self.nodes[node].parent_id].children_ids.remove(node)
                                trajectory_old = self.connect_node_to_point(
                                    self.nodes[self.nodes[node].parent_id].point,
                                    np.array(self.nodes[node].point[:2, -1].reshape((2, 1))))

                                # add node to pt
                                self.nodes[node].cost = cost
                                self.nodes[node].parent_id = self.nodes[-1].tag
                                self.nodes[node].traj = trajectory_node[:3, :].reshape((3, 10))

                                # updating children of pt
                                self.nodes[self.nodes[-1].tag].children_ids.append(node)
                                self.update_children(node)

                                temp_pt = np.array(trajectory_node[0:2, :]).copy().T
                                temp_pt_old = np.array(trajectory_old[0:2, :]).copy().T
                                # self.window.add_se2_pose(np.array(trajectory_node[:, -1].reshape((3,))))
                                # for i in temp_pt:
                                #     self.window.add_point(i)

                # visualize
                temp_pt = np.array(trajectory_o[0:2, :]).copy().T
                self.window.add_se2_pose(np.array(trajectory_o[:, -1].reshape((3,))))
                for i in temp_pt:
                    self.window.add_point(i)

                # Check if goal has been reached
                coords = [trajectory_o[0, -1], trajectory_o[1, -1]]
                if abs(euclidean(np.ravel(self.goal_point), coords)) < self.stopping_dist:
                    print("rrt star success")
                    print("iterations: ", counter)
                    return self.recover_path()
                counter += 1

    def recover_path(self, node_id=-1):
        path = [self.nodes[node_id].point]
        current_node_id = self.nodes[node_id].parent_id
        while current_node_id > -1:
            path.append(self.nodes[current_node_id].point)
            current_node_id = self.nodes[current_node_id].parent_id
        path.reverse()
        return path


def main():
    # Set map information
    # map_filename = "willowgarageworld_05res.png"
    map_filename = "myhal.png"
    # map_setings_filename = "willowgarageworld_05res.yaml"
    map_setings_filename = "myhal.yaml"

    # robot information
    # goal_point = np.array([[42], [-44]]) #m
    goal_point = np.array([[7], [0]])  # m
    stopping_dist = 0.1  # m

    # RRT precursor
    path_planner = PathPlanner(map_filename, map_setings_filename, goal_point, stopping_dist)
    nodes = path_planner.rrt_star_planning()
    # nodes = path_planner.rrt_planning()
    node_path_metric = np.hstack(path_planner.recover_path())

    # Leftover test functions
    np.save("shortest_path.npy", node_path_metric)

    # Plot path
    path = "./shortest_path.npy"
    filename = os.path.splitext(os.path.split(path)[-1])[0]
    path = np.load(path).T[:, :, None]
    for p1, p2 in zip(path[:-1], path[1:]):
        path_planner.window.add_line(
            p1[:2, 0].copy(), p2[:2, 0].copy(), width=5, color=(255, 0, 0))

    pygame.image.save(path_planner.window.screen, f"{filename}.png")
    input()


if __name__ == '__main__':
    main()
