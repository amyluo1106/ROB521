#!/usr/bin/env python3
#Standard Libraries
import numpy as np
import yaml
import pygame
import time
import pygame_utils
import matplotlib.image as mpimg
from skimage.draw import disk
from scipy.linalg import block_diag
import math
import scipy

def load_map(filename):
    im = mpimg.imread("../maps/" + filename)
    if len(im.shape) > 2:
        im = im[:,:,0]
    im_np = np.array(im)  #Whitespace is true, black is false
    #im_np = np.logical_not(im_np)    
    return im_np


def load_map_yaml(filename):
    with open("../maps/" + filename, "r") as stream:
            map_settings_dict = yaml.safe_load(stream)
    return map_settings_dict

#Node for building a graph
class Node:
    def __init__(self, point, parent_id, cost):
        self.point = point # A 3 by 1 vector [x, y, theta]
        self.parent_id = parent_id # The parent node id that leads to this node (There should only every be one parent in RRT)
        self.cost = cost # The cost to come to this node
        self.children_ids = [] # The children node ids of this node
        return

#Path Planner 
class PathPlanner:
    #A path planner capable of perfomring RRT and RRT*
    def __init__(self, map_filename, map_setings_filename, goal_point, stopping_dist):
        #Get map information
        self.occupancy_map = load_map(map_filename)
        self.map_shape = self.occupancy_map.shape
        self.map_settings_dict = load_map_yaml(map_setings_filename)

        #Get the metric bounds of the map
        self.bounds = np.zeros([2,2]) #m
        self.bounds[0, 0] = self.map_settings_dict["origin"][0]
        self.bounds[1, 0] = self.map_settings_dict["origin"][1]
        self.bounds[0, 1] = self.map_settings_dict["origin"][0] + self.map_shape[1] * self.map_settings_dict["resolution"]
        self.bounds[1, 1] = self.map_settings_dict["origin"][1] + self.map_shape[0] * self.map_settings_dict["resolution"]

        #Robot information
        self.robot_radius = 0.22 #m
        self.vel_max = 0.5 #m/s (Feel free to change!)
        self.rot_vel_max = 0.2 #rad/s (Feel free to change!)

        #Goal Parameters
        self.goal_point = goal_point #m
        self.stopping_dist = stopping_dist #m

        #Trajectory Simulation Parameters
        self.timestep = 1.0 #s
        self.num_substeps = 10

        #Planning storage
        self.nodes = [Node(np.zeros((3,1)), -1, 0)]

        #RRT* Specific Parameters
        self.lebesgue_free = np.sum(self.occupancy_map) * self.map_settings_dict["resolution"] **2
        self.zeta_d = np.pi
        self.gamma_RRT_star = 2 * (1 + 1/2) ** (1/2) * (self.lebesgue_free / self.zeta_d) ** (1/2)
        self.gamma_RRT = self.gamma_RRT_star + .1
        self.epsilon = 2.5
        
        #Pygame window for visualization
        self.window = pygame_utils.PygameWindow(
            "Path Planner", (1000, 1000), self.occupancy_map.shape, self.map_settings_dict, self.goal_point, self.stopping_dist)
        return

    #Functions required for RRT
    def sample_map_space(self):
        #Return an [x,y] coordinate to drive the robot towards
        print("TO DO: Sample point to drive towards")

        sample_goal = np.random.rand() < 0.05
        if not sample_goal:
            x = np.random.rand() * (self.bounds[0, 1] - self.bounds[0, 0])  + self.bounds[0, 0] 
            y = np.random.rand() * (self.bounds[1, 1] - self.bounds[1, 0])  + self.bounds[1, 0] 
            theta = np.random.rand() * 2 * np.pi  - np.pi # [pi, -pi]
            point = np.array([[x], [y], [theta]])
        else:
            theta = np.random.rand() * 2 * np.pi  - np.pi # [pi, -pi]
            dx = 4 * self.stopping_dist * np.random.randn()
            dy = 4 * self.stopping_dist * np.random.randn()
            point = np.array([[self.goal_point[0, 0] +dx], [self.goal_point[1, 0] + dy], [theta]])
        return point
    
    def check_if_duplicate(self, point):
        #Check if point is a duplicate of an already existing node
        #print("TO DO: Check that nodes are not duplicates")
        
        duplicate_range = 0.1
        for node in self.nodes:
            if np.linalg.norm(node.point.reshape(3, ) - point) <= duplicate_range:
                return True
        return False
    
    def closest_node(self, point, k=1):
        #Returns the index of the closest node
        #print("TO DO: Implement a method to get the closest node to a sapled point")
        
        kdtree = scipy.spatial.cKDTree(self.node_pts[:, :, 0])
        d, i = kdtree.query(point[:2, 0][None], k = k)

        return i
    
    def check_collision(self, robot_traj):
        occup = self.points_to_robot_circle(robot_traj)
        rows = occup[..., 1]
        cols = occup[..., 0]
        collision = np.any(self.occupancy_map[rows, cols] == 0, axis = -1)
        return collision
    
    def simulate_trajectory(self, node_i, point_s):
        #Simulates the non-holonomic motion of the robot.
        #This function drives the robot from node_i towards point_s. This function does has many solutions!
        #node_i is a 3 by 1 vector [x;y;theta] this can be used to construct the SE(2) matrix T_{OI} in course notation
        #point_s is the sampled point vector [x; y]
        #print("TO DO: Implment a method to simulate a trajectory given a sampled point")

        vel, rot_vel = self.robot_controller(node_i, point_s)
        if abs(np.linalg.norm(self.goal_point - node_i) > 1):  # 10 for sim
            robot_traj = self.trajectory_rollout(vel, rot_vel, node_i[0], node_i[1], node_i[2])
        else:
            # confused?
            min_dist_goal = float("inf")
            robot_traj = None
            for i in range(1, 11):
                if i <= 5:
                    robot_traj = self.trajectory_rollout(vel, rot_vel + (i * 0.025), node_i[0], node_i[1], node_i[2])
                else:
                    robot_traj = self.trajectory_rollout(vel, rot_vel - ((i - 5) * 0.025), node_i[0], node_i[1], node_i[2])
                dist_to_goal = np.linalg.norm(self.goal_point - robot_traj[:, -1])
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
        #This controller determines the velocities that will nominally move the robot from node i to node s
        #Max velocities should be enforced
        #print("TO DO: Implement a control scheme to drive you towards the sampled point")

        theta_d = np.arctan2((point_s[1] - node_i[1]), (point_s[0] - node_i[0]))
        theta = node_i[2]
        heading_error = theta_d - theta
        # normalize
        heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))
        # tune parameter?
        rot_vel = -0.1 * np.tan(heading_error)

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
        #print("TO DO: Implement a way to rollout the controls chosen")

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
            x = (vel / rot_vel) * (np.sin(rot_vel * t + theta0) - np.sin(theta0)) + x0
            y = -(vel / rot_vel) * (np.cos(rot_vel * t + theta0) - np.cos(theta0)) + y0
            theta = (rot_vel * t + theta0) % (2 * math.pi)
        return np.vstack((x, y, theta))
    
    def point_to_cell(self, point):
        #Convert a series of [x,y] points in the map to the indices for the corresponding cell in the occupancy map
        #point is a 2 by N matrix of points of interest
        #print("TO DO: Implement a method to get the map cell the robot is currently occupying")
        
        cells = []
        map_x = self.map_settings_dict["origin"][0]
        map_y = self.map_settings_dict["origin"][1]
        res = self.map_settings_dict["resolution"]
        # map height in meters 
        map_height = self.map_shape[1] * res    
        for i in range(len(point)):
            x = int((point[i][0] - map_x)/res)
            # point is wrt bottom left, while map_y is wrt top left
            y = int((map_height - (point[i][1] - map_y))/res)
            cells.append([x, y])
        return cells

    def points_to_robot_circle(self, points):
        #Convert a series of [x,y] points to robot map footprints for collision detection
        #Hint: The disk function is included to help you with this function
        #print("TO DO: Implement a method to get the pixel locations of the robot path")
        
        cells = self.point_to_cell(points)
        rr = []
        cc = []
        for cell in cells:
            r, c = disk(cell, int(self.robot_radius / self.map_settings_dict["resolution"]))
            rr.apend(r)
            cc.apend(c)
        return rr, cc
    #Note: If you have correctly completed all previous functions, then you should be able to create a working RRT function

    #RRT* specific functions
    def ball_radius(self):
        #Close neighbor distance
        card_V = len(self.nodes)
        return min(self.gamma_RRT * (np.log(card_V) / card_V ) ** (1.0/2.0), self.epsilon)
    
    def connect_node_to_point(self, node_i, point_f):
        #Given two nodes find the non-holonomic path that connects them
        #Settings
        #node is a 3 by 1 node
        #point is a 2 by 1 point
        print("TO DO: Implement a way to connect two already existing nodes (for rewiring).")
        return np.zeros((3, self.num_substeps))
    
    def cost_to_come(self, trajectory_o):
        #The cost to get to a node from lavalle 
        print("TO DO: Implement a cost to come metric")
        return 0
    
    def update_children(self, node_id):
        #Given a node_id with a changed cost, update all connected nodes with the new cost
        print("TO DO: Update the costs of connected nodes after rewiring.")
        return

    #Planner Functions
    def rrt_planning(self):
        #This function performs RRT on the given map and robot
        #You do not need to demonstrate this function to the TAs, but it is left in for you to check your work
        for i in range(50000): #Most likely need more iterations than this to complete the map!
            #Sample map space
            point = self.sample_map_space()

            #Get the closest point
            closest_node_id = self.closest_node(point)

            #Simulate driving the robot towards the closest point
            trajectory_o = self.simulate_trajectory(self.nodes[closest_node_id].point, point)

            #Check for collisions
            #print("TO DO: Check for collisions and add safe points to list of nodes.")
            collision = self.check_colision(trajectory_o)
            duplicate = self.check_if_duplicate(trajectory_o[:, -1])
            
            #Check if goal has been reached
            #print("TO DO: Check if at goal point.")
            if not (collision or duplicate):
                # Add node to list
                cost = 0
                self.nodes[closest_node_id].children_ids.append(self.nodes[-1].id + 1)
                self.nodes.append(Node(np.array(trajectory_o[:, -1].reshape((3, 1))), closest_node_id, cost, self.nodes[-1].id + 1))

                # visualize
                temp_pt = np.array(trajectory_o[0:2, :]).copy().T
                self.window.add_se2_pose(np.array(trajectory_o[:, -1].reshape((3,))))
                self.vis(temp_pt)

                [x, y] = [trajectory_o[0, -1], trajectory_o[1, -1]]
                if np.norm(self.goal_point[:2] - (x, y)) < self.stopping_dist:
                    print("rrt success")
                    return self.recover_path()
        return self.nodes
    
    def rrt_star_planning(self):
        #This function performs RRT* for the given map and robot        
        for i in range(1): #Most likely need more iterations than this to complete the map!
            #Sample
            point = self.sample_map_space()

            #Closest Node
            closest_node_id = self.closest_node(point)

            #Simulate trajectory
            trajectory_o = self.simulate_trajectory(self.nodes[closest_node_id].point, point)

            #Check for Collision
            print("TO DO: Check for collision.")

            #Last node rewire
            print("TO DO: Last node rewiring")

            #Close node rewire
            print("TO DO: Near point rewiring")

            #Check for early end
            print("TO DO: Check for early end")
        return self.nodes
    
    def recover_path(self, node_id = -1):
        path = [self.nodes[node_id].point]
        current_node_id = self.nodes[node_id].parent_id
        while current_node_id > -1:
            path.append(self.nodes[current_node_id].point)
            current_node_id = self.nodes[current_node_id].parent_id
        path.reverse()
        return path

def main():
    #Set map information
    map_filename = "willowgarageworld_05res.png"
    map_setings_filename = "willowgarageworld_05res.yaml"

    #robot information
    goal_point = np.array([[10], [10]]) #m
    stopping_dist = 0.5 #m

    #RRT precursor
    path_planner = PathPlanner(map_filename, map_setings_filename, goal_point, stopping_dist)
    nodes = path_planner.rrt_star_planning()
    node_path_metric = np.hstack(path_planner.recover_path())

    #Leftover test functions
    np.save("shortest_path.npy", node_path_metric)


if __name__ == '__main__':
    main()
