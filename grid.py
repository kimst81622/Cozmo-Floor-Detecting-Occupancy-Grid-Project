import numpy as np
import cv2

### codes for grid
# 100 is unexplored/unknown
# 150 is known floor
# 0 is obstacle
# 255 is explored

class OccupancyGrid():
    """
    Occupancy grid object. Assumes robot has a "classifier" object.
    """
    def __init__(self, robot, patch_size, grid_size=2000):
        self.robot = robot
        self.patch_size = patch_size
        self.grid_size = grid_size
        self.grid = np.full((grid_size, grid_size), 100, dtype=np.uint8)
        self.id = robot.pose.origin_id

    def process_patch(self, patch):
        """
        Adds the information from a patch to the map. Assumes the patch
        was just taken (so robot and camera location/orientation are the
        same).
        """
        # might break for odd patch sizes
        # also, only classifies the middle 1/4 of the patch
        # because the edges are less certain
        # commented out for now cause there are issues with mapping
        camera_center = (160, 120)
        #patch_top_left = (camera_center[0] - self.patch_size[0] // 4,
                #camera_center[1] - self.patch_size[1] // 4)
        #patch_top_right = (camera_center[0] + self.patch_size[0] // 4,
                #camera_center[1] - self.patch_size[1] // 4)
        #patch_bot_left = (camera_center[0] - self.patch_size[0] // 4,
                #camera_center[1] + self.patch_size[1] // 4)
        #patch_bot_right = (camera_center[0] + self.patch_size[0] // 4,
                #camera_center[1] + self.patch_size[1] // 4)

        #top_left_point = self.robot.kine.project_to_ground(*patch_top_left)
        #top_right_point = self.robot.kine.project_to_ground(*patch_top_right)
        #bot_right_point = self.robot.kine.project_to_ground(*patch_bot_right)
        #bot_left_point = self.robot.kine.project_to_ground(*patch_bot_left)
        center_point = self.robot.kine.project_to_ground(*camera_center)

        base_to_world = self.robot.kine.base_to_link('world')
        #world_top_left_point = base_to_world.dot(top_left_point)
        #world_top_right_point = base_to_world.dot(top_right_point)
        #world_bot_right_point = base_to_world.dot(bot_right_point)
        #world_bot_left_point = base_to_world.dot(bot_left_point)
        center_world = base_to_world.dot(center_point)

        #grid_top_x = int(world_top_left_point[0]) + self.grid_size // 2
        #grid_top_y = int(world_top_left_point[1]) + self.grid_size // 2
        #grid_bot_x = int(world_bot_right_point[0]) + self.grid_size // 2
        #grid_bot_y = int(world_bot_right_point[1]) + self.grid_size // 2
        grid_x = int(center_world[0]) + self.grid_size // 2
        grid_y = int(center_world[1]) + self.grid_size // 2

        if not self.robot.classifier(patch):
            # it's an obstacle
            #self.grid[grid_top_x:grid_bot_x, grid_top_y:grid_bot_y] = 0
            self.grid[grid_x-15:grid_x+15, grid_y-15:grid_y+15] = 0
        else:
            # pretty sure it's not
            #self.grid[grid_top_x:grid_bot_x, grid_top_y:grid_bot_y] = 150
            self.grid[grid_x-15:grid_x+15, grid_y-15:grid_y+15] = 150

    def update_location(self):
        """
        Update the occupancy grid to mark the robot's current location as
        explored.
        """
        # first, check to see if the origin_id has changed
        if self.robot.pose.origin_id != self.id:
            # could have been picked up, wipe the map
            self.grid = np.full((self.grid_size, self.grid_size), 127, dtype=np.uint8)
            self.id = self.robot.pose.origin_id

        # now, add Cozmo's location
        world_position_x = self.robot.pose.position.x
        world_position_y = self.robot.pose.position.y

        grid_position_x = int(world_position_x) + self.grid_size // 2
        grid_position_y = int(world_position_y) + self.grid_size // 2

        # he's roughly 30x30 mm
        self.grid[grid_position_x-15:grid_position_x+15,
                grid_position_y-15:grid_position_y+15] = 255


    def show(self):
        """
        Display the occupancy grid.
        """
        cv2.waitKey(1)
        cv2.imshow('Occupancy Grid', self.grid)
