import numpy as np
from skimage.draw import polygon
import cv2

### codes for grid
# 100 is unexplored/unknown
# 150 is known floor
# 0 is obstacle
# 255 is explored

class OccupancyGrid():
    """
    Occupancy grid object. Assumes robot has a "classifier" object, which
    you would set up in the FSM using this.

    Uses all the patches in the bottom half of the image.
    """
    def __init__(self, robot, patch_size, grid_size=3000):
        self.robot = robot
        self.patch_size = patch_size
        self.grid_size = grid_size
        self.grid = np.full((grid_size, grid_size), 100, dtype=np.uint8)
        self.id = robot.pose.origin_id

    def process_patch(self, image):
        """
        Adds the information from an image to the map. Assumes the image
        was just taken (so robot and camera location/orientation are the
        same). Uses the whole bottom half of the image, instead of just
        the center patch.

        If an obstacle is detected, the patches above it are not added to
        the grid (since their depths calculated from project_to_ground will
        be off).
        """
        # loop over all the patches in the bottom half
        # extracting them, classifying them, and handling the grid logic
        camera_center = (120, 160)
        h_patch_num = (camera_center[0] - self.patch_size[0]//2) // \
                self.patch_size[0]
        w_patch_num = (camera_center[1] - self.patch_size[1]//2) // \
                self.patch_size[1]

        for j in range(-w_patch_num, w_patch_num+1):
            # loop over columns first, so the column loop can break on obstacles

            for i in range(h_patch_num, -1, -1):
                # patch extraction
                x1 = (camera_center[0] - self.patch_size[0] // 2) + \
                        self.patch_size[0] * i
                y1 = (camera_center[1] - self.patch_size[1] // 2) + \
                        self.patch_size[1] * j

                x2 = x1 + self.patch_size[0]
                y2 = y1 + self.patch_size[1]

                patch = cv2.cvtColor(image[x1:x2, y1:y2, :], cv2.COLOR_RGB2BGR)

                # mapping patch to grid coordinates
                # flip x and y, because that's what project_to_ground expects
                patch_points = [(y1, x1), (y2, x1), (y2, x2), (y1, x2)]
                ground_points = [self.robot.kine.project_to_ground(*point)
                        for point in patch_points]
                world_points = [self.robot.kine.base_to_link('world').dot(point)
                        for point in ground_points]
                grid_points = [(int(p[0]) + self.grid_size // 2, int(p[1])
                    + self.grid_size // 2) for p in world_points]

                # need to draw a polygon, since image squares map to
                # grid trapezoids
                grid_rows, grid_cols = polygon([p[0] for p in grid_points],
                        [p[1] for p in grid_points], shape=self.grid.shape)

                # now, classify the patch
                if not self.robot.classifier(patch):
                    # it's an obstacle
                    # write in regardless if Cozmo's been there
                    # since the obstacle could have been added
                    self.grid[grid_rows, grid_cols] = 0

                    # then break the loop, because further-up patches
                    # might not be flat on the ground
                    break
                else:
                    # not an obstacle
                    # leave visited if Cozmo's already been there
                    grid_patch = self.grid[grid_rows, grid_cols]
                    if len(grid_patch) > 0 and np.max(grid_patch) != 255:
                        # no part of the patch was visited, can fill in
                        self.grid[grid_rows, grid_cols] = 150


    def update_location(self):
        """
        Update the occupancy grid to mark the robot's current location as
        explored. Also reset the whole thing if the robot is picked up and
        potentially moved.
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
        Display the occupancy grid. Should be called in user_image in an
        FSM, skipping frames only if computationally necessary.
        """
        cv2.waitKey(1)
        cv2.imshow('Occupancy Grid', self.grid)
