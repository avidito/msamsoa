"""
Solution Base Class

Base class for solution space, algorithm and agent (UAV) definition

Include:
    - Solution (class): Base class for problem definition
    - Agent (class): Base class for agent entity
"""

import numpy as np
import logging


class Solution:
    """
    Solution: Solution Base Class

    Defining problem space and general information about mission.

    Init Params:
    - space: numpy.array; Problem space in matrix form, consist of value 0 (for fertilized space) and 1 (for unfertilized space)
    """
    ##### Initialization Methods #####
    def __init__(self, space):
        logging.info("Initialize Solution Space")
        self.space = space
        self.boundary = space.shape[0]
        self.size = space.shape[0] ** 2
        self.target_cnt = self.size - sum(sum(space))
        self.name = "__BASE__"

    ##### Utilities Methods #####
    @staticmethod
    def check_boundary(x, y, boundary):
        """
        Check whether a coordinate is inside or outside boundary.
        """
        return (x >= 0 and x < boundary) and (y >= 0 and y < boundary)


class Agent:
    """
    Agent: Solution Base Agent

    Defining basic agent for solution's agents blueprint.

    Init Params:
    - space: numpy.array; Problem space in matrix form, consist of value 0 (for fertilized space) and 1 (for unfertilized space)
    """
    ##### Initialization Methods #####
    def __init__(self, idx, boundary):
        self.id = idx
        self.boundary = boundary

        self.position = None
        self.direction = None
        self.power = np.inf
        self.move_direction = []
        self.radar_range = 2

    ##### Navigation #####
    def get_available_grid(self, occupied_field):
        available_grid = []
        for choice in self.move_direction:
            candidate_x = self.position[0] + choice[0]
            candidate_y = self.position[1] + choice[1]
            if (
                (Solution.check_boundary(candidate_x, candidate_y, self.boundary)) and
                (occupied_field[candidate_x, candidate_y] == False)
            ):
                available_grid.append((candidate_x, candidate_y))
        return available_grid



    #     def availableGrid(self, amap, target_direction):
    #         nw, nl = self.boundary
    #         y, x = self.position
    #         choice = [(1,0), (1,1), (0,1), (-1,1), (-1,0), (-1,-1), (0,-1), (1,-1)]
    #         if(self.hungry):
    #             for i in range(len(choice)):
    #                 if(choice[i] == target_direction):
    #                     choice = [choice[i], choice[(i+1)%8], choice[i-1]]
    #                     break
    #
    #         extra = [(2*ny, 2*nx) for (ny,nx) in choice if ((ny == 0 or nx == 0) and ny ^ nx)]
    #         for e in extra:
    #             choice.append(e)
    #
    #         grida = []
    #         for (dy,dx) in choice:
    #             ny = y + dy
    #             nx = x + dx
    #             if(ny >= 0 and ny <= nw-1 and nx >= 0 and nx <= nl-1 and amap[ny, nx]):
    #                 grida.append([ny, nx])
    #         return grida.copy()
