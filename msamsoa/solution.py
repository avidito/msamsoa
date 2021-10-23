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
    def __init__(self, space):
        logging.info("Initialize Solution Space")
        self.space = space
        self.boundary = space.shape[0]
        self.size = space.shape[0] ** 2
        self.target_cnt = self.size - sum(sum(space))
        self.name = "__BASE__"

class Agent:
    """
    Agent: Solution Base Agent

    Defining basic agent for solution's agents blueprint.

    Init Params:
    - space: numpy.array; Problem space in matrix form, consist of value 0 (for fertilized space) and 1 (for unfertilized space)
    """

    def __init__(self, idx, boundary):
        self.id = idx
        self.boundary = boundary

        self.position = None
        self.direction = None
        self.power = np.inf
        self.move_direction = [(1,0), (1,1), (0,1), (-1,1), (-1,0), (-1,-1), (0,-1), (1,-1)]
