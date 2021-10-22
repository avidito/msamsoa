"""
Solution Base Class

Base class for solution space, algorithm and agent (UAV) definition

Include:
    - Solution (class): Base class for problem definition
    - Agent (class): Base class for agent entity
"""

import numpy as np


class Solution:
    """
    Solution Base Class

    Defining problem space and general information about mission.
    """
    def __init__(self, space):
        self.space = space
        self.boundary = space.shape[0]
        self.size = space.shape[0] ** 2
        self.target_cnt = self.size - sum(sum(space))
        self.name = "__BASE__"

class Agent:
    """
    Solution Base Agent

    Defining basic agent for solution's agents blueprint.
    """

    def __init__(self, idx, boundary):
        self.id = idx
        self.boundary = boundary

        self.position = None
        self.direction = None
        self.power = np.inf
        self.move_direction = [(1,0), (1,1), (0,1), (-1,1), (-1,0), (-1,-1), (0,-1), (1,-1)]
