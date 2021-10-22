"""
Solution Base Class

Base class for solution space, algorithm and agent (UAV) definition

Include:
    - Solution (class): Base class for problem definition
    - Agent (class): Base class for agent entity
"""

class Solution:
    """
    Solution Base Class

    Defining problem space and general information about mission.
    """
    def __init__(self, space):
        self.space = space
        self.boundary = space.shape
        self.size = (space.shape[0] * space.shape[1])
        self.target_cnt = self.size - sum(sum(space))
        self.name = "__BASE__"
