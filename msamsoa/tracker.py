"""
Logging and Tracking Modules

Module to initialize and configure progress tracker and logger.

Include:
    - init_logger (function): Initialize basic configuration for logger
"""

import argparse
import logging
import os

levels = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "warn": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL
}

def init_logger():
    # Get arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l",
        "--log",
        default = "info",
        help = (
            "Setup logging level (default = 'info')"
            "Example:"
            "--log debug"
        )
    )
    options = parser.parse_args()
    level = levels.get(options.log.lower())

    # Setup logger
    logging.basicConfig(
        format = "[%(asctime)s] %(levelname)s - %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
        level = level
    )

class Tracker:
    """
    Tracker: Progress Information Handler

    Object to handle mission progress information tracking. Focus on tracking agent movement and completion rate. By default, all tracking information will be exported into result folder.

    Init Params:
    - track_dir: string (default = "track"); Path for track result directory. Automatically create directory if not exists.
    """

    def __init__(self, track_dir="track"):
        logging.debug("Initialize Tracker with: track_dir='%s'", track_dir)
        if (not os.path.exists(track_dir)):
            logging.debug("Creating result directory at: '%s'", track_dir)
            os.makedirs(track_dir)

        self.dir = track_dir
#
#     # Inisiasi Kelas
#     def __init__(self, size, nt):
#         self.size = size
#         self.nt = nt
#         self.ts = 0
#         self.tf = 0
#
#         self.vm_tracks = []
#         self.tm_tracks = []
#         self.agent_position = []
#         self.s_tracks = []
#         self.f_tracks = []
#
#     # Menambahkan Informasi Simulasi
#     def update(self, iteration, vmap, emap, ni, agents, simulation):
#         self.iteration = iteration
#
#         if(simulation):
#             self.vm_tracks.append(vmap.copy())
#             self.tm_tracks.append(emap.copy())
#             self.agent_position.append(agents.copy())
#         self.s_tracks.append(sum(sum(vmap)))
#         self.f_tracks.append(ni)
#
#         self.ts = self.iteration if self.s_tracks[-1] != self.size else self.ts
#         self.tf = self.iteration if self.f_tracks[-1] != self.nt else self.tf
#
#     # Mendapatkan Posisi dari Agen pada Iterasi
#     def getAgentLocations(self, i):
#         if(i == 0):
#             return ([],[]), ([],[])
#         else:
#             nloc = np.array([agent[0] for agent in self.agent_position[i] if agent[1] != 2]).transpose()
#             bloc = np.array([agent[0] for agent in self.agent_position[i] if agent[1] == 2]).transpose()
#             if(nloc.shape[0] == 0):
#                 nloc = np.array([[], []])
#             if(bloc.shape[0] == 0):
#                 bloc = np.array([[], []])
#             return (nloc, bloc)
