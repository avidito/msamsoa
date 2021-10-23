"""
Logging and Tracking Modules

Module to initialize and configure progress tracker and logger.

Include:
    - init_logger (function): Initialize basic configuration for logger
"""

import argparse
import logging
import os
import csv

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
    - field_size: int; Total grid in space field.
    - track_dir: string (default = "track"); Path for track result directory. Automatically create directory if not exists.
    """
    ##### Initialization Methods #####
    def __init__(self, field_size, track_dir="track"):
        logging.debug("Initialize Tracker with: field_size=%d; track_dir='%s'", field_size, track_dir)
        if (not os.path.exists(track_dir)):
            logging.debug("Creating result directory at: '%s'", track_dir)
            os.makedirs(track_dir)

        self.dir = track_dir
        self.init_track_files(field_size)

    def init_track_files(self, field_size):
        """
        Initialize result file for each type of track information. All result will be saved as CSV. Field information will be saved as flatten matrix.
        """
        field_files = ["fertilized_field", "visited_field"]
        for filename in field_files:
            filepath = os.path.join(self.dir, f"{filename}.csv")
            with open(filepath, "w+", encoding="utf-8", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([f"g{i}" for i in range(field_size)])

        filepath = os.path.join(self.dir, "agents.csv")
        with open(filepath, "w+", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["iteration", "agent_id", "x", "y", "mission"])

    ##### Tracking Methods #####
    def add_snapshot(self, iteration, fertilized_field, visited_field, agents):
        """
        Add snapshot of algorithm progress
        """
        logging.debug(f"Add logging for iteration {iteration:4}")

        self.add_field_snapshot("fertilized_field", fertilized_field)
        self.add_field_snapshot("visited_field", visited_field)
        self.add_agents_snapshot("agents", iteration, agents)

    def add_field_snapshot(self, filename, field):
        """
        Add snapshot of field data.
        """
        logging.debug(f"Flatten '{filename}' data and save to '{filename}.csv'")
        flatten_field = field.flatten()
        filepath = os.path.join(self.dir, f"{filename}.csv")
        with open(filepath, "a", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(flatten_field)

    def add_agents_snapshot(self, filename, iteration, agents):
        """
        Add snapshot of agents data.
        """
        logging.debug(f"Parse agents information and save to '{filename}.csv'")
        agents_data = [
            {
                "iteration": iteration,
                "agent_id": agent.id,
                "x": agent.position[1],
                "y": agent.position[0],
                "mission": agent.mission if (agent.mission) else "inactive"
            } for agent in agents
        ]
        filepath = os.path.join(self.dir, f"{filename}.csv")
        with open(filepath, "a", encoding="utf-8", newline="") as file:
            fieldnames = ["iteration", "agent_id", "x", "y", "mission"]
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            for data in agents_data:
                writer.writerow(data)

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