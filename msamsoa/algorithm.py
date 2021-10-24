"""
MSAMSOA Solution Implementation

Solution implementation with MSAMSOA methods.

Include:
    - MSAMSOA (class): Modified Search-Attack Mission Self-Organized Algorithm (MSAMSOA) implementation in grid-based discrete problem space for crop field surveillance and fertilization. Targets are represented as 2D matrix with value 0 (fertilized) and 1 (unfertilized). Agents (UAV) represented as dot.
    - MSAMSOA_Agent (class): MSAMSOA agent, or UAV representation for MSAMSOA solution.
    - calculate_distance (function): Calculate distance between two grids.
"""

import numpy as np
from numpy import random

from msamsoa.solution import Solution, Agent
from msamsoa.tracker import init_logger, Tracker

import logging
init_logger()

class MSAMSOA(Solution):
    """
    MSAMSOA: Modified Search-Attack Mission Self-Organized Algorithm (MSAMSOA) implementation in grid-based discrete problem space for crop field surveillance and fertilization. Targets are represented as 2D matrix with value 0 (fertilized) and 1 (unfertilized). Agents (UAV) represented as dot.

    Init Params:
    - space: numpy.array; Problem space in matrix form, consist of value 0 (for fertilized space) and 1 (for unfertilized space)
    - agent_cnt: int; Total number of agents (UAV) for implementation.
    - a: float (default = 0.7); Pheromone value importance in navigation decision.
    - b: float (default = 0.3); Heuristic value importance in navigation decision.
    - d: int (default = 80);
    - diff_local_pheromone: float (default = 1.0); Base pheromone for local reduction.
    - diff_global_pheromone: float (default = 1.0); Base pheromone for global addition.
    - ht: int (default = 10); Maximum iteration of no new grid surveillance before agent enter hungry state.
    """

    ##### Initiation Methods #####
    def __init__(self, space, agents_cnt, a=0.7, b=0.3, d=80, diff_local_pheromone=1.0, diff_global_pheromone=1.0, ht=10):
        super().__init__(space)
        logging.info("Solution space update: Using MSAMSOA Algorithm")

        self.agents_cnt = agents_cnt
        agents_params = {
            "a": a,
            "b": b,
            "diff_local_pheromone": diff_local_pheromone,
            "diff_global_pheromone": diff_global_pheromone,
            "ht": ht
        }
        self.params = {
            "d": d,
            **agents_params
        }
        self.name = "MSAMSOA"

        logging.info("Populate agents with: count=%d; a=%.2f; b=%d", agents_cnt, a, b)
        self.init_agents(agents_params)

        logging.info("Space initialization and agents distribution is finish successfully")

    def __repr__(self):
        return ("SAMSOA Simulation. Agents:{}, a:{a}, b:{b}, d:{d}, dtl0:{dtl0}, dtg0:{dtg0}, ht:{ht}."
                ).format(self.agents_cnt, *(self.params))

    def init_agents(self, agent_params):
        self.agents = [
            MSAMSOA_Agent(i, self.boundary, **agent_params)
            for i in range(1, self.agents_cnt+1)
        ]
        self.random_pos_agents()

    def random_pos_agents(self):
        assigned_positions = []
        min_dist = self.boundary // self.agents_cnt # Minimal distance with other agent
        for agent in self.agents:
            dist = 0
            while (dist < min_dist):
                position = MSAMSOA.get_position(self.boundary)
                direction = MSAMSOA.get_direction(position, self.boundary)

                # Calculate minimum distance to other agents
                dist = min(list(
                    map(lambda p: np.linalg.norm(np.array(position) - np.array(p)), assigned_positions)
                )) if (assigned_positions) else np.inf

            agent.position = position
            agent.direction = direction
            assigned_positions.append(position)

    @staticmethod
    def get_position(boundary):
        side = -1 if (random.randint(2)) else boundary # Bot/Left or Top/Right side
        loc = random.randint(boundary) # Location between Boundary
        position = [side, loc] if (random.randint(2)) else [loc, side]
        return position

    @staticmethod
    def get_direction(position, boundary):
        if (position[0] == -1): direction = (1, 0)
        elif (position[1] == -1): direction = (0, 1)
        elif (position[0] == boundary): direction = (-1, 0)
        elif (position[1] == boundary): direction = (0, -1)
        return direction

    ##### Execution Methods #####
    def execute(self, max_iteration=np.inf, level="debug", track_process=True, track_path="track"):
        """
        Main execution method

        Params:
        - max_iteration: int (default = inf); Max iteration for algorithm to run.
        - level: str [option: "debug", "error", "silent"] (default = "debug"); Debugging level.
        - track_process: boolean (default = True); Track information per iteration and export to progress file.
        - log_path: str (default = "track_path"); Path to log file.
        """
        bound = self.boundary
        visited_field = np.zeros((bound, bound), bool)
        occupied_field = np.zeros((bound, bound), bool)
        fertilized_field = self.space.copy()
        agents = self.agents.copy()

        iteration = 0
        detected_targets = set()
        broken_agents = []

        target_cnt = self.size - np.sum(fertilized_field)
        tracking = Tracker(field_size=self.size, target_cnt=target_cnt)
        tracking.add_snapshot(0, visited_field, fertilized_field, agents)

        while(
            (np.sum(visited_field) < self.size or detected_targets) # While surveillance and fertilization not completed
            and iteration < max_iteration                           # or iteration lower than max_iteration
        ):
            # Calculating completion rate
            iteration += 1
            s_rate = np.sum(visited_field)/self.size
            f_rate = (target_cnt - (self.size - np.sum(fertilized_field))) / target_cnt

            # Grouping agent by mission
            surveillance_agents = [agent for agent in agents if (agent.mission == "surveillance")]
            fertilization_agents = [agent for agent in agents if (agent.mission == "fertilization")]
            logging.info(
                f"Starting iteration {iteration:4}. Surveillance: {s_rate*100:6.2f}%. Fertilization: {f_rate*100:6.2f}%. "
                f"Surveillance Agents: {len(surveillance_agents):3}. Fertilization Agents: {len(fertilization_agents):3}"
            )

            # Fertilization
            for agent in fertilization_agents:
                fertilized_field, detected_targets = MSAMSOA.fertilization(agent, fertilized_field, detected_targets)
                agent.set_mission("surveillance")

            # Surveillance
            for agent in surveillance_agents:
                origin, destination, occupied_field = agent.move(occupied_field)
                if(
                    (origin[0] >= 0 and origin[0] < self.boundary) and
                    (origin[1] >= 0 and origin[1] < self.boundary)
                ):
                    occupied_field[origin[0], origin[1]] = False
                occupied_field[destination[0], destination[1]] = True

                visited_field, detected_targets = MSAMSOA.surveillance(agent, visited_field, fertilized_field, detected_targets)
                x_pos, y_pos = agent.position
                if (fertilized_field[x_pos, y_pos] == False):
                    agent.set_mission("fertilization")
                else:
                    agent.check_hungry_state()

            for agent in surveillance_agents:
                agent.update_local_pheromone(surveillance_agents)

            # Reduce agents power
            for agent in agents:
                agent.reduce_power(1)

            # Take progress snapshot
            tracking.add_snapshot(iteration, visited_field, fertilized_field, agents)

    ##### Mission Methods #####
    @staticmethod
    def fertilization(agent, fertilized_field, detected_targets):
        fertilized_field, detected_targets = Solution.fertilization(agent, fertilized_field, detected_targets)
        if (agent.hungry_state):
            agent.set_hungry_state(False)
        return fertilized_field, detected_targets

    @staticmethod
    def surveillance(agent, visited_field, fertilized_field, detected_targets):
        visited_field, detected_targets, new_grid_cnt = Solution.surveillance(agent, visited_field, fertilized_field, detected_targets)
        if (agent.hungry_state == False):
            agent.no_new_grid_streak += 1
        return visited_field, detected_targets

                #     search_agents = []
                #     for agent in self.agents:
                #
                #         if(agent.power):
                #             agent.power -= 1
                #             if(agent.power == 0):
                #                 agent.mission = 2
                #                 y, x = agent.position
                #                 amap[y,x] = True
                #
                #         # Misi Search
                #         if(agent.mission == 0):
                #             search_agents.append(agent.i)
                #
                #             oy, ox = agent.position
                #             direction = self.unvisitedDirection(agent.position, vmap, emap) if agent.hungry else None
                #             y, x = agent.search(amap, vmap, direction)
                #             if(oy >= 0 and oy <= nw-1 and ox >= 0 and ox <= nl-1):
                #                 amap[oy, ox] = True
                #             amap[y, x] = False
                #
                #             # Identifikasi Daerah
                #             new_grid = 0
                #             for dy in [-2, -1, 0, 1, 2]:
                #                 for dx in [-2, -1, 0, 1, 2]:
                #                     ny = y + dy
                #                     nx = x + dx
                #                     if(ny >= 0 and ny <= nw-1 and nx >= 0 and nx <= nl-1 and vmap[ny,nx] == False):
                #                         vmap[ny,nx] = True
                #                         new_grid += 1
                #                         if(emap[ny,nx] == False and [ny,nx] not in target_list):
                #                             target_list.append((ny,nx))
                #             agent.hpoint = agent.hpoint + 1 if new_grid == 0 else 0
                #             agent.hungry = True if agent.hpoint > ht else False
                #
                #         # Misi Attack
                #         elif(agent.mission == 1):
                #             y, x = agent.position
                #             emap[y, x] = True
                #             target_list.remove((y, x))
                #             for a in self.agents:
                #                 a.cp[y, x] = 0
                #             ni += 1
                #             agent.mission = 0
                #
                #         # Agen Rusak
                #         elif(agent.mission == 2):
                #             y, x = agent.position
                #             if(agent.i not in broken_agents):
                #                 broken_agents.append(agent.i)
                #                 amap[y, x] = True
                #
                #     # Memperbaharui Nilai Feromon
                #     self.updatePheromone(search_agents, target_list)
                #
                #     # Memilih Misi Selanjutnya untuk Tiap Agen
                #     for agent in self.agents:
                #         y, x = agent.position
                #         if (agent.mission != 2):
                #             if emap[y,x]:
                #                 agent.mission = 0
                #             else:
                #                 agent.mission = 1
                #                 agent.hungry = False
                #                 agent.hpoint = 0
                #
                #     # Memperbaharui Nilai Srate dan Frate
                #     srate = sum(sum(vmap))
                #
                #     agent_info = [ (agent.position, agent.mission) for agent in self.agents]
                #     tracker.update(iteration, vmap, emap, ni, agent_info, simulation)
                #
                #     print(("Iteration {}. Srate:{:5.2f}%. Frate:{:5.2f}%"
                #           ).format(iteration, (srate/self.size)*100, (ni/self.nt)*100)) if show >= 1 else None
                #     if show >= 2:
                #         for agent in self.agents:
                #             print(("Agent-{}. Mission:{}. Position:({:2},{:2}). Hungry:{}"
                #                   ).format(agent.i, agent.mission, *(agent.position), agent.hungry))
                #         print('')
                # self.tracker = tracker

class MSAMSOA_Agent(Agent):
    """
    MSAMSOA agent, or UAV representation for MSAMSOA solution.

    Init Params:
    - idx: int; Unique identifier for each agents.
    - boundary: int; Boundary location (x or y axis) of space field.
    - a: float; Pheromone value importance in navigation decision.
    - b: float; Heuristic value importance in navigation decision.
    - diff_local_pheromone: float; Base pheromone for local reduction.
    - diff_global_pheromone: float; Base pheromone for global addition.
    - ht: int; Maximum iteration of no new grid surveillance before agent enter hungry state.
    """
    def __init__(self, idx, boundary, a, b, diff_local_pheromone, diff_global_pheromone, ht):
        super().__init__(idx, boundary)
        self.params = {
            "a": a,
            "b": b,
            "diff_local_pheromone": diff_local_pheromone,
            "diff_global_pheromone": diff_global_pheromone,
            "ht": ht
        }
        self.mission = "surveillance"
        self.power = 100

        self.move_direction = [(0,2), (2,0), (0,-2), (0,2), (0,1), (1,1), (1,0), (1,-1), (0,-1), (-1,-1), (-1,0), (-1,1)]
        self.local_pheromone = np.full((boundary, boundary), 10, dtype=float)
        self.convening_pheromone = np.zeros((boundary, boundary), dtype=float)
        self.hungry_point = 0
        self.hungry_state = False
        self.no_new_grid_streak = 0

    ##### State #####
    def set_mission(self, mission):
        self.mission = mission

    def reduce_power(self, power):
        if (self.mission):
            self.power -= 1
            if (self.power <= 0):
                self.mission = None # Broken/Dead

    ##### Navigation #####
    def move(self, occupied_field):
        available_grid = self.get_available_grid(occupied_field)
        grid_fitness = self.get_fitness_score(available_grid)
        grid_choices = list(zip(available_grid, grid_fitness))
        best_grid = max(grid_choices, key=lambda x: x[1])

        origin = self.position
        self.position = best_grid[0]
        destination = self.position
        return origin, destination, occupied_field

    def get_fitness_score(self, grids):
        a = self.params.get("a", 0.7)
        b = self.params.get("b", 0.3)

        grid_fitness = []
        for grid in grids:
            x_pos, y_pos = grid
            fitness = a * self.local_pheromone[x_pos, y_pos]
            grid_fitness.append(fitness)
        return grid_fitness

    ##### Pheromone Management #####
    def update_local_pheromone(self, surveillance_agents):
        R = self.radar_range
        F = random.random(1)[0]
        x_pos, y_pos = self.position
        diff_local_pheromone = self.params.get("diff_local_pheromone", 1.0)
        diff_global_pheromone = self.params.get("diff_global_pheromone", 0.1)

        for dy in range(-R, R):
            for dx in range(-R, R):
                x_grid = x_pos + dx
                y_grid = y_pos + dy
                if (MSAMSOA.check_boundary(x_grid, y_grid, self.boundary)):

                    # Calculate reduction from other agents
                    pheromone_reduction = 0
                    local_agents = [
                        (agent.position, np.linalg.norm(np.array([x_grid, y_grid]) - np.array(agent.position)))
                        for agent in surveillance_agents
                    ]
                    for agent in local_agents:
                        if ((agent[1]**4) <= R**4):
                            pheromone_reduction += diff_local_pheromone * ((R**4) * (agent[1]**4))/(R**4)

                    # Calculate global addition
                    pheromone_addition = F * diff_global_pheromone

                    self.local_pheromone[x_grid, y_grid] -= pheromone_reduction
                    self.local_pheromone[x_grid, y_grid] += pheromone_addition

    ##### Agent State Management #####
    def set_hungry_state(self, state):
        self.hungry_state = state

    def check_hungry_state(self):
        ht = self.params.get("ht", 10)
        if (self.no_new_grid_streak >= ht):
            self.set_hungry_state(True)


    #         nw, nl = self.boundary
    #         R = 2
    #         d, dtl0, dtg0= self.params[2], self.params[3], self.params[4]
    #
    #         for idx in search_agents:
    #             agent = self.agents[idx-1]
    #             y, x = agent.position
    #
    #             # Memperbaharui Feromon Lokal
    #             for dy in [-2, -1, 0, 1, 2]:
    #                 for dx in [-2, -1, 0, 1, 2]:
    #                     ny = y + dy
    #                     nx = x + dx
    #                     if(ny >= 0 and ny <= nw-1 and nx >= 0 and nx <= nl-1):
    #                         for oa_idx in search_agents:
    #                             other_agent = self.agents[oa_idx-1]
    #                             oay, oax = other_agent.position
    #                             dis = np.sqrt((oay - ny)**2 + (oax - nx)**2)
    #                             dt = dtl0 * (R**4 - dis**4)/(R**4) if R**4 >= dis**4 else 0
    #                             agent.p[ny, nx] = max(agent.p[ny, nx] - dt, 1e-15)
    #
    #             # Memperbaharui Feromon Penyusun
    #             for (ty, tx) in target_list:
    #                 if(agent.cp[ty, tx] == 0):
    #                     pow_dis = (ty - y)**2 + (tx - x)**2
    #                     agent.cp[ty, tx] = dtg0 * np.exp(-pow_dis/(2*(d**2)))

# import numpy as np
# from msamsoa.utils.tracker import Tracker
# from queue import Queue

# class MSAMSOA_Agent:
#
#     # Inisiasi Kelas
#     def __init__(self, i, a, b, boundary):
#
#         self.mission = 0
#         self.hpoint = 0
#         self.hungry = False
#         self.power = np.inf
#
#         self.boundary = boundary
#         nw, nl = boundary
#         self.p = np.ones((nw, nl), np.float32)
#         self.cp = np.zeros((nw, nl), np.float32)
#
#     # Menghasilkan Posisi yang Dapat Dikunjungi
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
#
#     # Menghitung Nilai Fitness
#     def calculateFitness(self, grida, vmap):
#         nw, nl = self.boundary
#         a, b = self.params
#         fitness = []
#         for (y, x) in grida:
#             pheromone = self.p[y,x] + self.cp[y,x]
#
#             v = vmap.copy()
#             for dy in [-2, -1, 0, 1, 2]:
#                 for dx in [-2, -1, 0, 1, 2]:
#                     ny = y + dy
#                     nx = x + dx
#                     if(ny >= 0 and ny <= nw-1 and nx >= 0 and nx <= nl-1):
#                         v[ny, nx] = True
#             p_srate = sum(sum(v)) / (nw*nl)
#
#             fit = (a * pheromone) + (b * p_srate)
#             fitness.append([fit, (y, x)])
#         return fitness
#
#     # Misi Search
#     def search(self, amap, vmap, target_direction):
#         grida = self.availableGrid(amap, target_direction)
#         fitness = self.calculateFitness(grida, vmap)
#
#         best = self.position
#         val = -1
#         for (fit, pos) in fitness:
#             if (val < fit):
#                 val = fit
#                 best = pos
#         self.position = best
#         return self.position
#
# class MSAMSOA:
#
#     # Inisiasi Kelas
#     def __init__(self, env, na, a=0.7, b=0.3, d=80, dtl0=0.1, dtg0=1.0, ht=10):
#         self.e = env.copy()
#         self.boundary = env.shape
#         self.size = env.shape[0] * env.shape[1]
#         self.nt = self.size - sum(sum(env))
#         self.params = (a, b, d, dtl0, dtg0, ht)
#         self.na = na
#         self.agents = [ SAMSOA_Agent(i, a, b, env.shape) for i in range(1,na+1) ]
#         self.randomPut()
#
#         self.name = 'SAMSOA'
#
#     # Representasi Kelas
#     def __repr__(self):
#         return ("SAMSOA Simulation. NA:{}, a:{}, b:{}, d:{}, dtl0:{}, dtg0:{}, ht:{}."
#                ).format(self.na, *(self.params))
#
#     # Menempatkan Agen di posisi acak
#     def randomPut(self):
#         mindis = self.boundary[0]//self.na
#         bot, top = (-1, self.boundary[0])
#         agent_pos = []
#         for _ in range(1, self.na+1):
#             dis = 0
#             while(dis <= mindis):
#                 out = bot if np.random.randint(10)%2 else top
#                 ins = np.random.randint(self.boundary[0])
#                 pos = (out, ins) if np.random.randint(10)%2 else (ins, out)
#
#                 dis = 200 if len(agent_pos) > 1 else mindis+1
#                 for p in agent_pos:
#                     ndis = np.sqrt((pos[0]-p[0])**2 + (pos[1]-p[1])**2)
#                     dis = ndis if dis > ndis else dis
#             agent_pos.append(pos)
#
#         for i in range(len(agent_pos)):
#             self.agents[i].position = agent_pos[i]
#             for (dy, dx) in zip([-1,1,0,0], [0,0,-1,1]):
#                     ny = agent_pos[i][0] + dy
#                     nx = agent_pos[i][1] + dx
#                     if(ny >= 0 and ny <= self.boundary[0]-1 and nx >= 0 and nx <= self.boundary[0]-1):
#                         self.agents[i].direction = (dy, dx)
#
#     # Mencari Daerah yang Belum Dikunjungi
#     def unvisitedDirection(self, start, vmap, emap):
#         nw, nl = self.boundary
#         sy, sx = start
#         cmap = np.zeros((nw, nl), np.bool)
#
#         q = Queue()
#         q.put((sy, sx))
#         cmap[sy, sx] = True
#         loc = (sy, sx)
#         while(q.qsize()):
#             y, x = q.get()
#
#             if(vmap[y,x] == False or emap[y,x] == False):
#                 loc = (y,x)
#                 break
#
#             for dy in [-1, 0, 1]:
#                 for dx in [-1, 0, 1]:
#                     ny = y + dy
#                     nx = x + dx
#                     if(ny >= 0 and ny <= nw-1 and nx >= 0 and nx <= nl-1 and cmap[ny,nx] == False):
#                         q.put((ny, nx))
#                         cmap[ny, nx] = True
#
#         ydir = max(-1, min(1, loc[0] - sy))
#         xdir = max(-1, min(1, loc[1] - sx))
#         return (ydir, xdir)
#
#     # Memperbaharui Nilai Feromon
#     def updatePheromone(self, search_agents, target_list):
#         nw, nl = self.boundary
#         R = 2
#         d, dtl0, dtg0= self.params[2], self.params[3], self.params[4]
#
#         for idx in search_agents:
#             agent = self.agents[idx-1]
#             y, x = agent.position
#
#             # Memperbaharui Feromon Lokal
#             for dy in [-2, -1, 0, 1, 2]:
#                 for dx in [-2, -1, 0, 1, 2]:
#                     ny = y + dy
#                     nx = x + dx
#                     if(ny >= 0 and ny <= nw-1 and nx >= 0 and nx <= nl-1):
#                         for oa_idx in search_agents:
#                             other_agent = self.agents[oa_idx-1]
#                             oay, oax = other_agent.position
#                             dis = np.sqrt((oay - ny)**2 + (oax - nx)**2)
#                             dt = dtl0 * (R**4 - dis**4)/(R**4) if R**4 >= dis**4 else 0
#                             agent.p[ny, nx] = max(agent.p[ny, nx] - dt, 1e-15)
#
#             # Memperbaharui Feromon Penyusun
#             for (ty, tx) in target_list:
#                 if(agent.cp[ty, tx] == 0):
#                     pow_dis = (ty - y)**2 + (tx - x)**2
#                     agent.cp[ty, tx] = dtg0 * np.exp(-pow_dis/(2*(d**2)))
#
#     # Menyelesaikan masalah
#     def execute(self, show=0, simulation=False, max_iteration=np.inf):
#         nw, nl = self.boundary
#         ht = self.params[5]
#         emap = self.e.copy()
#         vmap = np.zeros((nw, nl), np.bool)
#         amap = np.ones((nw, nl), np.bool)
#
#         iteration = 0
#         srate = 0
#         ni = 0
#         target_list = []
#         broken_agents = []
#         tracker = Tracker(self.size, self.nt)
#         agent_info = [ (agent.position, agent.mission) for agent in self.agents]
#         tracker.update(0, vmap, emap, ni, agent_info, simulation)
#         while((srate < self.size or ni < self.nt) and iteration < max_iteration):
#             iteration += 1
#
#             # Melakukan Misi
#             search_agents = []
#             for agent in self.agents:
#
#                 if(agent.power):
#                     agent.power -= 1
#                     if(agent.power == 0):
#                         agent.mission = 2
#                         y, x = agent.position
#                         amap[y,x] = True
#
#                 # Misi Search
#                 if(agent.mission == 0):
#                     search_agents.append(agent.i)
#
#                     oy, ox = agent.position
#                     direction = self.unvisitedDirection(agent.position, vmap, emap) if agent.hungry else None
#                     y, x = agent.search(amap, vmap, direction)
#                     if(oy >= 0 and oy <= nw-1 and ox >= 0 and ox <= nl-1):
#                         amap[oy, ox] = True
#                     amap[y, x] = False
#
#                     # Identifikasi Daerah
#                     new_grid = 0
#                     for dy in [-2, -1, 0, 1, 2]:
#                         for dx in [-2, -1, 0, 1, 2]:
#                             ny = y + dy
#                             nx = x + dx
#                             if(ny >= 0 and ny <= nw-1 and nx >= 0 and nx <= nl-1 and vmap[ny,nx] == False):
#                                 vmap[ny,nx] = True
#                                 new_grid += 1
#                                 if(emap[ny,nx] == False and [ny,nx] not in target_list):
#                                     target_list.append((ny,nx))
#                     agent.hpoint = agent.hpoint + 1 if new_grid == 0 else 0
#                     agent.hungry = True if agent.hpoint > ht else False
#
#                 # Misi Attack
#                 elif(agent.mission == 1):
#                     y, x = agent.position
#                     emap[y, x] = True
#                     target_list.remove((y, x))
#                     for a in self.agents:
#                         a.cp[y, x] = 0
#                     ni += 1
#                     agent.mission = 0
#
#                 # Agen Rusak
#                 elif(agent.mission == 2):
#                     y, x = agent.position
#                     if(agent.i not in broken_agents):
#                         broken_agents.append(agent.i)
#                         amap[y, x] = True
#
#             # Memperbaharui Nilai Feromon
#             self.updatePheromone(search_agents, target_list)
#
#             # Memilih Misi Selanjutnya untuk Tiap Agen
#             for agent in self.agents:
#                 y, x = agent.position
#                 if (agent.mission != 2):
#                     if emap[y,x]:
#                         agent.mission = 0
#                     else:
#                         agent.mission = 1
#                         agent.hungry = False
#                         agent.hpoint = 0
#
#             # Memperbaharui Nilai Srate dan Frate
#             srate = sum(sum(vmap))
#
#             agent_info = [ (agent.position, agent.mission) for agent in self.agents]
#             tracker.update(iteration, vmap, emap, ni, agent_info, simulation)
#
#             print(("Iteration {}. Srate:{:5.2f}%. Frate:{:5.2f}%"
#                   ).format(iteration, (srate/self.size)*100, (ni/self.nt)*100)) if show >= 1 else None
#             if show >= 2:
#                 for agent in self.agents:
#                     print(("Agent-{}. Mission:{}. Position:({:2},{:2}). Hungry:{}"
#                           ).format(agent.i, agent.mission, *(agent.position), agent.hungry))
#                 print('')
#         self.tracker = tracker
