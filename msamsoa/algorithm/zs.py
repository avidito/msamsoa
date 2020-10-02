import sys
sys.path.append('..')

import numpy as np
from msamsoa.utils.tracker import Tracker
from queue import Queue

class ZS_Agent:

    # Inisiasi Kelas
    def __init__(self, i, boundary, dis):
        self.i = i
        self.boundary = boundary
        self.segment = ((i-1)*dis)+2
        self.region = ((i-1)*dis, i*dis)

        self.position = (-2, self.segment)
        self.edge = False
        self.forward = True
        self.mission = 0
        self.work = True
        self.power = np.inf

        self.lf_pid = -1
        self.lf_track = None

    # Melakukan Perpindahan Gerak Lurus
    def moveStraight(self):
        nw, nl = self.boundary

        if (self.forward):
            self.position = (min(self.position[0] + 2, nw-1), self.position[1])
            self.edge = True if(self.position[0] == nw-1) else False
        else:
            self.position = (max(self.position[0] - 2, 0), self.position[1])
            self.edge = True if(self.position[0] == 0) else False

    # Melakukan Perpindahan ke Segmen Selanjutnya
    def changeSegment(self, next_segment):
        self.position = (self.position[0], min(self.position[1]+2, next_segment))
        if(self.position[1] == next_segment):
            self.segment = next_segment
            self.edge = False
            self.forward = not (self.forward)

    # Melakukan Perpindahan untuk Pemupukan Lokal
    def localFertilization(self):
        self.position = self.lf_track[self.lf_pid]
        self.lf_pid += 1
        if(self.lf_pid == len(self.lf_track)):
            self.lf_pid = -1
            self.mission = 0

class ZS_Problem:

    # Inisiasi Kelas
    def __init__(self, env, na):
        self.e = env.copy()
        self.boundary = env.shape
        self.size = env.shape[0] * env.shape[1]
        self.nt = self.size - sum(sum(env))

        self.na = na
        dis = env.shape[0] // na
        self.agents = [ ZS_Agent(i, env.shape, dis) for i in range(1,na+1) ]
        self.agents[-1].region = (self.agents[-1].region[0], self.boundary[1])

        self.name = 'ZS'

    # Membuta Jalur Pemupukan Lokal
    def generateLFTrack(self, start, forward):
        nw, nl = self.boundary
        if(start[0] == nw-1 or start[0] == 0):
            t = np.array([(0,1), (0,2), (1,2), (2,2), (2,1), (1,1), (2,0), (1,0), (2,-1), (1,-1), (2,-2),
                          (1,-2), (0,-2), (0,-1), (0,0)], np.int)
        elif((start[0] == nw-2 and forward) or (start[0] == 1 and not forward)):
            t = np.array([(0,1), (0,2), (1,2), (1,1), (1,0), (1,-1), (1,-2), (0,-2), (0,-1), (0,0)], np.int)
        else:
            t = np.array([(1,1), (1,2), (2,2), (2,1), (2,0), (2,-1), (2,-2), (1,-2), (1,-1), (1,0), (0,0)],
                         np.int)
        t = t if (forward) else -t

        return [(start[0] + dy, start[1] + dx) for (dy, dx) in t]

    # Menyelesaikan Masalah
    def execute(self, show=0, simulation=False, max_iteration=np.inf):
        nw, nl = self.boundary
        emap = self.e.copy()
        vmap = np.zeros((nw, nl), np.bool)

        iteration = 0
        srate = 0
        ni = 0
        tracker = Tracker(self.size, self.nt)
        agent_info = agent_info = [ (agent.position, agent.mission) for agent in self.agents]
        tracker.update(0, vmap, emap, ni, agent_info, simulation)
        done = 0
        while(((srate < self.size) or (ni < self.nt)) and iteration < max_iteration and done < self.na):
            iteration += 1

            for agent in self.agents:

                if(agent.power):
                    agent.power -= 1
                    if(agent.power == 0):
                        agent.mission = 2
                        done += 1

                # Melakukan Perpindahan untuk Membaca Lahan
                if(agent.mission == 0):
                    if (agent.edge and agent.segment+3 >= agent.region[1]):
                        if(agent.work):
                            agent.work = False
                            done += 1
                        continue

                    scan = False
                    if not (agent.edge):
                        agent.moveStraight()
                        scan = True
                    else:
                        next_segment = min(agent.segment + 5, agent.region[1]-3)
                        agent.changeSegment(next_segment)
                        if(agent.segment == next_segment):
                            scan = True
                    if (scan):
                        y, x = agent.position
                        for dy in [-2, -1, 0, 1, 2]:
                            for dx in [-2, -1, 0, 1, 2]:
                                ny = y + dy
                                nx = x + dx
                                if(ny >= 0 and ny <= nw-1 and nx >= 0 and nx <= nl-1 and vmap[ny,nx]== False):
                                    vmap[ny, nx] = True
                                    if(emap[ny, nx] == False):
                                        agent.mission = 1

                # Melakukan Perpindahan untuk Pemupukan
                elif (agent.mission == 1):
                    y, x = agent.position
                    if (emap[y,x] == False):
                        emap[y,x] = True
                        ni += 1
                    else:
                        if (agent.lf_pid == -1):
                            agent.lf_track = self.generateLFTrack((y,x), agent.forward)
                            agent.lf_pid = 0
                        agent.localFertilization()

            srate = sum(sum(vmap))

            agent_info = agent_info = [ (agent.position, agent.mission) for agent in self.agents]
            tracker.update(iteration, vmap, emap, ni, agent_info, simulation)
            print(("Iteration {}. Srate:{:5.2f}%. Frate:{:5.2f}%"
                      ).format(iteration, (srate/self.size)*100, (ni/self.nt)*100)) if show >= 1 else None
            if show >= 2:
                for agent in self.agents:
                    print(("Agent-{}. Mission:{}. Position:({:2},{:2}). Forward={}. Edge={}"
                          ).format(agent.i, agent.mission, *(agent.position), agent.forward, agent.edge))
                print('')
        self.tracker = tracker
