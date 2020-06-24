import numpy as np

class Tracker:

    # Inisiasi Kelas
    def __init__(self, size, nt):
        self.size = size
        self.nt = nt
        self.ts = 0
        self.tf = 0

        self.vm_tracks = []
        self.tm_tracks = []
        self.agent_position = []
        self.s_tracks = []
        self.f_tracks = []

    # Menambahkan Informasi Simulasi
    def update(self, iteration, vmap, emap, ni, agents, simulation):
        self.iteration = iteration

        if(simulation):
            self.vm_tracks.append(vmap.copy())
            self.tm_tracks.append(emap.copy())
            self.agent_position.append(agents.copy())
        self.s_tracks.append(sum(sum(vmap)))
        self.f_tracks.append(ni)

        self.ts = self.iteration if self.s_tracks[-1] != self.size else self.ts
        self.tf = self.iteration if self.f_tracks[-1] != self.nt else self.tf

    # Mendapatkan Posisi dari Agen pada Iterasi
    def getAgentLocations(self, i):
        if(i == 0):
            return ([],[]), ([],[])
        else:
            nloc = np.array([agent[0] for agent in self.agent_position[i] if agent[1] != 2]).transpose()
            bloc = np.array([agent[0] for agent in self.agent_position[i] if agent[1] == 2]).transpose()
            if(nloc.shape[0] == 0):
                nloc = np.array([[], []])
            if(bloc.shape[0] == 0):
                bloc = np.array([[], []])
            return (nloc, bloc)
