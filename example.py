from msamsoa.algorithm import MSAMSOA
from msamsoa.scenarios import sc1

model = MSAMSOA(sc1, 10)
model.execute(max_iteration=100)
for agent in model.agents:
    print(agent.position, agent.direction)
