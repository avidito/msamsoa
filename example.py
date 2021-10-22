from msamsoa.algorithm import MSAMSOA
from msamsoa.scenarios import sc1

model = MSAMSOA(sc1, 10)
for agent in model.agents:
    print(f"{agent.id} : {agent.position}")
