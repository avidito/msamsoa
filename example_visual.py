from msamsoa.visualization.parser import parse_field_data, parse_agents_data
from msamsoa.visualization.visualizer import Visualizer

field_gen = parse_field_data("track", "fertilized_field.csv")
agents_gen = parse_agents_data("track", "agents.csv")

viz = Visualizer()
viz.visualize_field(next(field_gen), title="Fertilized Field")
