from msamsoa.visualization.parser import parse_field_data, parse_agents_data
from msamsoa.visualization.visualizer import Visualizer

field_gen = parse_field_data("track", "fertilized_field.csv")
agents_gen = parse_agents_data("track", "agents.csv")

# Single Visualization
# Visualizer.visualize_field(next(field_gen), title="Fertilized Field")

# Visualize from Tracks
viz = Visualizer(track_dir="track")
viz.visualize_frame(1)
viz.visualize_frame(2)
viz.visualize_frame(3)
viz.visualize_frame(4)
