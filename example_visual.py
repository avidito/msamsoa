from msamsoa.visualization.parser import parse_field_data, parse_agents_data, parse_summary_data
from msamsoa.visualization.visualizer import Visualizer

# Single Visualization
field_gen = parse_field_data("track", "fertilized_field.csv")
agents_gen = parse_agents_data("track", "agents.csv")
summary_gen = parse_summary_data("track", "summary.csv")
summary = next(summary_gen)

# Visualizer.visualize_field(
#     next(field_gen),
#     next(agents_gen),
#     {"surveillance": summary["surveillance_rate"], "fertilization": summary["fertilization_rate"]},
#     10,
#     iteration=0
# )

# Visualize from Tracks
viz = Visualizer(track_dir="track")
viz.visualize_frame(50)
