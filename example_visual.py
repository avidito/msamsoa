from msamsoa.visualization.parser import parse_field_data, parse_agents_data

field_gen = parse_field_data("track", "fertilized_field.csv")
agents_gen = parse_agents_data("track", "agents.csv")
