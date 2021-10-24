"""
Parser Module

Module to handle field or agents data parsing from CSV to representable format.

Include:
    - parse_field_data (function): Convert flatten field data into 2D Numpy Array generator.
    - parse_agents_data (function): Convert agents data to 2D list generator.
    - parse_summary_data (function): Convert summary data to Dictionary generator.
"""

import os
import csv
import numpy as np

def parse_field_data(dir, filename):
    """
    Convert flatten field data into 2D Numpy Array generator.

    Params:
    - dir: string; Directory of track records.
    - filename: string; Full name (with extension) of field data CSV file.
    """
    filepath = os.path.join(dir, filename)
    try:
        file = open(filepath, "r", encoding="utf-8")
        reader = csv.reader(file)
        header = next(reader)
        size = int(np.sqrt(len(header)))
        for row in reader:
            row_int = np.vectorize(int)(row)
            matrix_data = np.reshape(row_int, (size, size))
            yield matrix_data
    finally:
        file.close()

def parse_agents_data(dir, filename):
    """
    Convert agents data to 2D list generator.

    Params:
    - dir: string; Directory of track records.
    - filename: string; Full name (with extension) of field data CSV file.
    """
    filepath = os.path.join(dir, filename)
    try:
        file = open(filepath, "r", encoding="utf-8")
        reader = csv.reader(file)
        header = next(reader)

        iteration_pivot = 0
        batch_data = []
        for row in reader:
            iteration = int(row[0])
            if (iteration == iteration_pivot):
                batch_data.append(row)

            elif (batch_data):
                iteration_pivot = iteration
                yield batch_data

                batch_data.clear()
                batch_data.append(row)
    finally:
        file.close()

def parse_summary_data(dir, filename):
    """
    Convert summary data to Dictionary generator.

    Params:
    - dir: string; Directory of track records.
    - filename: string; Full name (with extension) of field data CSV file.
    """
    filepath = os.path.join(dir, filename)
    columns = ["iteration", "active_agents", "agent_surveillance", "agent_fertilization", "surveillance_grid", "surveillance_rate", "fertilization_grid", "fertilization_rate"]
    columns_int = ["iteration", "active_agents", "agent_surveillance", "agent_fertilization", "surveillance_grid", "fertilization_grid"]
    columns_float = ["surveillance_rate", "fertilization_rate"]
    try:
        file = open(filepath, "r", encoding="utf-8")
        reader = csv.reader(file)
        header = next(reader)
        for row in reader:
            data_raw = {col: value for col, value in zip(columns, row)}
            data_int = {col: int(value) for col, value in data_raw.items() if (col in columns_int)}
            data_float = {col: float(value) for col, value in data_raw.items() if (col in columns_float)}
            data = {**data_int, **data_float}
            yield data

    finally:
        file.close()
