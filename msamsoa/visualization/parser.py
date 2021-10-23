"""
Parser Module

Module to handle field or agents data parsing from CSV to representable format.

Include:
    - parse_field_data (function): Convert flatten field data into 2D Numpy Array generator.
    - parse_agents_data (function): Convert agents data to 2D list generator.
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
        for row in reader:
            matrix_data = np.reshape(row, (size, size))
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
                print(batch_data)
                yield batch_data

                batch_data.clear()
                batch_data.append(row)
    finally:
        file.close()
