import os
import sys

import numpy as np
import pandas as pd

try:
    instance = sys.argv[1]
except IndexError:
    print("{0} <instance>".format(sys.argv[0]))
    sys.exit(1)

instance_name = os.path.split(instance)[1].replace('_instance', '')
matrix_flow_file = f'data/results/instances_output/summary_outputs/' \
                   f'{instance_name}_flowmatrix2020to2050.csv'

# Load matrix_flow_data as dataframe
matrix_flow_data = pd.read_csv(matrix_flow_file, index_col=0)
col_names = matrix_flow_data.columns
row_names = matrix_flow_data.index

matrix_flow_data = np.array(matrix_flow_data)

# Convert all diagonal values to 0
np.fill_diagonal(matrix_flow_data, 0)

for row in range(matrix_flow_data.shape[0]):
    for col in range(row, matrix_flow_data.shape[1]):

        effective_flow = matrix_flow_data[row, col] - matrix_flow_data[
            col, row]
        if effective_flow >= 0:
            matrix_flow_data[row, col] = effective_flow
            matrix_flow_data[col, row] = 0
        else:
            matrix_flow_data[row, col] = 0
            matrix_flow_data[col, row] = -effective_flow

matrix_flow_data = pd.DataFrame(matrix_flow_data)
matrix_flow_data.columns = col_names
matrix_flow_data.index = row_names

effective_matrix_flow_file = f'data/results/instances_output/' \
                             f'summary_outputs/{instance_name}' \
                             f'_effective_flowmatrix2020to2050.csv'

matrix_flow_data.to_csv(effective_matrix_flow_file, index=True)
