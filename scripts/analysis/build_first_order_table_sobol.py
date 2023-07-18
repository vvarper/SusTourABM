import os
import sys

import numpy as np
import pandas as pd
from SALib.analyze import sobol as sobol_analyze

from sustourabm.climatic_policy import ClimaticPolicy
from sustourabm.util.io import load_model_instance_from_json

# Process arguments from command line #########################################

try:
    instance = sys.argv[1]
    n = int(sys.argv[2])
    policy_start = int(sys.argv[3])
    mc = int(sys.argv[4])
    num_processes = int(sys.argv[5])
    num_agents = int(sys.argv[6])

except IndexError:
    print(
        "{0} <instance> <n> <start> <mc> <num_processes> <num_agents>".format(
            sys.argv[0]))
    sys.exit(1)

instance_path = 'data/instances/' + instance + '.json'
instance_name = os.path.split(instance)[1].replace('_instance', '')
output_file = f'data/results/sensitivity_analysis/sobol_n{n}_{instance_name}' \
              f'_{policy_start}_{mc}_{num_processes}.csv'
parameters, climate_factors, destinations = load_model_instance_from_json(
    instance_path)
parameters['num_tourists'] = num_agents

# Create empty dataframe
sobol_df = pd.DataFrame(columns=destinations)

for destination_id, destination_name in enumerate(destinations):
    eval_path = f'data/results/sensitivity_analysis/sample_evals_n{n}' \
                f'_{instance_name}_{destination_name}_{policy_start}' \
                f'_{mc}_{num_processes}.txt'

    Y = np.loadtxt(eval_path)

    problem = ClimaticPolicy(destination_id, destination_name, climate_factors,
                             parameters, policy_start, parameters['num_steps'],
                             mc, num_processes)

    Si = sobol_analyze.analyze(problem.problem, Y, print_to_console=True)
    sobol_df[destination_name] = Si['S1']

sobol_df.index = climate_factors
sobol_df = sobol_df.T

# Calculate mean and standard deviation of the sensitivity indices for each
# column
mean = sobol_df.mean(axis=0)
std = sobol_df.std(axis=0)

# Add mean and standard deviation to the dataframe as new rows
sobol_df.loc['mean'] = mean
sobol_df.loc['std'] = std

sobol_df = sobol_df.round(4)
os.makedirs(os.path.dirname(output_file), exist_ok=True)
sobol_df.to_csv(output_file)
