import os
import sys

import numpy as np
from SALib.analyze import sobol as sobol_analyze
from SALib.sample import sobol as sobol_sample
from matplotlib import pyplot as plt

from sustourabm.climatic_policy.climatic_policy import ClimaticPolicy
from sustourabm.util.io import load_model_instance_from_json

# Process arguments from command line #########################################

try:
    instance = sys.argv[1]
    destination_id = int(sys.argv[2])
    n = int(sys.argv[3])
    policy_start = int(sys.argv[4])
    mc = int(sys.argv[5])
    num_processes = int(sys.argv[6])
    num_agents = int(sys.argv[7])

except IndexError:
    print(
        "{0} <instance> <destination_id> <n> <start> <mc> <num_processes> "
        "<num_agents>".format(sys.argv[0]))
    sys.exit(1)

# Example from SALib ##########################################################

seed = 121
instance_path = 'data/instances/' + instance + '.json'
instance_name = os.path.split(instance)[1].replace('_instance', '')
parameters, climate_factors, destinations = load_model_instance_from_json(
    instance_path)
parameters['num_tourists'] = num_agents

problem = ClimaticPolicy(destination_id, destinations[destination_id],
                         climate_factors, parameters, policy_start,
                         parameters['num_steps'], mc, num_processes)

# Generate parameter samples ##################################################

param_values = sobol_sample.sample(problem.problem, n, seed=seed)
sample_output = f'data/sensitivity_analysis/sample_params_n{n}.txt'
os.makedirs(os.path.dirname(sample_output), exist_ok=True)
np.savetxt(sample_output, param_values)

# Evaluate the parameters of the sample #######################################

Y = np.zeros([param_values.shape[0]])
for i, X in enumerate(param_values):
    Y[i] = problem.evaluate(X)

eval_output = f'data/sensitivity_analysis/sample_evals_n{n}_{instance_name}' \
              f'_{destinations[destination_id]}_{policy_start}' \
              f'_{mc}_{num_processes}.txt'
os.makedirs(os.path.dirname(eval_output), exist_ok=True)
np.savetxt(eval_output, Y)

# Perform the sensitivity analysis ############################################

Si = sobol_analyze.analyze(problem.problem, Y, print_to_console=True)

Si.plot()
plot_output = f'data/sensitivity_analysis/sobol_n{n}_{instance_name}' \
              f'_{destinations[destination_id]}_{policy_start}' \
              f'_{mc}_{num_processes}.png'
os.makedirs(os.path.dirname(plot_output), exist_ok=True)
plt.savefig(plot_output)
