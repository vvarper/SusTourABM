import sys
import os

from sustourabm.runner import simulate
from sustourabm.util.io import load_model_instance_from_json

# Process arguments from command line #########################################

try:
    instance = sys.argv[1]
    mc = int(sys.argv[2])
    num_processes = int(sys.argv[3])
    track_agents = bool(int(sys.argv[4]))

except IndexError:
    print("{0} <instance> <mc> <num_processes> "
          "<track_agents>".format(sys.argv[0]))
    sys.exit(1)

instance_path = 'data/instances/' + instance + '.json'
base_output_folder = 'data/results/instances_output/'
instance_name = os.path.split(instance)[1].replace('_instance', '')
simulation_config = {'MC': mc, 'track_agents': track_agents}

parameters, climate_factors, destinations = load_model_instance_from_json(
    instance_path)

results = simulate(parameters, mc, num_processes, track_agents)

results.save(base_output_folder, instance_name, simulation_config)
