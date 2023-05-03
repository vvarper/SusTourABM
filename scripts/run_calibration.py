import json
import random
import sys

import numpy as np
import pandas as pd
from jmetal.algorithm.multiobjective import GDE3
from jmetal.util.generator import InjectorGenerator
from jmetal.util.observer import BasicObserver
from jmetal.util.termination_criterion import StoppingByEvaluations

from sustourabm.calibration.algorithm import HillClimbing
from sustourabm.calibration.metrics import mape
from sustourabm.calibration.problem import ABMCalibrationProblem
from sustourabm.utils import totuple


def create_initial_solution(climate_preferences, calibrate_states,
                            calibrate_factors, variation_range):
    int_ranges = []
    real_ranges = []
    chromosome_map = []
    real_initial_solution = []

    # Climate factors order: [0 4 6 7 8 5 3 1 2 9]
    climate_preferences_order = np.argsort(climate_preferences)

    # 2. Destination states
    if calibrate_states:
        # Map: ('state_by_destination_step_factor', destination, factor)
        # Destination order: Canary, Balearic, Sicily, Crete, Sardinia
        destination_order = [0, 1, 7, 2, 8]

        state_parameters_map = [
            ('state_by_destination_step_factor', dest_order, factor_order) for
            dest_order in destination_order for factor_order in
            climate_preferences_order]

        num_states_parameters = len(destination_order) * len(
            climate_preferences_order)

        chromosome_map += state_parameters_map
        int_ranges += [(0, 1000) for _ in range(num_states_parameters)]
        real_ranges += [(-variation_range, variation_range) for _ in
                        range(num_states_parameters)]

        real_initial_solution += [0.0 for _ in range(num_states_parameters)]

    # 3. Climate preferences
    if calibrate_factors:
        climate_parameters_map = [('mean_tourist_preferences_by_factor', order)
                                  for order in climate_preferences_order]

        chromosome_map += climate_parameters_map
        int_ranges += [(0, 1000) for _ in climate_preferences]
        real_ranges += [((1 - variation_range) * preference,
                         (1 + variation_range) * preference) for preference in
                        climate_preferences[climate_preferences_order]]

        real_initial_solution += [preference for preference in
                                  climate_preferences[
                                      climate_preferences_order]]

    real_ranges = np.array(real_ranges)
    int_ranges = np.array(int_ranges)

    return real_initial_solution, real_ranges, int_ranges, chromosome_map


def get_best_solution(algorithm):
    # Print results ###########################################################

    front = algorithm.get_result()

    print("Algorithm: " + algorithm.get_name())
    print("Problem: " + problem.get_name())
    print("Computing time: " + str(algorithm.total_computing_time))

    if type(front) is not list:
        front = [front]

    for (i, solution) in enumerate(front):
        print(f"Solution {i}: Score {solution.objectives[0]}, variables: "
              f"{solution.variables}")

    # Find solution with best score ###########################################

    best_solution = front[0]
    best_score = best_solution.objectives[0]

    for solution in front:
        if solution.objectives[0] < best_score:
            best_score = solution.objectives[0]
            best_solution = solution

    print(
        f"\nBest solution: Score {best_score}, variables: "
        f"{best_solution.variables}")

    return best_solution


# Define input variables #######################################################

try:
    mc = int(sys.argv[1])
    num_agents = int(sys.argv[2])
    max_evaluations = int(sys.argv[3])
    calibrate_states = bool(int(sys.argv[4]))
    calibrate_factors = bool(int(sys.argv[5]))
    variation_range = float(sys.argv[6])
except IndexError:
    print("{0} <mc> <num_agents> <max_evaluations> "
          "<calibrate_states> <calibrate_factors> "
          "<variation_range>".format(sys.argv[0]))
    sys.exit(1)

instance_path = 'data/instances/instances_real/rcp85_instance.json'
history_path = 'data/base_data/history_arrivals.csv'
num_processes = mc
metric = mape
calibration_seed = 1024

population_size = 100
cr = 0.5
f = 0.5

# Read instance parameters #####################################################

with open(instance_path) as json_file:
    data = json_file.read()
    parameters = json.loads(data)["abm_parameters"]

parameters['num_steps'] = 20
parameters['num_tourists'] = num_agents
states = np.array(parameters['state_by_destination_step_factor'])
states = states[:, :20, :]
parameters['state_by_destination_step_factor'] = list(states)

for key in parameters:
    if type(parameters[key]) == list:
        parameters[key] = [totuple(parameters[key])]

# Load share history ###########################################################

history = pd.read_csv('data/base_data/history_arrivals.csv', index_col=0)
share_columns = [title for title in history.columns if
                 title.startswith('Share')]

history = history[['Year'] + share_columns]

# Define the calibration problem and the initial solution #####################

climate_preferences = np.array(
    parameters['mean_tourist_preferences_by_factor'][0], dtype=np.float32)

real_initial_solution, real_ranges, \
    int_ranges, chromosome_map = create_initial_solution(
    climate_preferences, calibrate_states, calibrate_factors, variation_range)

problem = ABMCalibrationProblem(parameters, mc, history, num_processes,
                                chromosome_map, real_ranges, int_ranges,
                                metric)

initial_solution = problem.create_solution_from_reals(real_initial_solution)

# 1. DE Optimizer #############################################################

de_algorithm = GDE3(problem=problem, population_size=population_size, cr=cr,
                    f=f,
                    termination_criterion=StoppingByEvaluations(
                        max_evaluations),
                    population_generator=InjectorGenerator([initial_solution]))

de_algorithm.observable.register(observer=BasicObserver(frequency=1))

# Run DE algorithm
random.seed(calibration_seed)
de_algorithm.run()
best_solution = get_best_solution(de_algorithm)

# 2. HC algorithm #############################################################

initial_solution = np.array([round(x) for x in best_solution.variables])
initial_solution = problem.create_solution_from_integers(initial_solution)
hc_algorithm = HillClimbing(problem=problem, max_evaluations=1000,
                            solution_generator=InjectorGenerator(
                                [initial_solution]))

hc_algorithm.observable.register(observer=BasicObserver(frequency=1))

# Run HC algorithm
hc_algorithm.run()

get_best_solution(hc_algorithm)
