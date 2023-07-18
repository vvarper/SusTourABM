import random
import sys

import numpy as np
from jmetal.algorithm.multiobjective import GDE3
from jmetal.util.generator import InjectorGenerator
from jmetal.util.observer import BasicObserver
from jmetal.util.termination_criterion import StoppingByEvaluations

from sustourabm.calibration.algorithm_solutions_explorer import \
    get_best_solution
from sustourabm.calibration.chromosome_map import \
    define_factors_5_dest_states_map
from sustourabm.calibration.metrics import mape
from sustourabm.calibration.optimization.algorithm import HillClimbing
from sustourabm.calibration.problem import ABMCalibrationProblem
from sustourabm.util.io import to_tuple, load_history_from_csv, \
    load_model_instance_from_json, save_dict2json

# Process arguments from command line #########################################

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

# Define input variables ######################################################

calibrating_instance_path = 'data/instances/original/rcp85_instance.json'
history_path = 'data/base_data/history_arrivals.csv'
num_processes = mc
metric = mape
calibration_seed = 1024

population_size = 100
cr = 0.5
f = 0.5
hc_evaluations = 1000

# Read instance parameters ####################################################

parameters, climate_factors, destinations = load_model_instance_from_json(
    calibrating_instance_path)

parameters['num_steps'] = 20
parameters['num_tourists'] = num_agents
states = np.array(parameters['state_by_destination_step_factor'][0])[:, :20, :]
parameters['state_by_destination_step_factor'] = [to_tuple(states)]

# Load share history ##########################################################

history = load_history_from_csv(history_path)

# Define the calibration problem and the initial solution #####################

chromosome_map, real_ranges, int_ranges = define_factors_5_dest_states_map(
    np.array(parameters['mean_tourist_preferences_by_factor'][0],
             dtype=np.float32), calibrate_states, calibrate_factors,
    variation_range)

problem = ABMCalibrationProblem(parameters, mc, history, num_processes,
                                chromosome_map, real_ranges, int_ranges,
                                metric)

initial_solution = problem.create_solution_from_integers(
    [500 for _ in int_ranges])

# 1. DE Optimizer #############################################################

de_algorithm = GDE3(problem=problem, population_size=population_size, cr=cr,
                    f=f, termination_criterion=StoppingByEvaluations(
        max_evaluations),
                    population_generator=InjectorGenerator([initial_solution]))

de_algorithm.observable.register(observer=BasicObserver(frequency=1))

# Run DE algorithm
random.seed(calibration_seed)
de_algorithm.run()
best_solution = get_best_solution(de_algorithm, problem.get_name())

# 2. HC algorithm #############################################################

initial_solution = np.array([round(x) for x in best_solution.variables])
initial_solution = problem.create_solution_from_integers(initial_solution)
hc_algorithm = HillClimbing(problem=problem, max_evaluations=hc_evaluations,
                            solution_generator=InjectorGenerator(
                                [initial_solution]))

hc_algorithm.observable.register(observer=BasicObserver(frequency=1))

# Run HC algorithm
hc_algorithm.run()

final_solution = get_best_solution(hc_algorithm, problem.get_name())

# 3. Save solution as new instances ###########################################

calibration_info = {'mc': mc, 'num_agents': num_agents,
                    'max_evaluations': max_evaluations,
                    'calibrate_states': calibrate_states,
                    'calibrate_factors': calibrate_factors,
                    'variation_range': variation_range,
                    'fitness': final_solution.objectives[0],
                    'integer_solution': [int(x) for x in
                                         final_solution.variables]}

for instance in ['rcp26_instance', 'rcp85_instance']:
    instance_path = f'data/instances/original/{instance}.json'
    parameters, climate_factors, destinations = load_model_instance_from_json(
        instance_path)

    problem = ABMCalibrationProblem(parameters, mc, history, num_processes,
                                    chromosome_map, real_ranges, int_ranges,
                                    metric)

    calibration_info['real_solution'] = list(
        problem.integer_to_real_solution(final_solution.variables))

    real_calibrated_parameters = problem.decode_solution(
        final_solution.variables)

    for key in real_calibrated_parameters:
        if type(real_calibrated_parameters[key]) == list:
            real_calibrated_parameters[key] = real_calibrated_parameters[key][
                0]

    calibrated_instance_data = {'climate_factors': climate_factors,
                                'destinations': destinations,
                                'abm_parameters': real_calibrated_parameters,
                                'calibration_info': calibration_info}

    if calibrate_factors:
        solution_path = f'data/instances/calibrated/calibrated_{instance}.json'
    else:
        solution_path = f'data/instances/calibrated_onlystates/calibrated' \
                        f'_{instance}_onlystates.json'

    save_dict2json(calibrated_instance_data, solution_path, indent=4,
                   sort_keys=True)
