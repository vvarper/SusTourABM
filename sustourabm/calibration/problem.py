import copy

import numpy as np
from jmetal.core.problem import IntegerProblem
from mesa.batchrunner import batch_run

from sustourabm.model import SustainableTourismModel
from sustourabm.statistics import SimulationStatistics
from sustourabm.utils import totuple


class ABMCalibrationProblem(IntegerProblem):
    def __init__(self, model_parameters, mc, history_by_destination,
                 num_processes, chromosome_map, real_ranges, int_ranges,
                 time_series_metric):

        # Generic problem parameters
        super().__init__()
        self.number_of_variables = len(chromosome_map)
        self.number_of_objectives = 1
        self.number_of_constraints = 0
        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['Share fitness']

        self.lower_bound = int_ranges[:, 0]
        self.upper_bound = int_ranges[:, 1]

        # Specific problem parameters
        self.chromosome_map = chromosome_map
        self.real_ranges = real_ranges
        self.factors = (real_ranges[:, 1] - real_ranges[:, 0]) / int_ranges[:,
                                                                 1]
        self.model_parameters = model_parameters
        self.model_parameters['seed'] = range(mc)
        self.num_processes = num_processes

        self.history_by_destination = history_by_destination.drop(
            columns='Year').to_numpy().T
        self.steps = history_by_destination['Year'] - 2000
        self.compute_time_series_fitness = time_series_metric

    def evaluate(self, solution):

        integer_solution = [round(x) for x in solution.variables]

        print("Evaluating solution: {}".format(integer_solution))

        solution_parameters = self.decode_solution(integer_solution)
        results = SimulationStatistics(
            batch_run(SustainableTourismModel, parameters=solution_parameters,
                      data_collection_period=1,
                      number_processes=self.num_processes, iterations=1),
            self.model_parameters['num_destinations'])

        results = results.raw_results
        results = results.loc[results['Step'].isin(self.steps)]

        solution.objectives[0] = self.compute_fitness(results)

        return solution

    def get_name(self) -> str:
        return "Sustainable Tourism ABM Calibration Problem"

    def create_solution_from_integers(self, integer_solution):
        solution = self.create_solution()
        solution.variables = integer_solution
        return solution

    def create_solution_from_reals(self, real_solution):
        solution = self.create_solution()
        solution.variables = self.encode_solution(real_solution)
        return solution

    def encode_solution(self, real_solution):
        integer_solution = (real_solution - self.real_ranges[:,
                                            0]) / self.factors

        return (np.rint(integer_solution)).astype(int)

    def decode_solution(self, integer_solution):
        complete_solution = copy.deepcopy(self.model_parameters)
        real_solution = self.integer_to_real_solution(integer_solution)

        for i in range(len(real_solution)):
            if isinstance(self.chromosome_map[i], tuple):

                new_tuple = np.array(
                    complete_solution[self.chromosome_map[i][0]][0])

                if (self.chromosome_map[i][
                    0] == 'state_by_destination_step_factor'):
                    destination = self.chromosome_map[i][1]
                    factor = self.chromosome_map[i][2]
                    new_tuple[destination, :, factor] = np.clip(
                        new_tuple[destination, :, factor] + real_solution[i],
                        0,
                        1)

                else:
                    new_tuple[self.chromosome_map[i][1]] = real_solution[i]

                complete_solution[self.chromosome_map[i][0]][0] = totuple(
                    new_tuple)

            else:
                complete_solution[self.chromosome_map[i]] = real_solution[i]

        return complete_solution

    def integer_to_real_solution(self, integer_solution):
        real_solution = self.real_ranges[:,
                        0] + integer_solution * self.factors

        return real_solution

    def compute_fitness(self, results):
        mc_fitnesses = np.empty(shape=len(self.model_parameters['seed']))

        for mc in self.model_parameters['seed']:
            dest_fitness = np.empty(
                shape=self.model_parameters['num_destinations'])
            results_mc = results.loc[results['seed'] == mc]
            for dest in range(self.model_parameters['num_destinations']):
                dest_fitness[dest] = self.compute_time_series_fitness(
                    self.history_by_destination[dest],
                    results_mc['Share ' + str(dest)].to_numpy())

            mc_fitnesses[mc] = dest_fitness.mean()

        return mc_fitnesses.mean()
