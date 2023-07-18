import numpy as np

from sustourabm.runner import simulate
from sustourabm.util.io import to_tuple


class ClimaticPolicy:

    def __init__(self, destination_id, destination_name, factors_names,
                 instance_parameters, policy_start, policy_end, mc,
                 num_processes):
        self.destination_id = destination_id
        self.destination_name = destination_name
        self.factors_names = factors_names
        self.instance_parameters = instance_parameters
        self.policy_start = policy_start
        self.policy_end = policy_end
        self.mc = mc
        self.num_processes = num_processes
        self.problem = {'names': factors_names,
                        'bounds': [[0, 1]] * len(factors_names),
                        'num_vars': len(factors_names),
                        'outputs': ['Final share']}

    def evaluate(self, solution):
        policy_parameters = self.get_parameters_from_solution(solution)

        results = simulate(parameters=policy_parameters, mc=self.mc,
                           num_processes=self.num_processes)

        result = results.summary_results[
            'Share ' + str(self.destination_id), 'mean'].iloc[
            self.policy_end - 1]

        return result

    def get_parameters_from_solution(self, solution):
        policy_parameters = self.instance_parameters.copy()
        states = np.array(
            policy_parameters['state_by_destination_step_factor'][0])

        original_start_share = states[self.destination_id, self.policy_start]
        original_final_share = states[self.destination_id, self.policy_end - 1]
        deterioration = np.maximum(original_final_share - original_start_share,
                                   0)
        new_final_share = solution * deterioration + original_start_share

        states[self.destination_id, self.policy_start:self.policy_end] = \
            np.linspace(original_start_share,
                        new_final_share,
                        self.policy_end - self.policy_start)

        policy_parameters['state_by_destination_step_factor'] = [
            to_tuple(states)]

        return policy_parameters
