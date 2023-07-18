import random

from jmetal.core.operator import Mutation
from jmetal.core.solution import IntegerSolution
from jmetal.util.ckecking import Check


class OneStepIntegerMutation(Mutation[IntegerSolution]):
    def __init__(self, probability: float):
        super(OneStepIntegerMutation, self).__init__(probability=probability)

    def execute(self, solution: IntegerSolution) -> IntegerSolution:
        Check.that(type(solution) is IntegerSolution, "Solution type invalid")

        num_mutations = int(self.probability * solution.number_of_variables)
        parameters_choice = random.sample(range(solution.number_of_variables),
                                          num_mutations)

        for i in parameters_choice:
            variations = []

            if solution.variables[i] > solution.lower_bound[i]:
                variations.append(solution.variables[i] - 1)
            if solution.variables[i] < solution.upper_bound[i]:
                variations.append(solution.variables[i] + 1)

            solution.variables[i] = random.choice(variations)

        return solution

    def get_name(self) -> str:
        return 'One-Step Integer Mutation'
