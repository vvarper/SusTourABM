import copy

from jmetal.core.solution import IntegerSolution
from jmetal.util.ckecking import Check


class OneStepIntegerNeighborOperator:
    def __init__(self):
        self.original_solution = None
        self.current_parameter = 0
        self.current_move = 1

    def reset(self, solution: IntegerSolution):
        Check.that(type(solution) is IntegerSolution, "Solution type invalid")

        self.current_parameter = 0
        self.current_move = -1
        self.original_solution = solution

    def generate_neighbor(self):
        solution = copy.deepcopy(self.original_solution)

        while not self.finished():
            new_value = solution.variables[
                            self.current_parameter] + self.current_move
            if solution.lower_bound[self.current_parameter] <= new_value <= \
                    solution.upper_bound[self.current_parameter]:
                solution.variables[self.current_parameter] = new_value
                break
            else:
                self.move()

        return solution

    def move(self):
        self.current_move = -self.current_move
        if self.current_move == -1:
            self.current_parameter += 1

    def finished(self) -> bool:
        n = self.original_solution.number_of_variables
        return self.current_parameter == n or (
                self.current_parameter == n - 1 and self.current_move == 1 and
                self.original_solution.variables[n - 1] ==
                self.original_solution.upper_bound[n - 1])

    @staticmethod
    def get_name() -> str:
        return 'One-Step Integer Neighbor Operator'
