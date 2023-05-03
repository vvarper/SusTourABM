import copy
import threading
import time

from jmetal.config import store
from jmetal.core.algorithm import Algorithm
from jmetal.core.observer import Observer
from jmetal.core.operator import Mutation
from jmetal.core.problem import Problem
from jmetal.core.solution import IntegerSolution
from jmetal.util.comparator import Comparator
from jmetal.util.generator import Generator, InjectorGenerator
from jmetal.util.observer import BasicObserver
from jmetal.util.termination_criterion import TerminationCriterion

from sustourabm.calibration.neighborhood_generator import \
    OneStepIntegerNeighborOperator
from sustourabm.calibration.termination_criterion import \
    StoppingByEvaluationsOrLocalOptimum


class HillClimbing(Algorithm[IntegerSolution, IntegerSolution],
                   threading.Thread):
    def __init__(self, problem: Problem[IntegerSolution], max_evaluations: int,
                 comparator: Comparator = store.default_comparator,
                 solution_generator: Generator = store.default_generator):

        super(HillClimbing, self).__init__()
        self.comparator = comparator
        self.problem = problem
        self.neighbor_operator = OneStepIntegerNeighborOperator()
        self.termination_criterion = StoppingByEvaluationsOrLocalOptimum(
            max_evaluations)
        self.observable.register(self.termination_criterion)
        self.solution_generator = solution_generator

    def create_initial_solutions(self):
        return [self.solution_generator.new(self.problem)]

    def evaluate(self, solutions):
        return [self.problem.evaluate(solutions[0])]

    def init_progress(self):
        self.neighbor_operator.reset(self.solutions[0])
        self.evaluations = 0

    def stopping_condition_is_met(self) -> bool:
        return self.termination_criterion.is_met

    def step(self):
        if not self.neighbor_operator.finished():
            neighbor_solution = self.neighbor_operator.generate_neighbor()
            neighbor_solution = self.evaluate([neighbor_solution])[0]

            result = self.comparator.compare(neighbor_solution,
                                             self.solutions[0])
            if result == -1:
                self.solutions[0] = neighbor_solution
                self.neighbor_operator.reset(self.solutions[0])
            else:
                self.neighbor_operator.move()

    def update_progress(self):
        self.evaluations += 1

        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)

    def get_observable_data(self) -> dict:
        ctime = time.time() - self.start_computing_time
        return {"PROBLEM": self.problem, "EVALUATIONS": self.evaluations,
                "SOLUTIONS": self.get_result(), "COMPUTING_TIME": ctime,
                "LOCAL_OPTIMUM": self.neighbor_operator.finished()}

    def get_result(self):
        return self.solutions[0]

    def get_name(self):
        return 'Hill Climbing'


class IteratedLocalSearch(Algorithm[IntegerSolution, IntegerSolution]):
    def __init__(self, problem: Problem[IntegerSolution],
                 mutation: Mutation[IntegerSolution],
                 local_search_evaluations: int,
                 local_search_observer: Observer = BasicObserver(frequency=1),
                 termination_criterion: TerminationCriterion =
                 store.default_termination_criteria,
                 comparator: Comparator = store.default_comparator,
                 solution_generator: Generator = store.default_generator):
        super(IteratedLocalSearch, self).__init__()
        self.iterations = None
        self.local_search = None
        self.comparator = comparator
        self.termination_criterion = termination_criterion
        self.observable.register(self.termination_criterion)
        self.solution_generator = solution_generator
        self.mutation = mutation
        self.problem = problem
        self.local_search_evaluations = local_search_evaluations
        self.local_search_observer = local_search_observer

    def create_initial_solutions(self):
        return [self.solution_generator.new(self.problem)]

    def evaluate(self, solutions):
        return [self.problem.evaluate(solutions[0])]

    def init_progress(self):
        self.run_local_search(self.solutions[0])
        self.solutions[0] = self.local_search.get_result()

        self.evaluations = self.local_search.evaluations
        self.iterations = 1

    def stopping_condition_is_met(self) -> bool:
        return self.termination_criterion.is_met

    def step(self):
        mutated_solution = copy.deepcopy(self.solutions[0])
        mutated_solution = self.mutation.execute(mutated_solution)

        self.run_local_search(mutated_solution)

        new_solution = self.local_search.get_result()

        result = self.comparator.compare(new_solution, self.solutions[0])

        if result == -1:
            self.solutions[0] = new_solution

    def run_local_search(self, solution: IntegerSolution):
        self.local_search = HillClimbing(problem=self.problem,
                                         max_evaluations=self.local_search_evaluations,
                                         solution_generator=InjectorGenerator(
                                             [solution]))

        self.local_search.observable.register(self.local_search_observer)
        self.local_search.run()

    def update_progress(self):
        self.evaluations += self.local_search.evaluations
        self.iterations += 1

        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)

    def get_observable_data(self) -> dict:
        ctime = time.time() - self.start_computing_time
        return {"PROBLEM": self.problem, "EVALUATIONS": self.evaluations,
                "SOLUTIONS": self.get_result(), "COMPUTING_TIME": ctime,
                "ITERATIONS": self.iterations}

    def get_result(self):
        return self.solutions[0]

    def get_name(self):
        return 'Iterated Local Search'
