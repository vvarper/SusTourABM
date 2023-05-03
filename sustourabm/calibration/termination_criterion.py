from jmetal.util.termination_criterion import TerminationCriterion


class StoppingByEvaluationsOrLocalOptimum(TerminationCriterion):
    def __init__(self, max_evaluations: int):
        super(StoppingByEvaluationsOrLocalOptimum, self).__init__()
        self.max_evaluations = max_evaluations
        self.evaluations = 0
        self.local_optimum = False

    def update(self, *args, **kwargs):
        self.evaluations = kwargs["EVALUATIONS"]
        self.local_optimum = kwargs["LOCAL_OPTIMUM"]

    @property
    def is_met(self):
        return self.evaluations >= self.max_evaluations or self.local_optimum


class StoppingByIterations(TerminationCriterion):
    def __init__(self, max_iterations: int):
        super(StoppingByIterations, self).__init__()
        self.max_iterations = max_iterations
        self.iterations = 0

    def update(self, *args, **kwargs):
        self.iterations = kwargs["ITERATIONS"]

    @property
    def is_met(self):
        return self.iterations >= self.max_iterations
