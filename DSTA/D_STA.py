from DSTA import Operators
import numpy as np
from util.functions import restrain_param
from DSTA.Operators import default_criterion


class Dsta(object):
    def __init__(self, param, fitness, old_set, SE):
        self.best_param = param
        self.best_solution = fitness(param, train=False)
        self.param = self.best_param
        self.solution = self.best_solution
        self.direct = np.zeros_like(param)
        self.memorize = np.zeros_like(param).astype('float64')
        self.old_set = old_set
        self.prob_recover = 0
        self.SE = SE

    def optimizer(self, fitness, criterion=default_criterion, cnn=None):
        operator_list = ['substitute', 'swap', 'shift', 'symmetry', ]
        for item in operator_list:
            self.param, self.solution, self.direct = Operators.operation(item, self.param,
                                                                         self.solution, fitness,
                                                                         self.old_set, self.SE, criterion, cnn)
            self.memorize += self.direct
        print('使用记忆算子')
        self.param, self.solution = Operators.operation_memo(self.param, self.solution, fitness, self.memorize,
                                                             self.old_set, criterion, cnn)
        if self.solution >= self.best_solution:
            self.best_param = self.param
            self.best_solution = self.solution


