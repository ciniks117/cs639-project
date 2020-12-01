from .SetBasedAbstraction import SetBasedAbstraction
from .Star import Star


class StarAbstraction(SetBasedAbstraction):
    def __init__(self, confidence_fun, size=1, epsilon=0., epsilon_relative=True):
        super().__init__(confidence_fun, size, epsilon, epsilon_relative)

    def name(self):
        return "Star"

    def set_type(self):
        return Star
