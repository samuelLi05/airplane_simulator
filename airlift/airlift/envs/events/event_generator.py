import numpy as np
from gym.utils import seeding


class EventGenerator:
    """
    Probability of a single agent to break at a given time step. According to Poisson process with given rate.
    Rate indicates average number of malfunctions per time step.
    The malfunction probability returned by this method indicates the probability that one or more malfunction events occur in the current time step.
    See https://en.wikipedia.org/wiki/Poisson_point_process
    """

    def __init__(self, rate: float):
        if rate <= 0:
            self._event_prob = 0.
        else:
            # The probability that no malfunctions occur (under the Poisson distribution) is exp(-rate), so the probability
            # that at least one malfunction event occurs is 1-exp(-rate)
            self._event_prob = 1 - np.exp(-rate)

        self._np_random = None
        self.randcache = []

    def seed(self, seed=None):
        self._np_random, seed = seeding.np_random(seed)

    def generate(self) -> bool:
        if not self.randcache:
            self.randcache = list(self._np_random.random(size=1000))
        return self.randcache.pop() < self._event_prob

    @property
    def event_prob(self):
        return self._event_prob
