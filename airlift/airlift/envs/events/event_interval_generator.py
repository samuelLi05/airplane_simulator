from typing import NamedTuple
from gym.utils import seeding

EventInterval = NamedTuple('EventInterval', [('num_broken_steps', int)])


class EventIntervalGenerator:
    """
    Generates an interval that a route can go offline. This generator can be used for making anything in the environment
    unavailable from min to max steps inclusive. At the moment it is only being utilized to manage routes becoming unavailable.
    This generator is based upon the Flatland Train Malfunction generator and uses poisson distribution as defined in the
    EventGenerator class.
    """
    def __init__(self, min_duration: int, max_duration: int):
        self.min_duration = min_duration
        self.max_duration = max_duration
        self._np_random = None
        self._randcache = []

    def seed(self, seed=None):
        self._np_random, seed = seeding.np_random(seed)

    def generate(self) -> EventInterval:
        """
        Generates an Event with an Interval from min duration to max duration

        :return: `EventInterval`: a NamedTuple that contains the number of steps something will become unavailable for. The number of
            steps is 0 of an interval wasn't generated.

        """

        if not self._randcache:
            self._randcache = list(self._np_random.integers(self.min_duration, self.max_duration, size=1000))
            num_broken_steps = self._randcache.pop() + 1

        else:
            num_broken_steps = self._randcache.pop() + 1

        # Keep returning the named tuple
        return EventInterval(num_broken_steps)

    @property
    def expected_mal_steps(self):
        return (self.max_duration - self.min_duration) / 2


class NoEventIntervalGen(EventIntervalGenerator):
    """
    Used when items that utilize the EventIntervalGenerator are toggled off.
    """
    def __init__(self):
        super().__init__(0, 0)

    def generate(self):
        return EventInterval(0)

    def expected_mal_steps(self):
        return 0