from airlift.envs.events.event_interval_generator import EventIntervalGenerator


class MalfunctionHandler:
    """
    Once an EventInterval is generated with "num broken steps" being greater than 0, the MalfunctionHandler takes
    care of keeping track of the "num broken steps". Once it reaches 0, the event is over.
    """
    def __init__(self, malfunction_generator: EventIntervalGenerator = None):
        self.malfunction_down_counter = 0
        self.num_malfunctions = 0
        self.malfunction_generator = malfunction_generator

    def reset(self):
        self.malfunction_down_counter = 0
        self.num_malfunctions = 0

    @property
    def in_malfunction(self):
        return self.malfunction_down_counter > 0

    @property
    def malfunction_counter_complete(self):
        return self.malfunction_down_counter == 0

    def step(self):
        # Only set new malfunction value if old malfunction is completed
        if self.malfunction_down_counter == 0:
            num_broken_steps = self.malfunction_generator.generate().num_broken_steps
            if num_broken_steps > 0:
                self.malfunction_down_counter = num_broken_steps
                self.num_malfunctions += 1
        # Let's avoid adding to the malfunction and decrementing right away. Decrement on the following step call
        elif self.malfunction_down_counter > 0:
            self.malfunction_down_counter -= 1

    def __repr__(self):
        return f"malfunction_down_counter: {self.malfunction_down_counter} \
                in_malfunction: {self.in_malfunction} \
                num_malfunctions: {self.num_malfunctions}"
