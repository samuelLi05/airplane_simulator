from typing import List

PlaneTypeID = int


class PlaneType:
    """
    Manages Airplane Models and characteristics common to all airplanes of a given type.
    These can also be directly passing to the world generator when creating the environment.
    """

    def __init__(self, id: PlaneTypeID, max_range: float = 1, speed: float = 1, max_weight: float = 1,
                 model: str = None):
        self.id = id

        if model is None:
            self.model = 'A{}'.format(id)
        else:
            self.model = model

        self.max_range = max_range
        self.speed = speed
        self.max_weight = max_weight

    def __str__(self):
        return self.model

    def __repr__(self):
        return self.model
