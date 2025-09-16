class RunningStat:
    def __init__(self):
        self.count = 0
        self.mean = None
        self.max = None
        self.min = None

    def update(self, new_value):
        """
        Updates the running statistics min/mean/max with the new value
        """
        if self.count == 0:
            self.mean = new_value
            self.min = new_value
            self.max = new_value
        else:
            self.mean = ((self.mean * self.count) + new_value) / (self.count + 1)
            self.min = min(new_value, self.min)
            self.max = max(new_value, self.max)

        self.count += 1