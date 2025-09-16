import numpy as np
from gym.utils import seeding
from matplotlib import pyplot as plt


class PoissonDistribution:
    def __init__(self, lambda_value):
        # The Lambda value determines the expected average rate of events occurring
        # Increase of Lamba values skews the distribution more to the right
        self.lambda_value = lambda_value
        self._np_random = None

        self.event_counts = []


    def seed(self, seed=None):
        self._np_random, seed = seeding.np_random(seed)

    # Generate a distribution
    def generate_samples(self, num_samples):
        # samples = np.random.poisson(self.lambda_value, num_samples)
        samples = self._np_random.poisson(self.lambda_value, num_samples)
        return samples

    # call this at each step in the routemap
    def generate_events(self):
        return self._np_random.poisson(self.lambda_value)

    # Just so we can see what the distribution looks like for testing/tuning purposes.
    def plot_distribution(self, num_samples=None, generated_samples=None):
        # If there is no generated samples passed, create some samples
        if generated_samples is None:
            samples = self.generate_samples(num_samples)
        # Otherwise plot the pre-generated samples.
        else:
            samples = generated_samples

        plt.figure(figsize=(8, 6))
        plt.hist(samples, bins=range(int(max(samples)) + 1), density=True, color='skyblue', edgecolor='black')
        plt.title(f'Poisson Distribution (λ = {self.lambda_value})')
        plt.xlabel('Number of Events')
        plt.ylabel('Probability')
        plt.grid(True)
        plt.show()

    # Num steps would be max_cycles, to ensure we have an number of events for each time-step?
    # Just a test of implementation details for our environment
    def simulate_time_steps(self, num_steps):
        for i in range(num_steps):
            num_events = self._np_random.poisson(self.lambda_value)
            self.event_counts.append(num_events)
            print(f"Time step: {i}, Number of events: {num_events}")
