import random
import numpy as np

class OU():
    def noise(self, x, mu, theta, sigma):
        return theta * (mu - x) + sigma * np.random.randn()

