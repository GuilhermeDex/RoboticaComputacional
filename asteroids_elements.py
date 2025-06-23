# asteroids_elements.py
import numpy as np

class Element:
    def __init__(self, name, position):
        self.name = name
        self.position = position  # (x, y)

    def distance_to(self, other):
        return np.linalg.norm(np.array(self.position) - np.array(other.position))

class Ship(Element):
    def __init__(self, position):
        super().__init__('Ship', position)

class Asteroid(Element):
    def __init__(self, position):
        super().__init__('Asteroid', position)