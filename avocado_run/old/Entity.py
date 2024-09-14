import numpy as np
import random


class Entity:
    """
    A class for an entity used in the AvocadoRun environment
    """

    def __init__(self, env_size, starting_position=None) -> None:
        if starting_position:
            self.x = starting_position[0]
            self.y = starting_position[1]
        else:
            self.x = np.random.randint(0, env_size)
            self.y = np.random.randint(0, env_size)
        self.env_size = env_size

    def __sub__(self, other):
        return (self.x - other.x, self.y - other.y)

    def __eq__(self, other) -> bool:
        return self.x == other.x and self.y == other.y

    def action(self, choice):
        if choice == 0:
            self.move(y=-1)  # Up
        elif choice == 1:
            self.move(x=1)  # Right
        elif choice == 2:
            self.move(y=1)  # Down
        elif choice == 3:
            self.move(x=-1)  # Left
        elif choice == 4:
            self.move()  # Do nothing

    def move(self, x=0, y=0):
        self.x = max(0, min(self.x + x, self.env_size - 1))
        self.y = max(0, min(self.y + y, self.env_size - 1))

    def move_towards_target(self, other):
        potential_directions = []

        if self.x < other.x:
            potential_directions.append(('x', 1))   # Move right
        elif self.x > other.x:
            potential_directions.append(('x', -1))  # Move left

        if self.y < other.y:
            potential_directions.append(('y', 1))   # Move down
        elif self.y > other.y:
            potential_directions.append(('y', -1))  # Move up

        if potential_directions:
            axis, delta = random.choice(potential_directions)
            if axis == 'x':
                self.move(x=delta)
            else:
                self.move(y=delta)

    def random_action(self):
        # Should always move, so action 4 is omitted
        rand_act = np.random.randint(0, 4)
        self.action(rand_act)
