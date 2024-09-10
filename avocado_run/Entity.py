import numpy as np


class Entity:
    """
    A class for an entity used in the AvocadoRun environment
    """

    def __init__(self, env_size) -> None:
        self.x = np.random.randint(0, env_size)
        self.y = np.random.randint(0, env_size)
        self.env_size = env_size

    def __sub__(self, other):
        return (self.x - other.x, self.y - other.y)

    def __eq__(self, other) -> bool:
        return self.x == other.x and self.y == other.y

    def action(self, choice):
        if choice == 0:
            self.move(y=1)  # Up
        elif choice == 1:
            self.move(x=1)  # Right
        elif choice == 2:
            self.move(y=-1)  # Down
        elif choice == 3:
            self.move(x=-1)  # Left
        elif choice == 4:
            self.move()  # Do nothing

    def move(self, x=0, y=0):
        self.x = max(0, min(self.x + x, self.env_size - 1))
        self.y = max(0, min(self.y + y, self.env_size - 1))

    def move_towards_target(self, other):
        if self.x < other.x:
            self.x += 1
        elif self.x > other.x:
            self.x -= 1

        if self.y < other.y:
            self.y += 1
        elif self.y > other.y:
            self.y -= 1

    def random_action(self):
        # Should always move, so action 4 is omitted
        rand_act = np.random.randint(0, 4)
        self.action(rand_act)
