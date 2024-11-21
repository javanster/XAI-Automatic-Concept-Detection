import numpy as np


class Entity:
    """
    A class for an entity used in the AvocadoRun environment
    """

    def __init__(self, grid_side_height, grid_side_width, starting_position=None) -> None:
        self._starting_position_validation(
            grid_side_height=grid_side_height, grid_side_width=grid_side_width, starting_position=starting_position)

        if starting_position:
            self.x = starting_position[0]
            self.y = starting_position[1]
        else:
            self.x = np.random.randint(0, grid_side_width)
            self.y = np.random.randint(0, grid_side_height)
        self.grid_side_height = grid_side_height
        self.grid_side_width = grid_side_width

    def _starting_position_validation(self, grid_side_height, grid_side_width, starting_position=None):
        if not starting_position:
            return
        if any(starting_coord < 0 for starting_coord in starting_position):
            raise ValueError(
                "A starting coordinate may not be below 0")
        if starting_position[0] >= grid_side_width or starting_position[1] >= grid_side_height:
            raise ValueError(
                "A starting coordinate may not exceed the environment size")

    def __sub__(self, other):
        return (self.x - other.x, self.y - other.y)

    def __eq__(self, other):
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

    def move(self, x=0, y=0):
        self.x = max(0, min(self.x + x, self.grid_side_width - 1))
        self.y = max(0, min(self.y + y, self.grid_side_height - 1))
