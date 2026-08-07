"""
simulation_space.py

Defines the SimulationSpace class.

The SimulationSpace stores the complete state of the simulation including
the electromagnetic field, material properties, simulation clock and
grid information.

The WaveSolver updates the simulation using the public APIs provided by
this class.
"""

import numpy as np


class SimulationSpace:
    """
    Represents the two-dimensional simulation environment.
    """

    def __init__(
        self,
        width,
        height,
        resolution_x,
        resolution_y,
        dt,
    ):
        """
        Constructor Arguments
        ---------------------
        width           : Physical width of the simulation space.

        height          : Physical height of the simulation space.

        resolution_x    : Number of sample points along the x-axis.

        resolution_y    : Number of sample points along the y-axis.

        dt              : Simulation time step.
        """

        # Physical dimensions

        self.width = width
        self.height = height

        # Grid resolution

        self.resolution_x = resolution_x
        self.resolution_y = resolution_y

        # Physical spacing between neighbouring sample points

        self.dx = width / (resolution_x - 1)
        self.dy = height / (resolution_y - 1)

        # Simulation clock

        self.time = 0.0
        self.dt = dt

        # Simulation state

        self.running = False

        # Electromagnetic field storage

        self._current_field = np.zeros(
            (resolution_x, resolution_y),
            dtype=np.float64,
        )

        self._previous_field = np.zeros(
            (resolution_x, resolution_y),
            dtype=np.float64,
        )

        # Material properties

        self._wave_speed = np.full(
            (resolution_x, resolution_y),
            3.0e8,
            dtype=np.float64,
        )

        self._attenuation = np.zeros(
            (resolution_x, resolution_y),
            dtype=np.float64,
        )

    # Helper methods

    def _position_to_index(self, x, y):
        """
        Converts physical coordinates to grid indices.
        """

        i = int(round(x / self.dx))
        j = int(round(y / self.dy))

        return i, j


    def _index_to_position(self, i, j):
        """
        Converts grid indices to physical coordinates.
        """

        x = i * self.dx
        y = j * self.dy

        return x, y

    # Public API

    def get_field(self, x, y):
        """
        Returns the field value at the specified location.
        """

        if not self.is_inside(x, y):
            raise ValueError(
                f"Point ({x}, {y}) is outside the simulation space."
            )

        i, j = self._position_to_index(x, y)
        return self._current_field[i, j]

    def set_field(self, x, y, value):
        """
        Sets the field value at the specified location.
        """

        if not self.is_inside(x, y):
            raise ValueError(
                f"Point ({x}, {y}) is outside the simulation space."
            )

        i, j = self._position_to_index(x, y)
        self._current_field[i, j] = value

    def clear(self):
        """
        Resets every point in the simulation space.
        """

        self._current_field.fill(0.0)
        self._previous_field.fill(0.0)

    def is_inside(self, x, y):
        """
        Returns whether the specified location lies inside the simulation space.
        """

        return (
            0.0 <= x <= self.width
            and
            0.0 <= y <= self.height
        )

    # Simulation control

    def is_running(self):
        """
        Returns whether the simulation is currently running.
        """

        return self.running

    def set_running(self, running):
        """
        Updates the simulation running state.
        """

        self.running = running

    # Simulation clock

    def advance_time(self):
        """
        Advances the simulation clock by one time step.
        """

        self.time += self.dt

    def set_time(self, time):
        """
        Sets the current simulation time.
        """

        self.time = time

    # Material properties

    def get_wave_speed(self, x, y):
        """
        Returns the wave propagation speed at the specified location.
        """

        if not self.is_inside(x, y):
            raise ValueError(
                f"Point ({x}, {y}) is outside the simulation space."
            )

        i, j = self._position_to_index(x, y)
        return self._wave_speed[i, j]

    def set_wave_speed(self, x, y, value):
        """
        Sets the wave propagation speed at the specified location.
        """

        if not self.is_inside(x, y):
            raise ValueError(
                f"Point ({x}, {y}) is outside the simulation space."
            )

        i, j = self._position_to_index(x, y)
        self._wave_speed[i, j] = value

    def get_attenuation(self, x, y):
        """
        Returns the attenuation coefficient at the specified location.
        """

        if not self.is_inside(x, y):
            raise ValueError(
                f"Point ({x}, {y}) is outside the simulation space."
            )

        i, j = self._position_to_index(x, y)
        return self._attenuation[i, j]

    def set_attenuation(self, x, y, value):
        """
        Sets the attenuation coefficient at the specified location.
        """

        if not self.is_inside(x, y):
            raise ValueError(
                f"Point ({x}, {y}) is outside the simulation space."
            )
        i, j = self._position_to_index(x, y)
        self._attenuation[i, j] = value

    def set_global_wave_speed(self, value):
        """
        Sets the wave propagation speed throughout the simulation space.
        """

        self._wave_speed.fill(value)


    def set_wave_speed_rectangle(
        self,
        x1,
        y1,
        x2,
        y2,
        value,
    ):
        """
        Sets the wave propagation speed inside a rectangular region.
        """
        
        if x1 > x2 or y1 > y2:
            raise ValueError(
                "Rectangle coordinates are invalid."
            )
        if (
            not self.is_inside(x1, y1)
            or
            not self.is_inside(x2, y2)
        ):
            raise ValueError(
                "Rectangle is outside the simulation space."
            )

        i1, j1 = self._position_to_index(x1, y1)
        i2, j2 = self._position_to_index(x2, y2)

        self._wave_speed[
            i1:i2+1,
            j1:j2+1,
        ] = value


    def set_global_attenuation(self, value):
        """
        Sets the attenuation coefficient throughout the simulation space.
        """

        self._attenuation.fill(value)


    def set_attenuation_rectangle(
        self,
        x1,
        y1,
        x2,
        y2,
        value,
    ):
        """
        Sets the attenuation coefficient inside a rectangular region.
        """

        if x1 > x2 or y1 > y2:
            raise ValueError(
                "Rectangle coordinates are invalid."
            )

        if (
            not self.is_inside(x1, y1)
            or
            not self.is_inside(x2, y2)
        ):
            raise ValueError(
                "Rectangle is outside the simulation space."
            )

        i1, j1 = self._position_to_index(x1, y1)
        i2, j2 = self._position_to_index(x2, y2)

        self._attenuation[
            i1:i2+1,
            j1:j2+1,
        ] = value

    # WaveSolver interface

    def get_current_field(self):
        """
        Returns the current electromagnetic field.

        Intended for use by the WaveSolver.
        """

        return self._current_field

    def get_previous_field(self):
        """
        Returns the previous electromagnetic field.

        Intended for use by the WaveSolver.
        """

        return self._previous_field

    def get_wave_speed_map(self):
        """
        Returns the wave propagation speed map.

        Intended for use by the WaveSolver.
        """

        return self._wave_speed

    def get_attenuation_map(self):
        """
        Returns the attenuation coefficient map.

        Intended for use by the WaveSolver.
        """

        return self._attenuation

    def set_next_field(self, next_field):
        """
        Updates the electromagnetic field to the next simulation time step.
        """

        self._previous_field[:, :] = self._current_field
        self._current_field[:, :] = next_field
        
    def reset(self):
        """
        Resets the complete simulation state.
        """

        self.clear()
        self.time = 0.0
        self.running = False
