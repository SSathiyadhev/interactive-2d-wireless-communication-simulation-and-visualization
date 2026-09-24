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

        NOTE: if an absorbing boundary has been applied via
        set_absorbing_boundary(), calling this afterward will
        overwrite it (fill() replaces every cell unconditionally).
        Call set_global_attenuation() first, then
        set_absorbing_boundary(), if you want both.
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

    def set_absorbing_boundary(self, thickness_cells, max_attenuation):
        """
        Applies a tapered attenuation layer near all four edges of the
        grid, approximating an open (non-reflecting) boundary.

        This is a simplified alternative to a full Perfectly Matched
        Layer (PML). Attenuation increases smoothly (quadratically,
        not abruptly) from 0 at the inner edge of the layer to
        max_attenuation at the outermost cells. The taper matters:
        an abrupt jump in attenuation itself causes a partial
        reflection at the transition, which would partly defeat the
        purpose.

        Arguments
        ---------
        thickness_cells : Width of the absorbing layer, in grid
                          cells, measured inward from each edge.
        max_attenuation : Attenuation coefficient at the outermost
                          cells (same units as set_attenuation/
                          set_global_attenuation). Tune this
                          empirically -- too small and reflections
                          are barely reduced; too large relative to
                          dt can itself cause instability or
                          reflection at the layer's inner edge.

        This affects the SAME attenuation map used by
        set_global_attenuation() / set_attenuation_rectangle() --
        it takes the elementwise maximum with whatever is already
        set, so it will not reduce attenuation you set elsewhere
        (e.g. inside an obstacle), only add the boundary taper on
        top of it. Call this AFTER set_global_attenuation() /
        set_attenuation_rectangle(), not before, since those
        methods overwrite rather than combine.
        """

        thickness_cells = int(thickness_cells)

        if thickness_cells <= 0:
            raise ValueError(
                "thickness_cells must be a positive integer."
            )

        if thickness_cells > min(self.resolution_x, self.resolution_y) // 2:
            raise ValueError(
                "thickness_cells is too large for this grid -- the "
                "two boundary layers would overlap in the middle."
            )

        ix = np.arange(self.resolution_x)
        iy = np.arange(self.resolution_y)

        # Distance (in cells) from each index to the nearest edge
        # along that axis.
        distance_x = np.minimum(ix, self.resolution_x - 1 - ix)
        distance_y = np.minimum(iy, self.resolution_y - 1 - iy)

        distance_x_grid, distance_y_grid = np.meshgrid(
            distance_x, distance_y, indexing="ij",
        )

        distance_to_edge = np.minimum(distance_x_grid, distance_y_grid)

        # 0 at the layer's inner boundary (and everywhere further
        # inward, clipped), rising to 1 at the true edge.
        depth_ratio = np.clip(
            (thickness_cells - distance_to_edge) / thickness_cells,
            0.0,
            1.0,
        )

        boundary_attenuation = max_attenuation * depth_ratio ** 2

        self._attenuation = np.maximum(
            self._attenuation,
            boundary_attenuation,
        )

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