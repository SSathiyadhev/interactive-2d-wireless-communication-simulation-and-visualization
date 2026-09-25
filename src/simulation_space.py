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


EPSILON_0 = 8.8541878128e-12
MU_0 = 4.0e-7 * np.pi
C_0 = 3e8


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
        dt_stability_multiplier,
        absorbing_layer_thickness=3.0
    ):
        """
        Constructor Arguments
        ---------------------
        width                     : Physical width of the simulation space.

        height                    : Physical height of the simulation space.

        resolution_x              : Number of sample points along the x-axis.

        resolution_y              : Number of sample points along the y-axis.

        dt_stability_multiplier : Multiplier used to determine the
                                   simulation time step.

        absorbing_layer_thickness   : Thickness of the absorbing layer in meters.
        """

        # Physical dimensions of the user simulation domain

        self.width = float(width)
        self.height = float(height)

        # Absorbing boundary thickness

        self.absorbing_layer_thickness = float(
            absorbing_layer_thickness
        )

        if self.absorbing_layer_thickness < 0.0:
            raise ValueError(
                "Absorbing layer thickness cannot be negative."
            )

        # Computational domain dimensions
        #
        # The user still works with:
        #     0 <= x <= width
        #     0 <= y <= height
        #
        # The additional region exists only internally.

        self.computational_width = (
            self.width
            + 2.0 * self.absorbing_layer_thickness
        )

        self.computational_height = (
            self.height
            + 2.0 * self.absorbing_layer_thickness
        )

        # Grid resolution

        self.resolution_x = resolution_x
        self.resolution_y = resolution_y

        # Physical spacing between neighbouring sample points

        self.dx = (
            self.computational_width
            / (resolution_x - 1)
        )

        self.dy = (
            self.computational_height
            / (resolution_y - 1)
        )

        # Simulation clock

        self.time = 0.0

        self.dt = dt_stability_multiplier / (
            C_0 * np.sqrt(
                (1.0 / self.dx**2)
                +
                (1.0 / self.dy**2)
            )
        )

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

        # ---------------------------------------------------------
        # Material properties
        #
        # ε(x,y) : Permittivity
        # σ(x,y) : Electrical conductivity
        # μ(x,y) : Permeability
        #
        # Default material = vacuum
        # ---------------------------------------------------------

        self._epsilon = np.full(
            (resolution_x, resolution_y),
            EPSILON_0,
            dtype=np.float64,
        )

        self._sigma = np.zeros(
            (resolution_x, resolution_y),
            dtype=np.float64,
        )

        self._initialize_absorbing_layer()

        self._mu = np.full(
            (resolution_x, resolution_y),
            MU_0,
            dtype=np.float64,
        )

    # =============================================================
    # Helper methods (Corrected with Y-axis Inversion for Canvas Alignment)
    # =============================================================

    def _position_to_index(self, x, y):
        """
        Converts physical coordinates to grid indices with inverted y-axis 
        to match the frontend canvas coordinate orientation.
        """

        i = int(
            round(
                (x + self.absorbing_layer_thickness)
                / self.dx
            )
        )

        j = int(
            round(
                (
                    self.height
                    + self.absorbing_layer_thickness
                    - y
                )
                / self.dy
            )
        )

        return i, j

    def _index_to_position(self, i, j):
        """
        Converts grid indices to physical coordinates with inverted y-axis.
        """

        x = (
            i * self.dx
            - self.absorbing_layer_thickness
        )

        y = (
            self.height
            + self.absorbing_layer_thickness
            - j * self.dy
        )

        return x, y

    # =============================================================
    # Public API
    # =============================================================

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
        Returns whether the specified location lies inside
        the simulation space.
        """

        return (
            0.0 <= x <= self.width
            and
            0.0 <= y <= self.height
        )

    # =============================================================
    # Simulation control
    # =============================================================

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

    # =============================================================
    # Simulation clock
    # =============================================================

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

    # =============================================================
    # Material properties
    # =============================================================

    def get_permittivity(self, x, y):
        """
        Returns the permittivity at the specified location.
        """

        if not self.is_inside(x, y):
            raise ValueError(
                f"Point ({x}, {y}) is outside the simulation space."
            )

        i, j = self._position_to_index(x, y)
        return self._epsilon[i, j]

    def set_permittivity(self, x, y, value):
        """
        Sets the permittivity at the specified location.
        """

        if not self.is_inside(x, y):
            raise ValueError(
                f"Point ({x}, {y}) is outside the simulation space."
            )

        value = float(value)

        if value < EPSILON_0:
            raise ValueError(
                "Permittivity must be greater than or equal to "
                "vacuum permittivity."
            )

        i, j = self._position_to_index(x, y)
        self._epsilon[i, j] = value

    def get_conductivity(self, x, y):
        """
        Returns the electrical conductivity at the specified location.
        """

        if not self.is_inside(x, y):
            raise ValueError(
                f"Point ({x}, {y}) is outside the simulation space."
            )

        i, j = self._position_to_index(x, y)
        return self._sigma[i, j]

    def set_conductivity(self, x, y, value):
        """
        Sets the electrical conductivity at the specified location.
        """

        if not self.is_inside(x, y):
            raise ValueError(
                f"Point ({x}, {y}) is outside the simulation space."
            )

        value = float(value)

        if value < 0.0:
            raise ValueError(
                "Conductivity cannot be negative."
            )

        i, j = self._position_to_index(x, y)
        self._sigma[i, j] = value

    def get_permeability(self, x, y):
        """
        Returns the permeability at the specified location.
        """

        if not self.is_inside(x, y):
            raise ValueError(
                f"Point ({x}, {y}) is outside the simulation space."
            )

        i, j = self._position_to_index(x, y)
        return self._mu[i, j]

    def set_permeability(self, x, y, value):
        """
        Sets the permeability at the specified location.
        """

        if not self.is_inside(x, y):
            raise ValueError(
                f"Point ({x}, {y}) is outside the simulation space."
            )

        value = float(value)

        if value < MU_0:
            raise ValueError(
                "Permeability must be greater than or equal to "
                "vacuum permeability."
            )

        i, j = self._position_to_index(x, y)
        self._mu[i, j] = value

    # =============================================================
    # Global material properties
    # =============================================================

    def set_global_permittivity(self, value):
        """
        Sets the permittivity throughout the simulation space.
        """

        value = float(value)

        if value < EPSILON_0:
            raise ValueError(
                "Permittivity must be greater than or equal to "
                "vacuum permittivity."
            )

        self._epsilon.fill(value)

    def set_global_conductivity(self, value):
        """
        Sets the electrical conductivity throughout the simulation space.
        """

        value = float(value)

        if value < 0.0:
            raise ValueError(
                "Conductivity cannot be negative."
            )

        self._sigma.fill(value)

    def set_global_permeability(self, value):
        """
        Sets the permeability throughout the simulation space.
        """

        value = float(value)

        if value < MU_0:
            raise ValueError(
                "Permeability must be greater than or equal to "
                "vacuum permittivity."
            )

        self._mu.fill(value)

    # =============================================================
    # Rectangular material regions
    # =============================================================

    def set_permittivity_rectangle(
        self,
        x1,
        y1,
        x2,
        y2,
        value,
    ):
        """
        Sets the permittivity inside a rectangular region.
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

        value = float(value)

        if value < EPSILON_0:
            raise ValueError(
                "Permittivity must be greater than or equal to "
                "vacuum permittivity."
            )

        i1, j1 = self._position_to_index(x1, y1)
        i2, j2 = self._position_to_index(x2, y2)

        j_min, j_max = min(j1, j2), max(j1, j2)
        i_min, i_max = min(i1, i2), max(i1, i2)

        self._epsilon[
            i_min:i_max + 1,
            j_min:j_max + 1,
        ] = value

    def set_conductivity_rectangle(
        self,
        x1,
        y1,
        x2,
        y2,
        value,
    ):
        """
        Sets the electrical conductivity inside a rectangular region.
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

        value = float(value)

        if value < 0.0:
            raise ValueError(
                "Conductivity cannot be negative."
            )

        i1, j1 = self._position_to_index(x1, y1)
        i2, j2 = self._position_to_index(x2, y2)

        j_min, j_max = min(j1, j2), max(j1, j2)
        i_min, i_max = min(i1, i2), max(i1, i2)

        self._sigma[
            i_min:i_max + 1,
            j_min:j_max + 1,
        ] = value

    def set_permeability_rectangle(
        self,
        x1,
        y1,
        x2,
        y2,
        value,
    ):
        """
        Sets the permeability inside a rectangular region.
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

        value = float(value)

        if value < MU_0:
            raise ValueError(
                "Permeability must be greater than or equal to "
                "vacuum permittivity."
            )

        i1, j1 = self._position_to_index(x1, y1)
        i2, j2 = self._position_to_index(x2, y2)

        j_min, j_max = min(j1, j2), max(j1, j2)
        i_min, i_max = min(i1, i2), max(i1, i2)

        self._mu[
            i_min:i_max + 1,
            j_min:j_max + 1,
        ] = value

    # =============================================================
    # WaveSolver interface
    # =============================================================

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

    def get_permittivity_map(self):
        """
        Returns the spatial permittivity map.

        Intended for use by the WaveSolver.
        """

        return self._epsilon

    def get_conductivity_map(self):
        """
        Returns the spatial conductivity map.

        Intended for use by the WaveSolver.
        """

        return self._sigma

    def get_permeability_map(self):
        """
        Returns the spatial permeability map.

        Intended for use by the WaveSolver.
        """

        return self._mu

    def set_next_field(self, next_field):
        """
        Updates the electromagnetic field to the next simulation time step.
        """

        self._previous_field[:, :] = self._current_field
        self._current_field[:, :] = next_field

    # =============================================================
    # Reset
    # =============================================================

    def reset_materials(self):
        """
        Resets material properties to vacuum while
        preserving the absorbing boundary layer.
        """

        self._epsilon.fill(EPSILON_0)
        self._mu.fill(MU_0)
        self._sigma.fill(0.0)

        self._initialize_absorbing_layer()

    def reset(self):
        """
        Resets the complete simulation state.
        """

        self.clear()
        self.reset_materials()
        self.time = 0.0
        self.running = False

    def _initialize_absorbing_layer(self):
        """
        Initializes a graded conductivity profile in the
        absorbing layer surrounding the user simulation domain.
        """

        layer = self.absorbing_layer_thickness

        if layer <= 0.0:
            return

        # Internal computational coordinates.
        #
        # x: -layer ... width + layer
        # y: -layer ... height + layer

        x = (
            np.arange(self.resolution_x) * self.dx
            - layer
        )

        y = (
            self.height
            + layer
            - np.arange(self.resolution_y) * self.dy
        )

        # Create 2-D coordinate grids.
        #
        # X[i,j] and Y[i,j] give the physical coordinate
        # of every individual grid point.

        X, Y = np.meshgrid(
            x,
            y,
            indexing="ij"
        )

        # Distance from each grid point to the four
        # boundaries of the user simulation domain.

        distance_to_left = X
        distance_to_right = self.width - X
        distance_to_bottom = Y
        distance_to_top = self.height - Y

        # Distance to the nearest of the four boundaries.

        distance_to_user_domain = np.minimum.reduce(
            [
                distance_to_left,
                distance_to_right,
                distance_to_bottom,
                distance_to_top,
            ]
        )

        # Normalized depth inside the absorbing layer.
        #
        # 0 -> boundary of user domain
        # 1 -> outer computational boundary

        normalized_depth = np.clip(
            -distance_to_user_domain / layer,
            0.0,
            1.0,
        )

        # Maximum conductivity of the absorbing layer.

        sigma_max = 0.15

        # Quadratic grading.

        self._sigma[:, :] = (
            sigma_max
            * normalized_depth**2
        )