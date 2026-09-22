"""
materials.py

Defines the Material class.

The Material class represents a material occupying a rectangular
region of the SimulationSpace.

Each material is described by three electromagnetic properties:

    relative_permittivity  : εr
    relative_permeability  : μr
    conductivity           : σ [S/m]

From these properties the class calculates:

    wave_speed             : propagation speed [m/s]
    attenuation            : damping coefficient [1/s]

The class supports:

    1. Built-in materials:
           - concrete
           - glass
           - wood

    2. Custom materials:
           Any unknown material name can be used if the user provides
           εr, μr and σ.

The calculated wave speed and attenuation are written into the
corresponding rectangular region of SimulationSpace.
"""

import numpy as np


class Material:
    """
    Represents an electromagnetic material occupying a rectangular
    region of the simulation space.
    """

    # ------------------------------------------------------------------
    # Physical constants
    # ------------------------------------------------------------------

    # Permittivity of free space [F/m]
    EPSILON_0 = 8.8541878128e-12

    # Permeability of free space [H/m]
    MU_0 = 4.0e-7 * np.pi

    # Speed of light in vacuum [m/s]
    C_0 = 1.0 / np.sqrt(MU_0 * EPSILON_0)

    # ------------------------------------------------------------------
    # Built-in material database
    #
    # Values are representative simulation values, not exact universal
    # constants. Real materials vary with composition, moisture,
    # temperature and frequency.
    #
    # Keys are stored in lowercase so material names can be entered
    # case-insensitively.
    # ------------------------------------------------------------------

    MATERIAL_DATABASE = {

        "concrete": {
            "relative_permittivity": 5.0,
            "relative_permeability": 1.0,
            "conductivity": 0.01,
        },

        "glass": {
            "relative_permittivity": 6.0,
            "relative_permeability": 1.0,
            "conductivity": 1.0e-12,
        },

        "wood": {
            "relative_permittivity": 2.0,
            "relative_permeability": 1.0,
            "conductivity": 0.001,
        },
    }

    def __init__(
        self,
        simulation_space,
        name,
        x_min,
        x_max,
        y_min,
        y_max,
        relative_permittivity=None,
        relative_permeability=None,
        conductivity=None,
    ):
        """
        Constructor Arguments
        ---------------------

        simulation_space
            SimulationSpace whose material maps will be modified.

        name
            Material name.

            Built-in materials:
                "concrete"
                "glass"
                "wood"

            Any other name is treated as a custom material.

        x_min, x_max
            Minimum and maximum physical x-coordinates of the
            rectangular material region [m].

        y_min, y_max
            Minimum and maximum physical y-coordinates of the
            rectangular material region [m].

        relative_permittivity
            Relative permittivity εr.

            Required only for custom materials.

        relative_permeability
            Relative permeability μr.

            Required only for custom materials.

        conductivity
            Electrical conductivity σ [S/m].

            Required only for custom materials.
        """

        # ==============================================================
        # Store SimulationSpace
        # ==============================================================

        self.simulation_space = simulation_space

        # ==============================================================
        # Validate and store material name
        # ==============================================================

        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                "Material name must be a non-empty string."
            )

        # Remove unnecessary spaces and make the name case-insensitive.
        self.name = name.strip().lower()

        # ==============================================================
        # Validate rectangular coordinates
        # ==============================================================

        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.y_min = float(y_min)
        self.y_max = float(y_max)

        if self.x_min >= self.x_max:
            raise ValueError(
                "x_min must be smaller than x_max."
            )

        if self.y_min >= self.y_max:
            raise ValueError(
                "y_min must be smaller than y_max."
            )

        # Make sure the complete rectangle is inside the simulation.
        if not simulation_space.is_inside(
            self.x_min,
            self.y_min,
        ):
            raise ValueError(
                "Material region lower-left corner is outside "
                "the simulation space."
            )

        if not simulation_space.is_inside(
            self.x_max,
            self.y_max,
        ):
            raise ValueError(
                "Material region upper-right corner is outside "
                "the simulation space."
            )

        # ==============================================================
        # Load material properties
        # ==============================================================

        if self.name in self.MATERIAL_DATABASE:

            # ----------------------------------------------------------
            # Built-in material
            #
            # The user does not need to provide εr, μr or σ.
            # ----------------------------------------------------------

            properties = self.MATERIAL_DATABASE[self.name]

            self.relative_permittivity = (
                properties["relative_permittivity"]
            )

            self.relative_permeability = (
                properties["relative_permeability"]
            )

            self.conductivity = (
                properties["conductivity"]
            )

        else:

            # ----------------------------------------------------------
            # Custom material
            #
            # If the material name is not in the database, the user
            # must provide all three electromagnetic properties.
            # ----------------------------------------------------------

            if relative_permittivity is None:
                raise ValueError(
                    f"Unknown material '{name}'. "
                    "Custom materials require "
                    "relative_permittivity."
                )

            if relative_permeability is None:
                raise ValueError(
                    f"Unknown material '{name}'. "
                    "Custom materials require "
                    "relative_permeability."
                )

            if conductivity is None:
                raise ValueError(
                    f"Unknown material '{name}'. "
                    "Custom materials require conductivity."
                )

            self.relative_permittivity = float(
                relative_permittivity
            )

            self.relative_permeability = float(
                relative_permeability
            )

            self.conductivity = float(
                conductivity
            )

        # ==============================================================
        # Validate electromagnetic properties
        # ==============================================================

        if self.relative_permittivity <= 0.0:
            raise ValueError(
                "Relative permittivity must be greater than zero."
            )

        if self.relative_permeability <= 0.0:
            raise ValueError(
                "Relative permeability must be greater than zero."
            )

        if self.conductivity < 0.0:
            raise ValueError(
                "Conductivity cannot be negative."
            )

        # ==============================================================
        # Calculate material properties required by WaveSolver
        # ==============================================================

        self.wave_speed = self._calculate_wave_speed()

        self.attenuation = self._calculate_attenuation()

    # ==================================================================
    # MATERIAL CALCULATIONS
    # ==================================================================

    def _calculate_wave_speed(self):
        """
        Calculates electromagnetic wave propagation speed.

        For a material:

            ε = ε0 εr
            μ = μ0 μr

        Therefore:

            v = 1 / sqrt(με)

        which can also be written as:

            v = c0 / sqrt(μr εr)

        Returns
        -------
        float
            Wave propagation speed [m/s].
        """

        return self.C_0 / np.sqrt(
            self.relative_permeability
            * self.relative_permittivity
        )

    def _calculate_attenuation(self):
        """
        Calculates the damping coefficient used by the current
        WaveSolver equation.

        The solver uses:

            ∂²E/∂t² + α∂E/∂t
            =
            c²(∂²E/∂x² + ∂²E/∂y²)

        For a conductive medium, the conductivity term can be
        represented in this simplified model using:

            α = σ / ε

        where:

            ε = ε0 εr

        Therefore:

            α = σ / (ε0 εr)

        Returns
        -------
        float
            Attenuation/damping coefficient [1/s].
        """

        epsilon = (
            self.EPSILON_0
            * self.relative_permittivity
        )

        return self.conductivity / epsilon

    # ==================================================================
    # APPLY MATERIAL TO SIMULATION SPACE
    # ==================================================================

    def apply(self):
        """
        Applies this material to its rectangular region.

        The existing SimulationSpace rectangle APIs are used to
        update the corresponding wave-speed and attenuation maps.
        """

        # --------------------------------------------------------------
        # Set the calculated wave speed throughout the material
        # rectangle.
        #
        # SimulationSpace handles conversion from physical coordinates
        # to grid indices internally.
        # --------------------------------------------------------------

        self.simulation_space.set_wave_speed_rectangle(
            self.x_min,
            self.y_min,
            self.x_max,
            self.y_max,
            self.wave_speed,
        )

        # --------------------------------------------------------------
        # Set the calculated attenuation throughout the material
        # rectangle.
        #
        # SimulationSpace handles conversion from physical coordinates
        # to grid indices internally.
        # --------------------------------------------------------------

        self.simulation_space.set_attenuation_rectangle(
            self.x_min,
            self.y_min,
            self.x_max,
            self.y_max,
            self.attenuation,
        )

    # ==================================================================
    # GETTERS
    # ==================================================================

    def get_name(self):
        """
        Returns the material name.
        """

        return self.name

    def get_relative_permittivity(self):
        """
        Returns relative permittivity εr.
        """

        return self.relative_permittivity

    def get_relative_permeability(self):
        """
        Returns relative permeability μr.
        """

        return self.relative_permeability

    def get_conductivity(self):
        """
        Returns electrical conductivity σ [S/m].
        """

        return self.conductivity

    def get_wave_speed(self):
        """
        Returns wave propagation speed [m/s].
        """

        return self.wave_speed

    def get_attenuation(self):
        """
        Returns the attenuation/damping coefficient [1/s].
        """

        return self.attenuation

    def get_region(self):
        """
        Returns the rectangular material region.

        Returns
        -------
        tuple
            (x_min, x_max, y_min, y_max)
        """

        return (
            self.x_min,
            self.x_max,
            self.y_min,
            self.y_max,
        )

    def contains(self, x, y):
        """
        Checks whether a physical coordinate lies inside
        this material's rectangular region.

        Returns
        -------
        bool
            True if the point is inside the material region.
        """

        return (
            self.x_min <= x <= self.x_max
            and
            self.y_min <= y <= self.y_max
        )

    # ==================================================================
    # CLASS / DATABASE API
    # ==================================================================

    @classmethod
    def get_available_materials(cls):
        """
        Returns the names of all predefined materials.

        Example:

            ["concrete", "glass", "wood"]
        """

        return list(cls.MATERIAL_DATABASE.keys())

