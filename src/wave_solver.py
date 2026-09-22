"""
wave_solver.py

Defines the WaveSolver class.

The WaveSolver advances the electromagnetic field by one simulation
time step by solving the two-dimensional wave equation using the
Finite Difference Time Domain (FDTD) method.
"""

import numpy as np
from numba import njit, prange


# -------------------------------------------------------------------------
# JIT-compiled FDTD numerical kernel
#
# This function contains only numerical operations.
# No Python objects or SimulationSpace methods are passed into Numba.
# -------------------------------------------------------------------------

@njit(parallel=True, fastmath=True, cache=True)
def _fdtd_step(
    current_field,
    previous_field,
    next_field,
    current_coefficient,
    previous_coefficient,
    courant_x_coefficient,
    courant_y_coefficient,
    noise_level,
):
    """
    Performs one FDTD update over the complete simulation grid.

    All coefficients are precomputed outside this function so that
    they do not have to be recalculated at every simulation timestep.
    """

    rows = current_field.shape[0]
    cols = current_field.shape[1]

    # Preserve the original behavior:
    # boundary cells are reset to zero every timestep.
    next_field.fill(0.0)

    # Parallelize the spatial FDTD calculation.
    for i in prange(1, rows - 1):

        for j in range(1, cols - 1):

            center = current_field[i, j]

            # ---------------------------------------------------------
            # Spatial finite differences
            # ---------------------------------------------------------

            delta_x = (
                current_field[i + 1, j]
                - 2.0 * center
                + current_field[i - 1, j]
            )

            delta_y = (
                current_field[i, j + 1]
                - 2.0 * center
                + current_field[i, j - 1]
            )

            # ---------------------------------------------------------
            # Final FDTD update
            #
            # Coefficients have already been divided by
            # (1 + αdt/2).
            # ---------------------------------------------------------

            next_field[i, j] = (
                current_coefficient[i, j] * center
                + previous_coefficient[i, j] * previous_field[i, j]
                + courant_x_coefficient[i, j] * delta_x
                + courant_y_coefficient[i, j] * delta_y
            )

    # ---------------------------------------------------------
    # Add global AWGN to the computed field
    #
    # noise_level = standard deviation of the Gaussian noise
    # ---------------------------------------------------------

    if noise_level > 0.0:

        for i in prange(1, rows - 1):

            for j in range(1, cols - 1):

                next_field[i, j] += np.random.normal(
                    0.0,
                    noise_level,
                )


class WaveSolver:
    """
    Solves the two-dimensional wave equation using the
    Finite Difference Time Domain (FDTD) method.
    """

    def __init__(
        self,
        simulation_space,
        noise_level=0.0,
    ):
        """
        Constructor Arguments
        ---------------------
        simulation_space    : Reference to the SimulationSpace
                              to be updated.

        noise_level         : Standard deviation of the global
                              additive white Gaussian noise.
        """

        self.simulation_space = simulation_space
        self.noise_level = noise_level

        # ---------------------------------------------------------
        # Cache simulation parameters
        # ---------------------------------------------------------

        self._cache_simulation_parameters()

        # ---------------------------------------------------------
        # Validate the Courant stability condition
        # ---------------------------------------------------------

        self._validate_courant_stability()

        # ---------------------------------------------------------
        # Precompute FDTD coefficients
        # ---------------------------------------------------------

        self._calculate_coefficients()

        # ---------------------------------------------------------
        # Reusable next-field buffer
        # ---------------------------------------------------------

        self._next_field = np.zeros_like(
            simulation_space.get_current_field()
        )

    # =====================================================================
    # CACHE / SETUP
    # =====================================================================

    def _cache_simulation_parameters(self):
        """
        Caches simulation-space data that is unchanged during normal
        timestep execution.

        Keeping these references outside solve() avoids repeated
        SimulationSpace API calls during every timestep.
        """

        simulation_space = self.simulation_space

        # Material properties

        self._wave_speed = simulation_space.get_wave_speed_map()
        self._attenuation = simulation_space.get_attenuation_map()

        # Grid parameters

        self._dx = simulation_space.dx
        self._dy = simulation_space.dy
        self._dt = simulation_space.dt

    def _validate_courant_stability(self):
        """
        Validates the Courant stability condition.
        """

        wave_speed = self._wave_speed
        dt = self._dt
        dx = self._dx
        dy = self._dy

        courant = (
            (wave_speed * dt / dx) ** 2 +
            (wave_speed * dt / dy) ** 2
        )

        if np.any(courant > 1.0 + 1e-12):
            raise ValueError(
                "FDTD simulation is unstable. "
                "The Courant stability condition is violated."
            )

    def _calculate_coefficients(self):
        """
        Calculates and caches all FDTD coefficients.

        These coefficients depend only on:
            wave_speed
            attenuation
            dx
            dy
            dt

        Therefore they do not need to be recalculated every
        simulation timestep.
        """

        wave_speed = self._wave_speed
        attenuation = self._attenuation

        dx = self._dx
        dy = self._dy
        dt = self._dt

        # ---------------------------------------------------------
        # Courant numbers
        #
        # Cx² = (c·dt/dx)²
        #
        # Cy² = (c·dt/dy)²
        # ---------------------------------------------------------

        courant_x_sq = (wave_speed * dt / dx) ** 2
        courant_y_sq = (wave_speed * dt / dy) ** 2

        # ---------------------------------------------------------
        # Attenuation coefficient
        #
        # αdt/2
        # ---------------------------------------------------------

        alpha_dt_half = attenuation * dt / 2.0

        # ---------------------------------------------------------
        # Precompute the complete coefficients used by the
        # final FDTD equation.
        #
        # Eⁿ⁺¹ =
        #
        # [2Eⁿ
        #
        # - (1 - αdt/2)Eⁿ⁻¹
        #
        # + (cdt/dx)²Δx
        #
        # + (cdt/dy)²Δy]
        #
        # / (1 + αdt/2)
        # ---------------------------------------------------------

        denominator = 1.0 + alpha_dt_half

        self._current_coefficient = (
            2.0 / denominator
        )

        self._previous_coefficient = (
            -(1.0 - alpha_dt_half) / denominator
        )

        self._courant_x_coefficient = (
            courant_x_sq / denominator
        )

        self._courant_y_coefficient = (
            courant_y_sq / denominator
        )

    def _refresh_coefficients(self):
        """
        Refreshes cached simulation parameters and FDTD coefficients.

        This should be called if SimulationSpace grid/material parameters
        are changed after WaveSolver construction.
        """

        self._cache_simulation_parameters()
        self._validate_courant_stability()
        self._calculate_coefficients()

    # =====================================================================
    # SOLVER
    # =====================================================================

    def solve(self):
        """
        Solves the two-dimensional wave equation for one
        simulation time step and updates the SimulationSpace.
        """

        # Reference to the simulation space

        simulation_space = self.simulation_space

        # Electromagnetic fields
        #
        # These API calls are intentionally kept inside solve()
        # because the fields change every simulation timestep.

        current_field = simulation_space.get_current_field()
        previous_field = simulation_space.get_previous_field()

        # Field at the next simulation time step (Eⁿ⁺¹)

        next_field = self._next_field

        # ---------------------------------------------------------
        # Governing Equation Homogenious Wave Equation with Attenuation(homeginuois local material properties)
        # Later need to be updated to hectrogenious local material properties
        #
        # ∂²E/∂t² + α∂E/∂t = c²(∂²E/∂x² + ∂²E/∂y²)
        #
        # α = attenuation coefficient
        # c = wave propagation speed
        #
        # ---------------------------------------------------------
        # Finite Difference Approximations
        #
        # ∂²E/∂t² ≈ (Eⁿ⁺¹ - 2Eⁿ + Eⁿ⁻¹) / dt²
        #
        # ∂E/∂t ≈ (Eⁿ⁺¹ - Eⁿ⁻¹) / (2dt)
        #
        # ∂²E/∂x² ≈ (Eᵢ₊₁ - 2Eᵢ + Eᵢ₋₁) / dx²
        #
        # ∂²E/∂y² ≈ (Eⱼ₊₁ - 2Eⱼ + Eⱼ₋₁) / dy²
        #
        # ---------------------------------------------------------
        # Substitute into the wave equation
        #
        # (Eⁿ⁺¹ - 2Eⁿ + Eⁿ⁻¹)/dt²
        #
        # + α(Eⁿ⁺¹ - Eⁿ⁻¹)/(2dt)
        #
        # = c²[(Eᵢ₊₁ - 2Eᵢ + Eᵢ₋₁)/dx²
        #
        # + (Eⱼ₊₁ - 2Eⱼ + Eⱼ₋₁)/dy²]
        #
        # ---------------------------------------------------------
        # Multiply both sides by dt²
        #
        # Eⁿ⁺¹ - 2Eⁿ + Eⁿ⁻¹
        #
        # + (αdt/2)(Eⁿ⁺¹ - Eⁿ⁻¹)
        #
        # = (cdt/dx)²Δx + (cdt/dy)²Δy
        #
        # where
        #
        # Δx = Eᵢ₊₁ - 2Eᵢ + Eᵢ₋₁
        #
        # Δy = Eⱼ₊₁ - 2Eⱼ + Eⱼ₋₁
        #
        # ---------------------------------------------------------
        # Collect Eⁿ⁺¹ terms
        #
        # (1 + αdt/2)Eⁿ⁺¹
        #
        # = 2Eⁿ
        #
        # - (1 - αdt/2)Eⁿ⁻¹
        #
        # + (cdt/dx)²Δx
        #
        # + (cdt/dy)²Δy
        #
        # ---------------------------------------------------------
        # Final FDTD Update Equation
        #
        # Eⁿ⁺¹ =
        #
        # [2Eⁿ
        #
        # - (1 - αdt/2)Eⁿ⁻¹
        #
        # + (cdt/dx)²Δx
        #
        # + (cdt/dy)²Δy]
        #
        # / (1 + αdt/2)
        #
        # ---------------------------------------------------------
        # Courant numbers
        #
        # Cx² = (c·dt/dx)²
        #
        # Cy² = (c·dt/dy)²
        # ---------------------------------------------------------

        # ---------------------------------------------------------
        # Perform the computationally expensive FDTD step inside
        # the compiled parallel Numba kernel.
        # ---------------------------------------------------------

        _fdtd_step(
            current_field,
            previous_field,
            next_field,
            self._current_coefficient,
            self._previous_coefficient,
            self._courant_x_coefficient,
            self._courant_y_coefficient,
            self.noise_level,
        )

        # ---------------------------------------------------------
        # Pass the computed field back to SimulationSpace.
        # ---------------------------------------------------------

        simulation_space.set_next_field(next_field)

        # need to implement noise addition here, but for now noice level is set to 0.0 so no noise is added

    # =====================================================================
    # PUBLIC API
    # =====================================================================

    def set_noise_level(self, noise_level):
        """
        Updates the global AWGN noise level.
        """

        self.noise_level = noise_level

    def get_noise_level(self):
        """
        Returns the current global AWGN noise level.
        """

        return self.noise_level

    # =====================================================================
    # OPTIONAL CACHE REFRESH API
    # =====================================================================

    def refresh(self):
        """
        Refreshes cached SimulationSpace parameters and FDTD coefficients.

        Call this if the SimulationSpace's wave-speed map, attenuation map,
        dx, dy, or dt are changed after this WaveSolver was created.
        """

        self._refresh_coefficients()
    