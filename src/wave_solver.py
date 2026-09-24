"""
wave_solver.py

Defines the WaveSolver class.

The WaveSolver advances the electromagnetic field by one simulation
time step by solving the two-dimensional heterogeneous electromagnetic
wave equation using the Finite Difference Time Domain (FDTD) method.
"""

import numpy as np
from numba import njit, prange


# -------------------------------------------------------------------------
# JIT-compiled FDTD numerical kernel
#
# This function contains only numerical operations.
# No Python objects or SimulationSpace methods are passed into Numba.
# -------------------------------------------------------------------------

@njit(fastmath=True, cache=True)
def _fdtd_step(
    current_field,
    previous_field,
    next_field,
    current_coefficient,
    previous_coefficient,
    spatial_x_plus_coefficient,
    spatial_x_minus_coefficient,
    spatial_y_plus_coefficient,
    spatial_y_minus_coefficient,
    noise_level,
):
    """
    Performs one FDTD update over the complete simulation grid.

    The update solves the heterogeneous electromagnetic wave equation:

        ε(x,y) ∂²E/∂t² + σ(x,y) ∂E/∂t
        = ∇ · ((1/μ(x,y)) ∇E)

    All material-dependent coefficients are precomputed outside this
    function so that they do not have to be recalculated at every
    simulation timestep.
    """

    rows = current_field.shape[0]
    cols = current_field.shape[1]

    # Preserve the original behavior:
    # boundary cells are reset to zero every timestep.
    next_field.fill(0.0)

    # Parallelize the spatial FDTD calculation.
    for i in range(1, rows - 1):

        for j in range(1, cols - 1):

            center = current_field[i, j]

            # ---------------------------------------------------------
            # Spatial flux differences
            #
            # q = 1 / μ
            #
            # The heterogeneous spatial operator is:
            #
            # ∂/∂x (q ∂E/∂x) + ∂/∂y (q ∂E/∂y)
            #
            # rather than simply:
            #
            # q ∇²E
            #
            # This form correctly accounts for spatial variation
            # of the permeability.
            # ---------------------------------------------------------

            flux_x = (
                spatial_x_plus_coefficient[i, j]
                * (current_field[i + 1, j] - center)
                -
                spatial_x_minus_coefficient[i, j]
                * (center - current_field[i - 1, j])
            )

            flux_y = (
                spatial_y_plus_coefficient[i, j]
                * (current_field[i, j + 1] - center)
                -
                spatial_y_minus_coefficient[i, j]
                * (center - current_field[i, j - 1])
            )

            # ---------------------------------------------------------
            # Final heterogeneous FDTD update
            # ---------------------------------------------------------

            next_field[i, j] = (
                current_coefficient[i, j] * center
                + previous_coefficient[i, j] * previous_field[i, j]
                + flux_x
                + flux_y
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
    Solves the two-dimensional heterogeneous electromagnetic wave
    equation using the Finite Difference Time Domain (FDTD) method.
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

        # ---------------------------------------------------------
        # Material property maps
        #
        # ε(x,y) : Permittivity
        # σ(x,y) : Electrical conductivity
        # μ(x,y) : Permeability
        # ---------------------------------------------------------

        self._epsilon = simulation_space.get_permittivity_map()
        self._sigma = simulation_space.get_conductivity_map()
        self._mu = simulation_space.get_permeability_map()

        # Grid parameters

        self._dx = simulation_space.dx
        self._dy = simulation_space.dy
        self._dt = simulation_space.dt

    def _validate_courant_stability(self):
        """
        Validates the Courant stability condition for the heterogeneous
        electromagnetic wave equation.

        For the lossless local propagation speed:

            c(x,y) = 1 / sqrt(ε(x,y) μ(x,y))

        the Courant condition is:

            (c dt / dx)^2 + (c dt / dy)^2 <= 1

        The material constraints enforced by SimulationSpace ensure:

            ε(x,y) >= ε0
            μ(x,y) >= μ0

        and therefore:

            c(x,y) <= c0

        """

        epsilon = self._epsilon
        mu = self._mu

        dt = self._dt
        dx = self._dx
        dy = self._dy

        # ---------------------------------------------------------
        # Local squared propagation speed:
        #
        # c²(x,y) = 1 / (ε(x,y) μ(x,y))
        # ---------------------------------------------------------

        local_c_squared = 1.0 / (epsilon * mu)

        # ---------------------------------------------------------
        # Local Courant number:
        #
        # C² = c² dt² / dx² + c² dt² / dy²
        # ---------------------------------------------------------

        courant = (
            local_c_squared * dt**2
            * (
                (1.0 / dx**2)
                +
                (1.0 / dy**2)
            )
        )

        if np.any(courant > 1.0 + 1e-12):

            raise ValueError(
                "FDTD simulation is unstable. "
                "The Courant stability condition is violated."
            )

    def _calculate_coefficients(self):
        """
        Calculates and caches all FDTD coefficients.

        These coefficients depend on:

            ε(x,y)
            σ(x,y)
            μ(x,y)
            dx
            dy
            dt

        Therefore they do not need to be recalculated every
        simulation timestep.
        """

        epsilon = self._epsilon
        sigma = self._sigma
        mu = self._mu

        dx = self._dx
        dy = self._dy
        dt = self._dt

        # ---------------------------------------------------------
        # Reciprocal permeability
        #
        # q(x,y) = 1 / μ(x,y)
        # ---------------------------------------------------------

        q = 1.0 / mu

        # ---------------------------------------------------------
        # Interface values of q = 1/μ
        #
        # q(i+1/2,j) is the interface value between
        # grid points (i,j) and (i+1,j).
        #
        # q(i-1/2,j) is the interface value between
        # grid points (i-1,j) and (i,j).
        #
        # Likewise for the y direction.
        #
        # Harmonic averaging is used to obtain the interface
        # coefficient from the neighbouring grid-point values.
        #
        # This provides the interface coefficient for the
        # conservative flux-divergence discretization.
        # ---------------------------------------------------------

        q_x_plus = (
            2.0
            * q[1:, :]
            * q[:-1, :]
            / (
                q[1:, :]
                + q[:-1, :]
            )
        )

        q_y_plus = (
            2.0
            * q[:, 1:]
            * q[:, :-1]
            / (
                q[:, 1:]
                + q[:, :-1]
            )
        )

        # ---------------------------------------------------------
        # Create arrays for the four interface coefficients.
        #
        # They have the same shape as the complete simulation grid
        # so that the Numba kernel can access them directly using
        # [i, j].
        # ---------------------------------------------------------

        q_x_plus_full = np.zeros_like(q)
        q_x_minus_full = np.zeros_like(q)

        q_y_plus_full = np.zeros_like(q)
        q_y_minus_full = np.zeros_like(q)

        q_x_plus_full[:-1, :] = q_x_plus
        q_x_minus_full[1:, :] = q_x_plus

        q_y_plus_full[:, :-1] = q_y_plus
        q_y_minus_full[:, 1:] = q_y_plus

        # ---------------------------------------------------------
        # Conductivity term
        #
        # Starting from:
        #
        # ε(Eⁿ⁺¹ - 2Eⁿ + Eⁿ⁻¹)/dt² + σ(Eⁿ⁺¹ - Eⁿ⁻¹)/(2dt) = L(Eⁿ)
        #
        # multiplying by dt² and collecting Eⁿ⁺¹:
        #
        # (ε + σdt/2) Eⁿ⁺¹ = 2εEⁿ - (ε - σdt/2)Eⁿ⁻¹ + dt² L(Eⁿ)
        # ---------------------------------------------------------

        sigma_dt_half = sigma * dt / 2.0

        denominator = (
            epsilon
            +
            sigma_dt_half
        )

        # ---------------------------------------------------------
        # Coefficient multiplying Eⁿ
        # ---------------------------------------------------------

        self._current_coefficient = (
            2.0 * epsilon / denominator
        )

        # ---------------------------------------------------------
        # Coefficient multiplying Eⁿ⁻¹
        # ---------------------------------------------------------

        self._previous_coefficient = (
            -(epsilon - sigma_dt_half)
            / denominator
        )

        # ---------------------------------------------------------
        # Spatial flux coefficients
        #
        # dt² / dx² · q(i+1/2,j)
        # dt² / dx² · q(i-1/2,j)
        # dt² / dy² · q(i,j+1/2)
        # dt² / dy² · q(i,j-1/2)
        #
        # divided by:
        #
        # ε + σdt/2
        # ---------------------------------------------------------

        self._spatial_x_plus_coefficient = (
            dt**2
            * q_x_plus_full
            / (
                dx**2
                * denominator
            )
        )

        self._spatial_x_minus_coefficient = (
            dt**2
            * q_x_minus_full
            / (
                dx**2
                * denominator
            )
        )

        self._spatial_y_plus_coefficient = (
            dt**2
            * q_y_plus_full
            / (
                dy**2
                * denominator
            )
        )

        self._spatial_y_minus_coefficient = (
            dt**2
            * q_y_minus_full
            / (
                dy**2
                * denominator
            )
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
        Solves the two-dimensional heterogeneous electromagnetic wave
        equation for one simulation time step and updates the
        SimulationSpace.
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
        # Governing Equation
        #
        # Heterogeneous electromagnetic wave equation:
        #
        # ε(x,y) ∂²E/∂t² + σ(x,y) ∂E/∂t = ∇ · ((1/μ(x,y)) ∇E)
        # ---------------------------------------------------------
        # Define:
        #
        # q(x,y) = 1 / μ(x,y)
        #
        # Therefore:
        #
        # ∇ · (q ∇E) = ∂/∂x (q ∂E/∂x) + ∂/∂y (q ∂E/∂y)
        #
        # ---------------------------------------------------------
        # Spatial discretization
        #
        # The x-direction term is discretized as:
        #
        # [q(i+1/2,j)(E(i+1,j)-E(i,j)) - q(i-1/2,j)(E(i,j)-E(i-1,j))] / dx²
        #
        # and the y-direction term as:
        #
        # [q(i,j+1/2)(E(i,j+1)-E(i,j)) - q(i,j-1/2)(E(i,j)-E(i,j-1))] / dy²
        #
        # ---------------------------------------------------------
        # Time finite differences
        #
        # ∂²E/∂t² ≈ (Eⁿ⁺¹ - 2Eⁿ + Eⁿ⁻¹) / dt²
        #
        # ∂E/∂t ≈ (Eⁿ⁺¹ - Eⁿ⁻¹) / (2dt)
        #
        # ---------------------------------------------------------
        # Substituting into the governing equation:
        #
        # ε(Eⁿ⁺¹ - 2Eⁿ + Eⁿ⁻¹)/dt² + σ(Eⁿ⁺¹ - Eⁿ⁻¹)/(2dt) = L(Eⁿ)
        #
        # where L(Eⁿ) represents:
        #
        # ∇ · ((1/μ)∇E)
        #
        # evaluated using the spatial flux discretization above.
        #
        # ---------------------------------------------------------
        # Multiply by dt²:
        #
        # ε(Eⁿ⁺¹ - 2Eⁿ + Eⁿ⁻¹) + (σdt/2)(Eⁿ⁺¹ - Eⁿ⁻¹) = dt² L(Eⁿ)
        #
        # ---------------------------------------------------------
        # Collect Eⁿ⁺¹:
        #
        # (ε + σdt/2)Eⁿ⁺¹ = 2εEⁿ - (ε - σdt/2)Eⁿ⁻¹ + dt² L(Eⁿ)
        #
        # ---------------------------------------------------------
        # Final heterogeneous FDTD update:
        #
        # Eⁿ⁺¹ = [2εEⁿ - (ε - σdt/2)Eⁿ⁻¹ + dt² ∇ · ((1/μ)∇Eⁿ)] / (ε + σdt/2)
        #
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
            self._spatial_x_plus_coefficient,
            self._spatial_x_minus_coefficient,
            self._spatial_y_plus_coefficient,
            self._spatial_y_minus_coefficient,
            self.noise_level,
        )

        # ---------------------------------------------------------
        # Pass the computed field back to SimulationSpace.
        # ---------------------------------------------------------

        simulation_space.set_next_field(next_field)

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

        Call this if the SimulationSpace's material maps, dx, dy, or dt
        are changed after this WaveSolver was created.
        """

        self._refresh_coefficients()
    