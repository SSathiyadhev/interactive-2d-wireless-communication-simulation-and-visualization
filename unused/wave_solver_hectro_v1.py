"""
wave_solver.py

Defines the WaveSolver class.

The WaveSolver advances the electromagnetic field by one simulation
time step by solving the two-dimensional heterogeneous electromagnetic
wave equation using the Finite Difference Time Domain (FDTD) method.
"""

import numpy as np


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
        # Validate the Courant stability condition
        #
        # For the local lossless propagation speed:
        #
        # c(x,y) = 1 / sqrt(ε(x,y) μ(x,y))
        #
        # the Courant condition is:
        #
        # (c dt / dx)^2 + (c dt / dy)^2 <= 1
        # ---------------------------------------------------------

        epsilon = simulation_space.get_permittivity_map()
        mu = simulation_space.get_permeability_map()

        dx = simulation_space.dx
        dy = simulation_space.dy
        dt = simulation_space.dt

        local_c_squared = 1.0 / (epsilon * mu)

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

        # ---------------------------------------------------------
        # Reusable next-field buffer
        # ---------------------------------------------------------

        self._next_field = np.zeros_like(
            simulation_space.get_current_field()
        )

    def solve(self):
        """
        Solves the two-dimensional heterogeneous electromagnetic
        wave equation for one simulation time step and updates the
        SimulationSpace.
        """

        # Reference to the simulation space

        simulation_space = self.simulation_space

        # ---------------------------------------------------------
        # Electromagnetic fields
        # ---------------------------------------------------------

        current_field = simulation_space.get_current_field()
        previous_field = simulation_space.get_previous_field()

        # ---------------------------------------------------------
        # Material properties
        #
        # ε(x,y) : Permittivity
        # σ(x,y) : Electrical conductivity
        # μ(x,y) : Permeability
        # ---------------------------------------------------------

        epsilon = simulation_space.get_permittivity_map()
        sigma = simulation_space.get_conductivity_map()
        mu = simulation_space.get_permeability_map()

        # ---------------------------------------------------------
        # Grid parameters
        # ---------------------------------------------------------

        dx = simulation_space.dx
        dy = simulation_space.dy
        dt = simulation_space.dt

        # ---------------------------------------------------------
        # Field at the next simulation time step
        #
        # Eⁿ⁺¹
        # ---------------------------------------------------------

        next_field = self._next_field
        next_field.fill(0.0)

        # =========================================================
        # GOVERNING EQUATION
        # =========================================================
        #
        # ε(x,y) ∂²E/∂t² + σ(x,y) ∂E/∂t
        #
        # = ∇ · ((1/μ(x,y)) ∇E)
        #
        # ---------------------------------------------------------
        # Define:
        #
        # q(x,y) = 1 / μ(x,y)
        #
        # Therefore:
        #
        # ∇ · (q ∇E)
        #
        # = ∂/∂x(q ∂E/∂x)
        # + ∂/∂y(q ∂E/∂y)
        # =========================================================

        q = 1.0 / mu

        # =========================================================
        # INTERFACE VALUES OF q = 1/μ
        # =========================================================
        #
        # q(i+1/2,j) is the interface value between
        # grid points (i,j) and (i+1,j).
        #
        # q(i,j+1/2) is the interface value between
        # grid points (i,j) and (i,j+1).
        #
        # Harmonic averaging is used at material interfaces.
        # =========================================================

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

        # =========================================================
        # TIME FINITE DIFFERENCE APPROXIMATIONS
        # =========================================================
        #
        # ∂²E/∂t² ≈
        #
        # (Eⁿ⁺¹ - 2Eⁿ + Eⁿ⁻¹) / dt²
        #
        # ∂E/∂t ≈
        #
        # (Eⁿ⁺¹ - Eⁿ⁻¹) / (2dt)
        # =========================================================

        sigma_dt_half = sigma * dt / 2.0

        # =========================================================
        # COLLECT Eⁿ⁺¹ TERMS
        # =========================================================
        #
        # Starting from:
        #
        # ε(Eⁿ⁺¹ - 2Eⁿ + Eⁿ⁻¹)/dt²
        #
        # + σ(Eⁿ⁺¹ - Eⁿ⁻¹)/(2dt)
        #
        # = L(Eⁿ)
        #
        # where:
        #
        # L(Eⁿ) = ∇ · ((1/μ)∇Eⁿ)
        #
        # Multiplying by dt²:
        #
        # ε(Eⁿ⁺¹ - 2Eⁿ + Eⁿ⁻¹)
        #
        # + (σdt/2)(Eⁿ⁺¹ - Eⁿ⁻¹)
        #
        # = dt² L(Eⁿ)
        #
        # Therefore:
        #
        # (ε + σdt/2)Eⁿ⁺¹
        #
        # = 2εEⁿ
        #
        # - (ε - σdt/2)Eⁿ⁻¹
        #
        # + dt² L(Eⁿ)
        #
        # =========================================================

        denominator = (
            epsilon
            + sigma_dt_half
        )

        # =========================================================
        # SPATIAL FLUX-DIVERGENCE DISCRETIZATION
        # =========================================================
        #
        # x-direction:
        #
        # [q(i+1/2,j)(E(i+1,j)-E(i,j))
        #
        # - q(i-1/2,j)(E(i,j)-E(i-1,j))]
        #
        # / dx²
        #
        # y-direction:
        #
        # [q(i,j+1/2)(E(i,j+1)-E(i,j))
        #
        # - q(i,j-1/2)(E(i,j)-E(i,j-1))]
        #
        # / dy²
        #
        # The interface values are obtained using harmonic averaging.
        # =========================================================

        flux_x = (
            q_x_plus[1:, 1:-1]
            * (
                current_field[2:, 1:-1]
                - current_field[1:-1, 1:-1]
            )
            -
            q_x_plus[:-1, 1:-1]
            * (
                current_field[1:-1, 1:-1]
                - current_field[:-2, 1:-1]
            )
        ) / dx**2

        flux_y = (
            q_y_plus[1:-1, 1:]
            * (
                current_field[1:-1, 2:]
                - current_field[1:-1, 1:-1]
            )
            -
            q_y_plus[1:-1, :-1]
            * (
                current_field[1:-1, 1:-1]
                - current_field[1:-1, :-2]
            )
        ) / dy**2

        # =========================================================
        # FINAL HETEROGENEOUS FDTD UPDATE
        # =========================================================
        #
        # Eⁿ⁺¹ =
        #
        # [
        #     2εEⁿ
        #
        #     - (ε - σdt/2)Eⁿ⁻¹
        #
        #     + dt² L(Eⁿ)
        # ]
        #
        # / (ε + σdt/2)
        # =========================================================

        next_field[1:-1, 1:-1] = (
            (
                2.0
                * epsilon[1:-1, 1:-1]
                * current_field[1:-1, 1:-1]

                -

                (
                    epsilon[1:-1, 1:-1]
                    - sigma_dt_half[1:-1, 1:-1]
                )
                * previous_field[1:-1, 1:-1]

                +

                dt**2
                * (
                    flux_x
                    + flux_y
                )
            )
            / denominator[1:-1, 1:-1]
        )

        # =========================================================
        # ADD GLOBAL AWGN
        #
        # noise_level = standard deviation of the Gaussian noise
        # =========================================================

        if self.noise_level > 0.0:

            next_field[1:-1, 1:-1] += np.random.normal(
                loc=0.0,
                scale=self.noise_level,
                size=next_field[1:-1, 1:-1].shape,
            )

        # ---------------------------------------------------------
        # Pass the computed field back to SimulationSpace
        # ---------------------------------------------------------

        simulation_space.set_next_field(next_field)

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
