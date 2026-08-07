"""
wave_solver.py

Defines the WaveSolver class.

The WaveSolver advances the electromagnetic field by one simulation
time step by solving the two-dimensional wave equation using the
Finite Difference Time Domain (FDTD) method.
"""

import numpy as np

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

        # Validate the Courant stability condition

        wave_speed = simulation_space.get_wave_speed_map()
        dx = simulation_space.dx
        dy = simulation_space.dy
        dt = simulation_space.dt

        courant = (
            (wave_speed * dt / dx) ** 2 +
            (wave_speed * dt / dy) ** 2
        )

        if np.any(courant > 1.0 + 1e-12):
            raise ValueError(
                "FDTD simulation is unstable. "
                "The Courant stability condition is violated."
            )

        self._next_field = np.zeros_like(simulation_space.get_current_field())

    def solve(self):
        """
        Solves the two-dimensional wave equation for one
        simulation time step and updates the SimulationSpace.
        """

        # Reference to the simulation space

        simulation_space = self.simulation_space

        # Electromagnetic fields

        current_field = simulation_space.get_current_field()
        previous_field = simulation_space.get_previous_field()

        # Material properties

        wave_speed = simulation_space.get_wave_speed_map()
        attenuation = simulation_space.get_attenuation_map()

        # Grid parameters

        dx = simulation_space.dx
        dy = simulation_space.dy
        dt = simulation_space.dt

        # Field at the next simulation time step (Eⁿ⁺¹)

        next_field = self._next_field
        next_field.fill(0.0)

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

        courant_x_sq = (wave_speed * dt / dx) ** 2
        courant_y_sq = (wave_speed * dt / dy) ** 2

        alpha_dt_half = attenuation * dt / 2.0

        next_field[1:-1, 1:-1] = (
            (
                2.0 * current_field[1:-1, 1:-1]
                - (1.0 - alpha_dt_half[1:-1, 1:-1])
                * previous_field[1:-1, 1:-1]
                + courant_x_sq[1:-1, 1:-1] * (
                    current_field[2:, 1:-1]
                    - 2.0 * current_field[1:-1, 1:-1]
                    + current_field[:-2, 1:-1]
                )
                + courant_y_sq[1:-1, 1:-1] * (
                    current_field[1:-1, 2:]
                    - 2.0 * current_field[1:-1, 1:-1]
                    + current_field[1:-1, :-2]
                )
            )
            / (1.0 + alpha_dt_half[1:-1, 1:-1])
        )

        simulation_space.set_next_field(next_field)

        # need to implement noise addition here, but for now noice level is set to 0.0 so no noise is added
     
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
