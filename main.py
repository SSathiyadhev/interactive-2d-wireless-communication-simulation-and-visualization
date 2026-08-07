import numpy as np
import matplotlib.pyplot as plt

from src.simulation_space import SimulationSpace
from src.wave_solver import WaveSolver
from src.transmitter import Transmitter


def main():

    # ---------------------------------------------------------
    # Simulation Space Configuration
    # ---------------------------------------------------------

    width = 10.0
    height = 10.0

    resolution_x = 600
    resolution_y = 600

    dx = width / (resolution_x - 1) # Actually Calculated inside the SimulationSpace class here its just for the Courant condition calculation
    dy = height / (resolution_y - 1) # Actually Calculated inside the SimulationSpace class here its just for the Courant condition calculation

    # Speed of light in free space (m/s)

    c = 3.0e8 # Actually set seperatly in global material properties here its just for the Courant condition calculation

    # Stable simulation time step (Courant condition)

    dt = 0.85 / (
        c * np.sqrt(
            (1.0 / dx**2) +
            (1.0 / dy**2)
        )
    )

    # ---------------------------------------------------------
    # Create Simulation Space
    # ---------------------------------------------------------

    simulation_space = SimulationSpace(
        width=width,
        height=height,
        resolution_x=resolution_x,
        resolution_y=resolution_y,
        dt=dt,
    )

    # ---------------------------------------------------------
    # Global Material Properties
    # ---------------------------------------------------------

    simulation_space.set_global_wave_speed(c)
    simulation_space.set_global_attenuation(0.0)

    # ---------------------------------------------------------
    # Create Transmitters
    # ---------------------------------------------------------

    transmitters = [

        Transmitter(
            simulation_space=simulation_space,
            x=100,
            y=250,
            carrier_frequency=1.0e9,
            carrier_amplitude=2.0,
            bit_rate=1.0e6,
        ),

        Transmitter(
            simulation_space=simulation_space,
            x=100,
            y=300,
            carrier_frequency=1.0e9,
            carrier_amplitude=2.0,
            bit_rate=1.0e6,
        ),

        Transmitter(
            simulation_space=simulation_space,
            x=100,
            y=350,
            carrier_frequency=1.0e9,
            carrier_amplitude=2.0,
            bit_rate=1.0e6,
        ),

    ]

    # ---------------------------------------------------------
    # Create Wave Solver
    # ---------------------------------------------------------

    wave_solver = WaveSolver(
        simulation_space,
    )

    # ---------------------------------------------------------
    # Start Simulation
    # ---------------------------------------------------------

    simulation_space.set_running(True)

    # ---------------------------------------------------------
    # Create Visualization Window
    # ---------------------------------------------------------

    plt.ion()

    figure, axis = plt.subplots(
        figsize=(8, 8),
    )

    image = axis.imshow(
        simulation_space.get_current_field().T,
        cmap="RdBu_r",
        origin="lower",
        interpolation="nearest",
        vmin=-2.0,
        vmax=2.0,
    )

    plt.colorbar(image)

    # ---------------------------------------------------------
    # Simulation Time Display
    # ---------------------------------------------------------

    time_text = axis.text(
        0.02,
        0.98,
        "",
        transform=axis.transAxes,
        fontsize=12,
        color="black",
        verticalalignment="top",
        bbox=dict(
            facecolor="white",
            alpha=0.8,
        ),
    )

    frame = 0

    # Refresh the visualization every N simulation steps

    display_every = 5

    # ---------------------------------------------------------
    # Main Simulation Loop
    # ---------------------------------------------------------

    while simulation_space.is_running():

        # Update every transmitter for the current simulation time

        for transmitter in transmitters:
            transmitter.transmit()

        # Advance the electromagnetic field by one time step

        wave_solver.solve()

        # Update the visualization periodically

        if frame % display_every == 0:

            image.set_data(
                simulation_space.get_current_field().T
            )

            time_text.set_text(
                f"Simulation Time : "
                f"{simulation_space.time * 1e9:.3f} ns"
            )

            figure.canvas.draw_idle()
            figure.canvas.flush_events()

        # Advance the simulation clock

        simulation_space.advance_time()

        frame += 1

    # ---------------------------------------------------------
    # Close Interactive Mode
    # ---------------------------------------------------------

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
