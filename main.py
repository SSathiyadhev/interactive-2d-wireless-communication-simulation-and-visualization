import numpy as np
import matplotlib.pyplot as plt

from src.simulation_space import SimulationSpace
from src.wave_solver import WaveSolver
from src.transmitter import Transmitter


def main():

    resolution_x = 600
    resolution_y = 600

    dx = 10.0 / (resolution_x - 1)
    dy = 10.0 / (resolution_y - 1)

    c = 3.0e8

    dt = 0.85 / (
        c * np.sqrt(
            (1.0 / dx**2) +
            (1.0 / dy**2)
        )
    )

    simulation_space = SimulationSpace(
        width=10.0,
        height=10.0,
        resolution_x=resolution_x,
        resolution_y=resolution_y,
        dt=dt,
    )

    simulation_space.set_global_wave_speed(c)
    simulation_space.set_global_attenuation(0.0)

    transmitter = Transmitter(
        simulation_space=simulation_space,
        x=1.0,
        y=5.0,
        carrier_frequency=1.0e9,
        carrier_amplitude=2.0,
        bit_rate=500.0e6,
    )

    wave_solver = WaveSolver(simulation_space)

    simulation_space.set_running(True)

    plt.ion()

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(14, 6),
    )

    field_axis = axes[0]

    image = field_axis.imshow(
        simulation_space.get_current_field().T,
        cmap="RdBu_r",
        origin="lower",
        interpolation="nearest",
        vmin=-2.0,
        vmax=2.0,
    )

    plt.colorbar(image, ax=field_axis)

    field_axis.set_title("Electromagnetic Field")
    field_axis.set_xlabel("X")
    field_axis.set_ylabel("Y")

    waveform_axis = axes[1]

    waveform_line, = waveform_axis.plot([], [])

    waveform_axis.set_title("Transmitter BPSK Waveform")
    waveform_axis.set_xlabel("Time (ns)")
    waveform_axis.set_ylabel("Amplitude")
    waveform_axis.set_ylim(-2.5, 2.5)
    waveform_axis.set_xlim(0.0, 10.0)

    bit_text = waveform_axis.text(
        0.02,
        0.95,
        "",
        transform=waveform_axis.transAxes,
        fontsize=14,
        verticalalignment="top",
        bbox=dict(
            facecolor="white",
            alpha=0.8,
        ),
    )

    frame = 0

    while simulation_space.is_running():

        transmitter.transmit()

        wave_solver.solve()

        if frame % 1 == 0:

            image.set_data(
                simulation_space.get_current_field().T
            )

            time_values = np.asarray(
                transmitter.get_time_values()
            )

            bit_values = np.asarray(
                transmitter.get_bit_values()
            )

            bpsk_values = np.asarray(
                transmitter.get_bpsk_values()
            )

            if len(time_values) > 0:

                waveform_line.set_data(
                    time_values * 1e9,
                    bpsk_values,
                )

                current_time_ns = time_values[-1] * 1e9

                waveform_axis.set_xlim(
                    max(0.0, current_time_ns - 10.0),
                    max(10.0, current_time_ns),
                )

                bit_text.set_text(
                    f"Current Bit : {bit_values[-1]}"
                )

            figure.canvas.draw_idle()
            figure.canvas.flush_events()

        simulation_space.advance_time()

        frame += 1

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
