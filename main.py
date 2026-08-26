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

    # =============================================================
    # FIGURE LAYOUT
    #
    # LEFT:
    #   Electromagnetic field
    #
    # RIGHT:
    #   1. Original square-wave BPSK symbols
    #   2. RRC-shaped baseband
    #   3. RRC-BPSK transmitted signal
    # =============================================================

    figure = plt.figure(
        figsize=(18, 10)
    )

    grid = figure.add_gridspec(
        3,
        2,
        width_ratios=[1.0, 1.5],
        hspace=0.45,
    )

    # =============================================================
    # 1. ELECTROMAGNETIC FIELD
    # =============================================================

    field_axis = figure.add_subplot(
        grid[:, 0]
    )

    image = field_axis.imshow(
        simulation_space.get_current_field().T,
        cmap="RdBu_r",
        origin="lower",
        interpolation="nearest",
        vmin=-2.0,
        vmax=2.0,
    )

    figure.colorbar(
        image,
        ax=field_axis,
    )

    field_axis.set_title(
        "Electromagnetic Field"
    )

    field_axis.set_xlabel(
        "X"
    )

    field_axis.set_ylabel(
        "Y"
    )

    # =============================================================
    # 2. ORIGINAL SQUARE-WAVE BPSK SYMBOLS
    # =============================================================

    bit_axis = figure.add_subplot(
        grid[0, 1]
    )

    bit_line, = bit_axis.plot(
        [],
        [],
        drawstyle="steps-post",
    )

    bit_axis.set_title(
        "Original BPSK Symbol Sequence"
    )

    bit_axis.set_xlabel(
        "Time (ns)"
    )

    bit_axis.set_ylabel(
        "Symbol"
    )

    bit_axis.set_ylim(
        -1.5,
        1.5,
    )

    # =============================================================
    # 3. RRC SHAPED BASEBAND
    # =============================================================

    shaped_axis = figure.add_subplot(
        grid[1, 1]
    )

    shaped_line, = shaped_axis.plot(
        [],
        [],
    )

    shaped_axis.set_title(
        "RRC Shaped Baseband"
    )

    shaped_axis.set_xlabel(
        "Time (ns)"
    )

    shaped_axis.set_ylabel(
        "Amplitude"
    )

    shaped_axis.set_ylim(
        -1.5,
        1.5,
    )

    # =============================================================
    # 4. RRC-BPSK TRANSMITTED SIGNAL
    # =============================================================

    waveform_axis = figure.add_subplot(
        grid[2, 1]
    )

    bpsk_line, = waveform_axis.plot(
        [],
        [],
    )

    waveform_axis.set_title(
        "RRC-BPSK Transmitted Signal"
    )

    waveform_axis.set_xlabel(
        "Time (ns)"
    )

    waveform_axis.set_ylabel(
        "Amplitude"
    )

    waveform_axis.set_ylim(
        -2.5,
        2.5,
    )

    bit_text = waveform_axis.text(
        0.02,
        0.90,
        "",
        transform=waveform_axis.transAxes,
        fontsize=14,
        verticalalignment="top",
        bbox=dict(
            facecolor="white",
            alpha=0.8,
        ),
    )

    # =============================================================
    # SIMULATION LOOP
    # =============================================================

    frame = 0

    while simulation_space.is_running():

        # ---------------------------------------------------------
        # Transmitter
        # ---------------------------------------------------------

        transmitter.transmit()

        # ---------------------------------------------------------
        # Wave propagation
        # ---------------------------------------------------------

        wave_solver.solve()

        # ---------------------------------------------------------
        # Update plots
        # ---------------------------------------------------------

        if frame % 1 == 0:

            # =====================================================
            # ELECTROMAGNETIC FIELD
            # =====================================================

            image.set_data(
                simulation_space.get_current_field().T
            )

            # =====================================================
            # GET TRANSMITTER DATA
            # =====================================================

            time_values = np.asarray(
                transmitter.get_time_values()
            )

            bit_values = np.asarray(
                transmitter.get_bit_values()
            )

            shaped_values = np.asarray(
                transmitter.get_shaped_values()
            )

            bpsk_values = np.asarray(
                transmitter.get_bpsk_values()
            )

            # =====================================================
            # WAVEFORMS
            # =====================================================

            if len(time_values) > 0:

                time_ns = time_values * 1e9

                # -------------------------------------------------
                # Convert:
                #
                # bit 0 -> +1
                # bit 1 -> -1
                #
                # This gives the original rectangular BPSK
                # symbol waveform.
                # -------------------------------------------------

                symbol_values = np.where(
                    bit_values == 0,
                    1.0,
                    -1.0,
                )

                # -------------------------------------------------
                # Original square-wave BPSK
                # -------------------------------------------------

                bit_line.set_data(
                    time_ns,
                    symbol_values,
                )

                # -------------------------------------------------
                # RRC shaped baseband
                # -------------------------------------------------

                shaped_line.set_data(
                    time_ns,
                    shaped_values,
                )

                # -------------------------------------------------
                # RRC-BPSK carrier waveform
                # -------------------------------------------------

                bpsk_line.set_data(
                    time_ns,
                    bpsk_values,
                )

                # =================================================
                # ROLLING TIME WINDOW
                # =================================================

                current_time_ns = time_ns[-1]

                x_min = max(
                    0.0,
                    current_time_ns - 10.0,
                )

                x_max = max(
                    10.0,
                    current_time_ns,
                )

                bit_axis.set_xlim(
                    x_min,
                    x_max,
                )

                shaped_axis.set_xlim(
                    x_min,
                    x_max,
                )

                waveform_axis.set_xlim(
                    x_min,
                    x_max,
                )

                # =================================================
                # CURRENT BIT
                # =================================================

                bit_text.set_text(
                    f"Current Bit : {bit_values[-1]}"
                )

            # -----------------------------------------------------
            # Refresh figure
            # -----------------------------------------------------

            figure.canvas.draw_idle()
            figure.canvas.flush_events()

        # ---------------------------------------------------------
        # Advance simulation time
        # ---------------------------------------------------------

        simulation_space.advance_time()

        frame += 1

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
