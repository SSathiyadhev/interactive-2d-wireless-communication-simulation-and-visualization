import numpy as np
import matplotlib.pyplot as plt

from src.simulation_space import SimulationSpace
from src.wave_solver import WaveSolver
from src.transmitter import Transmitter
from src.receiver import Receiver


def main():

    # =========================================================
    # SIMULATION SETTINGS — UNCHANGED
    # =========================================================

    resolution_x = 600
    resolution_y = 600

    dx = 10.0 / (resolution_x - 1)
    dy = 10.0 / (resolution_y - 1)

    c = 3.0e8

    dt = 0.75 / (
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

    # =========================================================
    # TRANSMITTER — UNCHANGED
    # =========================================================

    transmitter = Transmitter(
        simulation_space=simulation_space,
        x=1.0,
        y=5.0,
        carrier_frequency=1.0e9,
        carrier_amplitude=2.0,
        bit_rate=500.0e6,
        window_duration=20e-9,
    )

    # =========================================================
    # RECEIVER — UNCHANGED
    # =========================================================

    receiver = Receiver(
        simulation_space=simulation_space,
        x=2.0,
        y=5.0,
        tuned_frequency=1.0e9,
        bit_rate=500.0e6,
        observation_window=20e-9,
    )

    # =========================================================
    # WAVE SOLVER — UNCHANGED
    # =========================================================

    wave_solver = WaveSolver(simulation_space)

    simulation_space.set_running(True)

    # =========================================================
    # PLOT
    # =========================================================

    plt.ion()

    figure, axes = plt.subplots(
        4,
        2,
        figsize=(15, 12),
    )

    # ---------------------------------------------------------
    # Transmitter
    # ---------------------------------------------------------

    bit_axis = axes[0, 0]
    shaped_axis = axes[1, 0]
    bpsk_axis = axes[2, 0]

    # ---------------------------------------------------------
    # Receiver
    # ---------------------------------------------------------

    received_axis = axes[0, 1]
    filtered_axis = axes[1, 1]
    mixed_axis = axes[2, 1]
    baseband_axis = axes[3, 0]

    # Unused
    axes[3, 1].axis("off")

    # =========================================================
    # LINES
    # =========================================================

    bit_line, = bit_axis.plot(
        [],
        [],
        drawstyle="steps-post",
    )

    shaped_line, = shaped_axis.plot(
        [],
        [],
    )

    bpsk_line, = bpsk_axis.plot(
        [],
        [],
    )

    received_line, = received_axis.plot(
        [],
        [],
    )

    filtered_line, = filtered_axis.plot(
        [],
        [],
    )

    mixed_line, = mixed_axis.plot(
        [],
        [],
    )

    baseband_line, = baseband_axis.plot(
        [],
        [],
    )

    # =========================================================
    # TITLES
    # =========================================================

    bit_axis.set_title(
        "Transmitter — Original BPSK Symbols"
    )

    shaped_axis.set_title(
        "Transmitter — RRC Shaped Baseband"
    )

    bpsk_axis.set_title(
        "Transmitter — RRC-BPSK"
    )

    received_axis.set_title(
        "Receiver — Received Signal"
    )

    filtered_axis.set_title(
        "Receiver — After Band-Pass Filter"
    )

    mixed_axis.set_title(
        "Receiver — After Mixing"
    )

    baseband_axis.set_title(
        "Receiver — After RRC Matched Filter"
    )

    # =========================================================
    # AXIS SETTINGS
    # =========================================================

    for axis in axes.flat:

        if axis.axison:

            axis.set_xlabel("Time (ns)")
            axis.set_ylabel("Amplitude")
            axis.grid(True)

    # Original symbols are fixed at +1 / -1
    bit_axis.set_ylim(
        -1.5,
        1.5,
    )

    figure.tight_layout()

    # =========================================================
    # Y-SCALE HELPER
    # =========================================================

    def autoscale_y(axis, values):

        if len(values) == 0:
            return

        value_min = np.min(values)
        value_max = np.max(values)

        if value_min == value_max:

            margin = max(
                abs(value_min) * 0.1,
                1e-12,
            )

        else:

            margin = 0.1 * (
                value_max - value_min
            )

        axis.set_ylim(
            value_min - margin,
            value_max + margin,
        )

    # =========================================================
    # SIMULATION LOOP
    # =========================================================

    frame = 20

    while simulation_space.is_running():

        # -----------------------------------------------------
        # Transmit
        # -----------------------------------------------------

        transmitter.transmit()

        # -----------------------------------------------------
        # Propagate
        # -----------------------------------------------------

        wave_solver.solve()

        # -----------------------------------------------------
        # Receive
        # -----------------------------------------------------

        receiver.receive()

        # =====================================================
        # UPDATE PLOTS
        # =====================================================

        if frame % 1 == 0:

            # =================================================
            # TRANSMITTER DATA
            # =================================================

            tx_time = np.asarray(
                transmitter.get_time_values()
            )

            tx_bits = np.asarray(
                transmitter.get_bit_values()
            )

            tx_shaped = np.asarray(
                transmitter.get_shaped_values()
            )

            tx_bpsk = np.asarray(
                transmitter.get_bpsk_values()
            )

            if len(tx_time) > 0:

                tx_time_ns = tx_time * 1e9

                # -------------------------------------------------
                # Convert bits to bipolar BPSK symbols
                #
                # bit 0 -> +1
                # bit 1 -> -1
                # -------------------------------------------------

                tx_symbols = np.where(
                    tx_bits == 0,
                    1.0,
                    -1.0,
                )

                # -------------------------------------------------
                # Original square symbol waveform
                # -------------------------------------------------

                bit_line.set_data(
                    tx_time_ns,
                    tx_symbols,
                )

                # -------------------------------------------------
                # RRC shaped baseband
                # -------------------------------------------------

                shaped_line.set_data(
                    tx_time_ns,
                    tx_shaped,
                )

                # -------------------------------------------------
                # RRC-BPSK carrier
                # -------------------------------------------------

                bpsk_line.set_data(
                    tx_time_ns,
                    tx_bpsk,
                )

                # -------------------------------------------------
                # Rolling X axis
                # -------------------------------------------------

                current_time_ns = tx_time_ns[-1]

                xmin = max(
                    0.0,
                    current_time_ns - 20.0,
                )

                xmax = max(
                    20.0,
                    current_time_ns,
                )

                bit_axis.set_xlim(
                    xmin,
                    xmax,
                )

                shaped_axis.set_xlim(
                    xmin,
                    xmax,
                )

                bpsk_axis.set_xlim(
                    xmin,
                    xmax,
                )

                # -------------------------------------------------
                # Y scaling
                # -------------------------------------------------

                autoscale_y(
                    shaped_axis,
                    tx_shaped,
                )

                autoscale_y(
                    bpsk_axis,
                    tx_bpsk,
                )

            # =================================================
            # RECEIVER DATA
            # =================================================

            rx_time = np.asarray(
                receiver.get_observation_times()
            )

            received = np.asarray(
                receiver.get_received_values()
            )

            filtered = np.asarray(
                receiver.get_filtered_values()
            )

            mixed = np.asarray(
                receiver.get_mixed_values()
            )

            baseband = np.asarray(
                receiver.get_baseband_values()
            )

            if len(rx_time) > 0:

                rx_time_ns = rx_time * 1e9

                # -------------------------------------------------
                # Received
                # -------------------------------------------------

                received_line.set_data(
                    rx_time_ns,
                    received,
                )

                # -------------------------------------------------
                # BPF
                # -------------------------------------------------

                filtered_line.set_data(
                    rx_time_ns,
                    filtered,
                )

                # -------------------------------------------------
                # Mixer
                # -------------------------------------------------

                mixed_line.set_data(
                    rx_time_ns,
                    mixed,
                )

                # -------------------------------------------------
                # Matched filter
                # -------------------------------------------------

                baseband_line.set_data(
                    rx_time_ns,
                    baseband,
                )

                # -------------------------------------------------
                # Rolling X axis
                # -------------------------------------------------

                current_time_ns = rx_time_ns[-1]

                xmin = max(
                    0.0,
                    current_time_ns - 20.0,
                )

                xmax = max(
                    20.0,
                    current_time_ns,
                )

                received_axis.set_xlim(
                    xmin,
                    xmax,
                )

                filtered_axis.set_xlim(
                    xmin,
                    xmax,
                )

                mixed_axis.set_xlim(
                    xmin,
                    xmax,
                )

                baseband_axis.set_xlim(
                    xmin,
                    xmax,
                )

                # -------------------------------------------------
                # Y scaling
                # -------------------------------------------------

                autoscale_y(
                    received_axis,
                    received,
                )

                autoscale_y(
                    filtered_axis,
                    filtered,
                )

                autoscale_y(
                    mixed_axis,
                    mixed,
                )

                autoscale_y(
                    baseband_axis,
                    baseband,
                )

            # =================================================
            # REDRAW
            # =================================================

            figure.canvas.draw_idle()
            figure.canvas.flush_events()

        # =====================================================
        # ADVANCE ONE DT
        # =====================================================

        simulation_space.advance_time()

        frame += 1

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
