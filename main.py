import numpy as np
import matplotlib.pyplot as plt

from src.simulation_space import SimulationSpace
from src.wave_solver import WaveSolver
from src.transmitter import Transmitter
from src.receiver import Receiver


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

    bit_rate = 500.0e6

    transmitter = Transmitter(
        simulation_space=simulation_space,
        x=1.0,
        y=5.0,
        carrier_frequency=1.0e9,
        carrier_amplitude=2.0,
        bit_rate=bit_rate,
    )

    # NOTE: transmitter=transmitter is required -- the Receiver uses
    # it (deliberately, as an oracle simplification -- see
    # receiver.py's module docstring) for local-oscillator phase
    # compensation, symbol-timing alignment, and BER ground truth.
    receiver = Receiver(
        simulation_space=simulation_space,
        x=2.0,
        y=5.0,
        tuned_frequency=1.0e9,
        bit_rate=bit_rate,
        transmitter=transmitter,
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
    # MIDDLE (Transmitter):
    #   1. Original square-wave BPSK symbols
    #   2. RRC-shaped baseband
    #   3. RRC-BPSK transmitted signal
    #
    # RIGHT (Receiver):
    #   1. Received signal
    #   2. After band-pass filter
    #   3. After mixing
    #   4. After RRC matched filter
    # =============================================================

    figure = plt.figure(
        figsize=(20, 10),
        constrained_layout=True,
    )

    grid = figure.add_gridspec(
        4,
        3,
        width_ratios=[1.0, 1.5, 1.5],
    )

    # =============================================================
    # 1. ELECTROMAGNETIC FIELD
    # =============================================================

    field_axis = figure.add_subplot(
        grid[0:3, 0]
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
    # STATUS / BER READOUT (own subplot -- kept out of any graph
    # so it never overlaps a waveform)
    # =============================================================

    status_axis = figure.add_subplot(
        grid[3, 0]
    )

    status_axis.axis("off")

    ber_text = status_axis.text(
        0.0,
        0.9,
        "",
        transform=status_axis.transAxes,
        fontsize=13,
        verticalalignment="top",
        family="monospace",
        bbox=dict(
            facecolor="white",
            alpha=0.8,
        ),
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
    # 5. RECEIVED SIGNAL
    # =============================================================

    received_axis = figure.add_subplot(
        grid[0, 2]
    )

    received_line, = received_axis.plot(
        [],
        [],
    )

    received_axis.set_title(
        "Receiver — Received Signal"
    )

    received_axis.set_xlabel(
        "Time (ns)"
    )

    received_axis.set_ylabel(
        "Amplitude"
    )

    # =============================================================
    # 6. AFTER BAND-PASS FILTER
    # =============================================================

    filtered_axis = figure.add_subplot(
        grid[1, 2]
    )

    filtered_line, = filtered_axis.plot(
        [],
        [],
    )

    filtered_axis.set_title(
        "Receiver — After Band-Pass Filter"
    )

    filtered_axis.set_xlabel(
        "Time (ns)"
    )

    filtered_axis.set_ylabel(
        "Amplitude"
    )

    # =============================================================
    # 7. AFTER MIXING
    # =============================================================

    mixed_axis = figure.add_subplot(
        grid[2, 2]
    )

    mixed_line, = mixed_axis.plot(
        [],
        [],
    )

    mixed_axis.set_title(
        "Receiver — After Mixing"
    )

    mixed_axis.set_xlabel(
        "Time (ns)"
    )

    mixed_axis.set_ylabel(
        "Amplitude"
    )

    # =============================================================
    # 8. AFTER RRC MATCHED FILTER
    # =============================================================

    baseband_axis = figure.add_subplot(
        grid[3, 2]
    )

    baseband_line, = baseband_axis.plot(
        [],
        [],
    )

    baseband_axis.set_title(
        "Receiver — After RRC Matched Filter"
    )

    baseband_axis.set_xlabel(
        "Time (ns)"
    )

    baseband_axis.set_ylabel(
        "Amplitude"
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
        # Receiver
        # ---------------------------------------------------------

        receiver.receive()

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

            # =====================================================
            # GET RECEIVER DATA
            # =====================================================

            rx_time_values = np.asarray(
                receiver.get_observation_times()
            )

            received_values = np.asarray(
                receiver.get_received_values()
            )

            filtered_values = np.asarray(
                receiver.get_filtered_values()
            )

            mixed_values = np.asarray(
                receiver.get_mixed_values()
            )

            baseband_values = np.asarray(
                receiver.get_baseband_values()
            )

            # =====================================================
            # RECEIVER WAVEFORMS
            # =====================================================

            if len(rx_time_values) > 0:

                rx_time_ns = rx_time_values * 1e9

                # -------------------------------------------------
                # Received signal
                # -------------------------------------------------

                received_line.set_data(
                    rx_time_ns,
                    received_values,
                )

                # -------------------------------------------------
                # After band-pass filter
                # -------------------------------------------------

                filtered_line.set_data(
                    rx_time_ns,
                    filtered_values,
                )

                # -------------------------------------------------
                # After mixing
                # -------------------------------------------------

                mixed_line.set_data(
                    rx_time_ns,
                    mixed_values,
                )

                # -------------------------------------------------
                # After RRC matched filter
                # -------------------------------------------------

                baseband_line.set_data(
                    rx_time_ns,
                    baseband_values,
                )

                # =================================================
                # ROLLING TIME WINDOW (RECEIVER)
                # =================================================

                rx_current_time_ns = rx_time_ns[-1]

                rx_x_min = max(
                    0.0,
                    rx_current_time_ns - 10.0,
                )

                rx_x_max = max(
                    10.0,
                    rx_current_time_ns,
                )

                received_axis.set_xlim(
                    rx_x_min,
                    rx_x_max,
                )

                filtered_axis.set_xlim(
                    rx_x_min,
                    rx_x_max,
                )

                mixed_axis.set_xlim(
                    rx_x_min,
                    rx_x_max,
                )

                baseband_axis.set_xlim(
                    rx_x_min,
                    rx_x_max,
                )

                # =================================================
                # Y-AXIS AUTOSCALE (RECEIVER)
                #
                # Unlike the transmitter panels, receiver signal
                # magnitudes depend on distance/filters and aren't
                # fixed ahead of time -- rescale to whatever the
                # data actually spans.
                # =================================================

                for axis, values in (
                    (received_axis, received_values),
                    (filtered_axis, filtered_values),
                    (mixed_axis, mixed_values),
                    (baseband_axis, baseband_values),
                ):

                    if len(values) == 0:
                        continue

                    value_min = np.min(values)
                    value_max = np.max(values)

                    if value_min == value_max:
                        margin = max(abs(value_min) * 0.1, 1e-12)
                    else:
                        margin = 0.1 * (value_max - value_min)

                    axis.set_ylim(
                        value_min - margin,
                        value_max + margin,
                    )

                # =================================================
                # BIT ERROR RATE
                # =================================================

                bit_error_rate = receiver.get_bit_error_rate()

                ber_display = (
                    f"{bit_error_rate:.4f}"
                    if bit_error_rate is not None
                    else "n/a"
                )

                ber_text.set_text(
                    f"t = {simulation_space.time * 1e9:.2f} ns\n"
                    f"Bits Compared : {receiver.get_total_bits_compared()}\n"
                    f"Bit Errors    : {receiver.get_bit_errors()}\n"
                    f"BER           : {ber_display}\n"
                    f"Est. Delay    : "
                    f"{receiver.get_estimated_total_delay_seconds()*1e9:.2f} ns"
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