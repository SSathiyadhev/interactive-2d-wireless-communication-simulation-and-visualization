import numpy as np
import matplotlib.pyplot as plt

from src.simulation_space import SimulationSpace
from src.wave_solver import WaveSolver
from src.transmitter import Transmitter
from src.receiver import Receiver
from src.link_evaluator import LinkEvaluator
<<<<<<< Updated upstream
from src.observation_point import ObservationPoint
=======
>>>>>>> Stashed changes


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

<<<<<<< Updated upstream
    # Receiver list (Baseline primary receiver is index 0 at x=2.0)
    receivers = [
        Receiver(simulation_space, x=2.0, y=5.0, tuned_frequency=1.0e9, bit_rate=bit_rate),
        Receiver(simulation_space, x=3.5, y=5.0, tuned_frequency=1.0e9, bit_rate=bit_rate),
    ]

    evaluators = []
    for rx in receivers:
        ev = LinkEvaluator(
            transmitter=transmitter,
            receiver=rx,
            speed_of_light=c,
            filter_group_delay_samples=8,
            warmup_bits=2,
        )
        ev.sync_receiver_delay()
        evaluators.append(ev)

    active_rx = [0]

    # ObservationPoint for spectral analysis and energy tracking
    observation_point = ObservationPoint(
        simulation_space=simulation_space,
        x=2.0,
        y=5.0,
        buffer_duration=15e-9,
        label="RX Point",
    )

    wave_solver = WaveSolver(simulation_space, noise_level=0)
=======
    receiver = Receiver(
        simulation_space=simulation_space,
        x=2.0,
        y=5.0,
        tuned_frequency=1.0e9,
        bit_rate=bit_rate,
    )

    # Total filter delay = 4 symbols (TX RRC) + 4 symbols (RX RRC) = 8 symbols
    evaluator = LinkEvaluator(
        transmitter=transmitter,
        receiver=receiver,
        speed_of_light=c,
        filter_group_delay_samples=8,
        warmup_bits=2,
    )
    evaluator.sync_receiver_delay()

    wave_solver = WaveSolver(simulation_space)
>>>>>>> Stashed changes

    simulation_space.set_running(True)

    plt.ion()

<<<<<<< Updated upstream
    # =============================================================
    # FIGURE LAYOUT (ORIGINAL CLEAN MATPLOTLIB STRUCTURE)
    # =============================================================

=======
>>>>>>> Stashed changes
    figure = plt.figure(
        figsize=(22, 11),
        constrained_layout=True,
    )

    grid = figure.add_gridspec(
        4,
        3,
        width_ratios=[1.0, 1.5, 1.5],
    )

<<<<<<< Updated upstream
    # =============================================================
    # 1. ELECTROMAGNETIC FIELD (Original dimensions, no clutter)
    # =============================================================

=======
    # 1. ELECTROMAGNETIC FIELD
>>>>>>> Stashed changes
    field_axis = figure.add_subplot(grid[0:3, 0])
    image = field_axis.imshow(
        simulation_space.get_current_field().T,
        cmap="RdBu_r",
        origin="lower",
        interpolation="nearest",
        vmin=-2.0,
        vmax=2.0,
    )
    figure.colorbar(image, ax=field_axis)
    field_axis.set_title("Electromagnetic Field")
    field_axis.set_xlabel("X")
    field_axis.set_ylabel("Y")

<<<<<<< Updated upstream
    # =============================================================
    # STATUS / BER & ENERGY READOUT
    # =============================================================

=======
    # STATUS / BER READOUT
>>>>>>> Stashed changes
    status_axis = figure.add_subplot(grid[3, 0])
    status_axis.axis("off")
    ber_text = status_axis.text(
        0.0,
        0.95,
        "",
        transform=status_axis.transAxes,
        fontsize=12,
        verticalalignment="top",
        family="monospace",
        bbox=dict(facecolor="white", alpha=0.8),
    )

<<<<<<< Updated upstream
    # =============================================================
    # TRANSMITTER PLOTS
    # =============================================================

    # 2. Original square-wave BPSK symbols
    bit_axis = figure.add_subplot(grid[0, 1])
    bit_line, = bit_axis.plot([], [], drawstyle="steps-post")
    bit_axis.set_title("Original BPSK Symbol Sequence")
    bit_axis.set_xlabel("Time (ns)")
    bit_axis.set_ylabel("Symbol")
    bit_axis.set_ylim(-1.5, 1.5)

    # 3. RRC shaped baseband
=======
    # 2. ORIGINAL SQUARE-WAVE BPSK SYMBOLS
    bit_axis = figure.add_subplot(grid[0, 1])
    bit_line, = bit_axis.plot([], [], drawstyle="steps-post")
    bit_axis.set_title("Original BPSK Symbol Sequence")
    bit_axis.set_xlabel("Time (ns)")
    bit_axis.set_ylabel("Symbol")
    bit_axis.set_ylim(-1.5, 1.5)

    # 3. RRC SHAPED BASEBAND
>>>>>>> Stashed changes
    shaped_axis = figure.add_subplot(grid[1, 1])
    shaped_line, = shaped_axis.plot([], [])
    shaped_axis.set_title("RRC Shaped Baseband")
    shaped_axis.set_xlabel("Time (ns)")
    shaped_axis.set_ylabel("Amplitude")
    shaped_axis.set_ylim(-1.5, 1.5)

<<<<<<< Updated upstream
    # 4. RRC-BPSK transmitted signal
=======
    # 4. RRC-BPSK TRANSMITTED SIGNAL
>>>>>>> Stashed changes
    waveform_axis = figure.add_subplot(grid[2, 1])
    bpsk_line, = waveform_axis.plot([], [])
    waveform_axis.set_title("RRC-BPSK Transmitted Signal")
    waveform_axis.set_xlabel("Time (ns)")
    waveform_axis.set_ylabel("Amplitude")
    waveform_axis.set_ylim(-2.5, 2.5)
    bit_text = waveform_axis.text(
        0.02,
        0.90,
        "",
        transform=waveform_axis.transAxes,
        fontsize=13,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.8),
    )

<<<<<<< Updated upstream
    # 5. Live FFT Spectrum at Observation Point
    fft_axis = figure.add_subplot(grid[3, 1])
    fft_line, = fft_axis.plot([], [], color="crimson")
    fft_axis.set_title("Observation Point — Field Spectrum (FFT)")
    fft_axis.set_xlabel("Frequency (GHz)")
    fft_axis.set_ylabel("Magnitude")
    fft_axis.set_xlim(0.0, 3.0)
    fft_axis.set_ylim(0.0, 1.0)
    fft_axis.grid(True, linestyle="--", alpha=0.5)

    # =============================================================
    # RECEIVER PLOTS
    # =============================================================

    # 6. Received signal
    received_axis = figure.add_subplot(grid[0, 2])
    received_line, = received_axis.plot([], [])
    rx_title = received_axis.set_title("Receiver — Received Signal")
    received_axis.set_xlabel("Time (ns)")
    received_axis.set_ylabel("Amplitude")

    # 7. After band-pass filter
=======
    # 5. RECEIVED SIGNAL
    received_axis = figure.add_subplot(grid[0, 2])
    received_line, = received_axis.plot([], [])
    received_axis.set_title("Receiver — Received Signal")
    received_axis.set_xlabel("Time (ns)")
    received_axis.set_ylabel("Amplitude")

    # 6. AFTER BAND-PASS FILTER
>>>>>>> Stashed changes
    filtered_axis = figure.add_subplot(grid[1, 2])
    filtered_line, = filtered_axis.plot([], [])
    filtered_axis.set_title("Receiver — After Band-Pass Filter")
    filtered_axis.set_xlabel("Time (ns)")
    filtered_axis.set_ylabel("Amplitude")
<<<<<<< Updated upstream

    # 8. After mixing
    mixed_axis = figure.add_subplot(grid[2, 2])
    mixed_line, = mixed_axis.plot([], [])
    mixed_axis.set_title("Receiver — After Mixing")
    mixed_axis.set_xlabel("Time (ns)")
    mixed_axis.set_ylabel("Amplitude")

    # 9. After RRC matched filter
    baseband_axis = figure.add_subplot(grid[3, 2])
    baseband_line, = baseband_axis.plot([], [])
    baseband_axis.set_title("Receiver — After RRC Matched Filter")
    baseband_axis.set_xlabel("Time (ns)")
    baseband_axis.set_ylabel("Amplitude")

    # =============================================================
    # INTERACTIVE SWITCH (Click anywhere near receiver or press keys)
    # =============================================================
    def on_click(event):
        if event.inaxes == field_axis and event.xdata is not None and event.ydata is not None:
            # Map pixel click coordinates back to physical meters (dx = 10 / 599)
            click_phys_x = event.xdata * dx
            click_phys_y = event.ydata * dy
            dists = [np.hypot(r.x - click_phys_x, r.y - click_phys_y) for r in receivers]
            nearest = int(np.argmin(dists))
            if dists[nearest] < 1.0:
                active_rx[0] = nearest

    def on_key(event):
        if event.key in [str(i + 1) for i in range(len(receivers))]:
            active_rx[0] = int(event.key) - 1

    figure.canvas.mpl_connect("button_press_event", on_click)
    figure.canvas.mpl_connect("key_press_event", on_key)

    frame = 50

    while simulation_space.is_running():

        transmitter.transmit()
        wave_solver.solve()

        for rx in receivers:
            rx.receive()
        for ev in evaluators:
            ev.evaluate()

        observation_point.sample()
=======

    # 7. AFTER MIXING
    mixed_axis = figure.add_subplot(grid[2, 2])
    mixed_line, = mixed_axis.plot([], [])
    mixed_axis.set_title("Receiver — After Mixing")
    mixed_axis.set_xlabel("Time (ns)")
    mixed_axis.set_ylabel("Amplitude")

    # 8. AFTER RRC MATCHED FILTER
    baseband_axis = figure.add_subplot(grid[3, 2])
    baseband_line, = baseband_axis.plot([], [])
    baseband_axis.set_title("Receiver — After RRC Matched Filter")
    baseband_axis.set_xlabel("Time (ns)")
    baseband_axis.set_ylabel("Amplitude")

    frame = 0

    while simulation_space.is_running():
        transmitter.transmit()
        wave_solver.solve()
        receiver.receive()

        evaluator.evaluate()
>>>>>>> Stashed changes

        if frame % 1 == 0:
            image.set_data(simulation_space.get_current_field().T)

            # Transmitter Data
            time_values = np.asarray(transmitter.get_time_values())
            bit_values = np.asarray(transmitter.get_bit_values())
            shaped_values = np.asarray(transmitter.get_shaped_values())
            bpsk_values = np.asarray(transmitter.get_bpsk_values())

            if len(time_values) > 0:
                time_ns = time_values * 1e9
                symbol_values = np.where(bit_values == 0, 1.0, -1.0)

                bit_line.set_data(time_ns, symbol_values)
                shaped_line.set_data(time_ns, shaped_values)
                bpsk_line.set_data(time_ns, bpsk_values)

                current_time_ns = time_ns[-1]
                x_min = max(0.0, current_time_ns - 10.0)
                x_max = max(10.0, current_time_ns)

                bit_axis.set_xlim(x_min, x_max)
                shaped_axis.set_xlim(x_min, x_max)
                waveform_axis.set_xlim(x_min, x_max)

                bit_text.set_text(f"Current Bit : {bit_values[-1]}")

<<<<<<< Updated upstream
            # Observation Point FFT Data
            freqs, amps = observation_point.compute_fft(window_type="hann")
            if len(freqs) > 0:
                freqs_ghz = freqs / 1e9
                mask = freqs_ghz <= 3.0
                fft_line.set_data(freqs_ghz[mask], amps[mask])
                max_amp = np.max(amps[mask]) if np.any(mask) else 1.0
                fft_axis.set_ylim(0.0, max(max_amp * 1.2, 0.05))

            # Active Receiver Binding
            current_rx = receivers[active_rx[0]]
            current_ev = evaluators[active_rx[0]]

            rx_title.set_text(f"Receiver {active_rx[0] + 1} ({current_rx.x:.2f}m) — Received Signal")

            # Receiver Data
            rx_time_values = np.asarray(current_rx.get_observation_times())
            received_values = np.asarray(current_rx.get_received_values())
            filtered_values = np.asarray(current_rx.get_filtered_values())
            mixed_values = np.asarray(current_rx.get_mixed_values())
            baseband_values = np.asarray(current_rx.get_baseband_values())
=======
            # Receiver Data
            rx_time_values = np.asarray(receiver.get_observation_times())
            received_values = np.asarray(receiver.get_received_values())
            filtered_values = np.asarray(receiver.get_filtered_values())
            mixed_values = np.asarray(receiver.get_mixed_values())
            baseband_values = np.asarray(receiver.get_baseband_values())
>>>>>>> Stashed changes

            if len(rx_time_values) > 0:
                rx_time_ns = rx_time_values * 1e9

                received_line.set_data(rx_time_ns, received_values)
                filtered_line.set_data(rx_time_ns, filtered_values)
                mixed_line.set_data(rx_time_ns, mixed_values)
                baseband_line.set_data(rx_time_ns, baseband_values)

                rx_current_time_ns = rx_time_ns[-1]
                rx_x_min = max(0.0, rx_current_time_ns - 10.0)
                rx_x_max = max(10.0, rx_current_time_ns)

                received_axis.set_xlim(rx_x_min, rx_x_max)
                filtered_axis.set_xlim(rx_x_min, rx_x_max)
                mixed_axis.set_xlim(rx_x_min, rx_x_max)
                baseband_axis.set_xlim(rx_x_min, rx_x_max)

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

                    axis.set_ylim(value_min - margin, value_max + margin)

<<<<<<< Updated upstream
                # Performance and Energy Telemetry Readout
                bit_error_rate = current_ev.get_bit_error_rate()
                ber_display = (
                    f"{bit_error_rate:.4f}"
                    if current_ev.get_total_bits_compared() > 0
=======
                bit_error_rate = evaluator.get_bit_error_rate()
                ber_display = (
                    f"{bit_error_rate:.4f}"
                    if evaluator.get_total_bits_compared() > 0
>>>>>>> Stashed changes
                    else "n/a"
                )

                peak_f, peak_a = observation_point.get_peak_frequency(window_type="hann")
                window_energy = observation_point.get_windowed_energy()

                ber_text.set_text(
                    f"Selected RX   : #{active_rx[0] + 1} ({current_rx.x:.2f} m, {current_rx.y:.2f} m)\n"
                    f"t = {simulation_space.time * 1e9:.2f} ns\n"
<<<<<<< Updated upstream
                    f"Bits Compared : {current_ev.get_total_bits_compared()}\n"
                    f"Bit Errors    : {current_ev.get_bit_errors()}\n"
                    f"BER           : {ber_display}\n"
                    f"Est. Delay    : "
                    f"{current_ev.get_estimated_total_delay_seconds() * 1e9:.2f} ns\n"
                    f"-------------------------\n"
                    f"Peak Freq     : {peak_f / 1e9:.3f} GHz\n"
                    f"Peak Mag      : {peak_a:.3e}\n"
                    f"Window Energy : {window_energy:.3e} J"
=======
                    f"Bits Compared : {evaluator.get_total_bits_compared()}\n"
                    f"Bit Errors    : {evaluator.get_bit_errors()}\n"
                    f"BER           : {ber_display}\n"
                    f"Est. Delay    : "
                    f"{evaluator.get_estimated_total_delay_seconds() * 1e9:.2f} ns"
>>>>>>> Stashed changes
                )

            figure.canvas.draw_idle()
            figure.canvas.flush_events()

        simulation_space.advance_time()
        frame += 1

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()