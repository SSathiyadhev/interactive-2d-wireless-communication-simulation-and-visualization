"""
src/receiver.py

Defines the Receiver class using a Costas loop for blind carrier phase tracking
and a Gardner loop for symbol-timing recovery.

Recovered symbols are hard-decoded into bits and exposed to LinkEvaluator.
"""

from collections import deque
import numpy as np

from src.filter import Filter
from src.costas_loop import CostasLoop
from src.gardner_loop import GardnerLoop


class Receiver:

    def __init__(
        self,
        simulation_space,     # SimulationSpace used by the receiver
        x,                    # Receiver physical x-coordinate
        y,                    # Receiver physical y-coordinate
        tuned_frequency,      # Carrier frequency to receive
        bit_rate,             # Transmitted bit rate in bits/second
        observation_window=10e-9,  # Duration of stored observation data
        fft_window=30e-9, #fft sample window
        rrc_rolloff=0.35,
        rrc_span=8,
        **kwargs,
    ):
        self.simulation_space = simulation_space
        self.x = float(x)
        self.y = float(y)
        self.tuned_frequency = float(tuned_frequency)
        self.bit_rate = float(bit_rate)
        self.rrc_rolloff = float(rrc_rolloff)
        self.rrc_span = int(rrc_span)
        self.observation_window = float(observation_window)

        self.fft_window = float(fft_window)
        self.fft_window_type = "hann"
        self.fft_zero_padding_factor = 4

        if self.fft_window <= 0:
            raise ValueError("FFT window must be strictly positive.")

        if not self.simulation_space.is_inside(self.x, self.y):
            raise ValueError(
                f"Receiver position ({self.x} m, {self.y} m) "
                f"is outside simulation space."
            )

        if self.bit_rate <= 0:
            raise ValueError("Bit rate must be greater than zero.")

        # Number of simulation samples that fit in the observation window
        max_samples = max(
            1,
            int(np.ceil(self.observation_window / self.simulation_space.dt)),
        )

        # Rolling sample-rate buffers (for oscilloscope UI visualization)
        self.time_values = deque(maxlen=max_samples)
        self.received_values = deque(maxlen=max_samples)
        self.filtered_values = deque(maxlen=max_samples)
        self.mixed_values = deque(maxlen=max_samples)
        self.baseband_values = deque(maxlen=max_samples)
        self.bit_values = deque(maxlen=max_samples)

        # Currently decoded bit
        self.current_bit = None

        fft_samples = max(
            8,
            int(np.ceil(
                self.fft_window / self.simulation_space.dt
            )),
        )

        self.filtered_fft_values = deque(maxlen=fft_samples)

        # Unbounded append-only stream of decoded bits for LinkEvaluator
        self.demodulated_bits = []


        # Simulation time at which each bit decision was made.
        self.demodulated_bit_times = []

        # Front-end RF Band-pass filter
        rf_bandwidth = (
            0.5
            * self.bit_rate
            * (1.0 + self.rrc_rolloff)
            * 1.10
        )

        self.bandpass_filter = Filter(
            "butterworth",
            self.simulation_space.dt,
            order=4,
            filter_response="bandpass",
            low_cutoff_frequency=self.tuned_frequency - rf_bandwidth,
            high_cutoff_frequency=self.tuned_frequency + rf_bandwidth,
        )

        self._samples_per_symbol = max(
            1,
            int(round((1.0 / self.bit_rate) / self.simulation_space.dt)),
        )

        # RRC matched baseband filter
        self.matched_filter = Filter(
            "rrc",
            self.simulation_space.dt,
            rolloff=self.rrc_rolloff,
            samples_per_symbol=self._samples_per_symbol,
            span=self.rrc_span,
            normalize="energy",
        )

        # Gardner symbol-timing recovery
        self.gardner_loop = GardnerLoop(
            dt=self.simulation_space.dt,
            symbol_rate=self.bit_rate
        )

        # Costas Loop for carrier phase tracking
        self.costas_loop = CostasLoop(
            dt=self.simulation_space.dt,
            carrier_frequency=self.tuned_frequency,
            bit_rate=self.bit_rate,
            rrc_rolloff=self.rrc_rolloff,
        )
        
    def _sample_field(self):
        t = self.simulation_space.time
        received_value = self.simulation_space.get_field(self.x, self.y)
        self.time_values.append(t)
        self.received_values.append(received_value)
        return t, received_value

    def _filter_signal(self, received_value):
        filtered_value = self.bandpass_filter.filter(received_value)
        self.filtered_values.append(filtered_value)
        self.filtered_fft_values.append(filtered_value)
        return filtered_value

    def _matched_filter_stage(self, mixed_value):
        baseband_value = self.matched_filter.filter(mixed_value)
        self.baseband_values.append(baseband_value)
        return baseband_value

    def _decide_bit(self, recovered_symbol, decision_time):
        """
        Makes a BPSK hard decision from a Gardner-recovered symbol
        and records the simulation time of the decision.
        """

        decoded_bit = (
            0
            if recovered_symbol >= 0.0
            else 1
        )

        self.current_bit = decoded_bit
        self.demodulated_bits.append(decoded_bit)
        self.demodulated_bit_times.append(decision_time)

    def receive(self):
        """Processes one simulation timestep."""

        current_time, received_value = self._sample_field()

        filtered_value = self._filter_signal(
            received_value
        )

        # Carrier recovery / downconversion
        mixed_value, _ = self.costas_loop.process(
            filtered_value,
            current_time,
        )

        self.mixed_values.append(
            mixed_value
        )

        # RRC matched filter
        baseband_value = self._matched_filter_stage(
            mixed_value
        )

        # Gardner symbol timing recovery
        recovered_symbol = self.gardner_loop.process(
            baseband_value
        )

        # BPSK decision
        if recovered_symbol is not None:
            self._decide_bit(
                recovered_symbol,
                current_time
            )

        # Store latest decoded bit for visualization
        self.bit_values.append(
            self.current_bit
        )

    def _design_filter(self):
        rf_bandwidth = (
            0.5
            * self.bit_rate
            * (1.0 + self.rrc_rolloff)
            * 1.5
        )

        self.bandpass_filter.set_parameters(
            order=4,
            filter_response="bandpass",
            low_cutoff_frequency=self.tuned_frequency - rf_bandwidth,
            high_cutoff_frequency=self.tuned_frequency + rf_bandwidth,
        )

    def compute_filtered_fft(self):
        """
        Computes the single-sided amplitude spectrum of the
        RF band-pass filtered signal.

        Returns:
            frequencies
            amplitudes
            frequency_bin_spacing
            true_frequency_resolution
        """

        samples = np.asarray(
            self.filtered_fft_values,
            dtype=np.float64,
        )

        n = len(samples)

        if n < 8:
            return (
                np.array([]),
                np.array([]),
                0.0,
                0.0,
            )

        # Window
        if self.fft_window_type == "hann":
            window = np.hanning(n)
        elif self.fft_window_type == "hamming":
            window = np.hamming(n)
        elif self.fft_window_type == "blackman":
            window = np.blackman(n)
        else:
            window = np.ones(n)

        windowed_signal = samples * window

        # Coherent gain correction
        window_sum = np.sum(window)
        coherent_gain = (
            window_sum / n
            if window_sum > 0
            else 1.0
        )

        # FFT
        n_fft = n * self.fft_zero_padding_factor
        dt = self.simulation_space.dt

        fft_values = np.fft.rfft(
            windowed_signal,
            n=n_fft,
        )

        frequencies = np.fft.rfftfreq(
            n_fft,
            d=dt,
        )

        amplitudes = (
            2.0 * np.abs(fft_values) / n
        ) / coherent_gain

        amplitudes[0] /= 2.0

        frequency_bin_spacing = (
            frequencies[1] - frequencies[0]
            if len(frequencies) > 1
            else 0.0
        )

        true_frequency_resolution = 1.0 / (n * dt)

        return (
            frequencies,
            amplitudes,
            frequency_bin_spacing,
            true_frequency_resolution,
        )

    def set_position(self, x, y):
        self.x = float(x)
        self.y = float(y)

    def get_position(self):
        return self.x, self.y

    def set_tuned_frequency(self, value):
        self.tuned_frequency = float(value)

        self._design_filter()

        self.costas_loop = CostasLoop(
            dt=self.simulation_space.dt,
            carrier_frequency=self.tuned_frequency,
            bit_rate=self.bit_rate,
            rrc_rolloff=self.rrc_rolloff,
        )

    def get_tuned_frequency(self):
        return self.tuned_frequency

    def set_bit_rate(self, value):
        self.bit_rate = float(value)

        self._design_filter()

        self._samples_per_symbol = max(
            1,
            int(
                round(
                    (1.0 / self.bit_rate)
                    / self.simulation_space.dt
                )
            ),
        )

        self.matched_filter.set_parameters(
            rolloff=self.rrc_rolloff,
            samples_per_symbol=self._samples_per_symbol,
            span=self.rrc_span,
            normalize="energy",
        )

        self.gardner_loop = GardnerLoop(
            dt=self.simulation_space.dt,
            symbol_rate=self.bit_rate,
        )

        self.costas_loop = CostasLoop(
            dt=self.simulation_space.dt,
            carrier_frequency=self.tuned_frequency,
            bit_rate=self.bit_rate,
            rrc_rolloff=self.rrc_rolloff,
        )

    def set_observation_window(self, observation_window):
        observation_window = float(observation_window)

        if observation_window <= 0:
            raise ValueError(
                "Observation window duration must be greater than zero."
            )

        self.observation_window = observation_window

        max_samples = max(
            1,
            int(
                np.ceil(
                    self.observation_window
                    / self.simulation_space.dt
                )
            ),
        )

        self.time_values = deque(maxlen=max_samples)
        self.received_values = deque(maxlen=max_samples)
        self.filtered_values = deque(maxlen=max_samples)
        self.mixed_values = deque(maxlen=max_samples)
        self.baseband_values = deque(maxlen=max_samples)
        self.bit_values = deque(maxlen=max_samples)

    def set_fft_window(self, fft_window):
        fft_window = float(fft_window)

        if fft_window <= 0:
            raise ValueError(
                "FFT window must be strictly positive."
            )

        self.fft_window = fft_window

        fft_samples = max(
            8,
            int(
                np.ceil(
                    self.fft_window
                    / self.simulation_space.dt
                )
            ),
        )

        self.filtered_fft_values = deque(
            maxlen=fft_samples
        )

    def get_bit_rate(self):
        return self.bit_rate

    def get_current_received_value(self):
        return self.received_values[-1] if self.received_values else None

    def get_received_values(self):
        return list(self.received_values)

    def get_filtered_values(self):
        return list(self.filtered_values)

    def get_mixed_values(self):
        return list(self.mixed_values)

    def get_demodulated_bits(self):
        return self.demodulated_bits

    def get_demodulated_bit_times(self):
        return self.demodulated_bit_times

    def get_baseband_values(self):
        return list(self.baseband_values)

    def get_observation_times(self):
        return list(self.time_values)

    def get_observation_window(self):
        return self.observation_window

    def get_fft_window(self):
        return self.fft_window

    def set_fft_parameters(
        self,
        window_type=None,
        zero_padding_factor=None,
    ):
        if window_type is not None:
            window_type = str(window_type).lower()

            if window_type not in (
                "rect",
                "hann",
                "hamming",
                "blackman",
            ):
                raise ValueError(
                    f"Unsupported FFT window type: {window_type}"
                )

            self.fft_window_type = window_type

        if zero_padding_factor is not None:
            zero_padding_factor = int(zero_padding_factor)

            if zero_padding_factor < 1:
                raise ValueError(
                    "Zero-padding factor must be >= 1."
                )

            self.fft_zero_padding_factor = zero_padding_factor


    def get_fft_parameters(self):
        return (
            self.fft_window_type,
            self.fft_zero_padding_factor,
        )

    def get_filtered_fft_values(self):
        return list(self.filtered_fft_values)

    def get_bit_values(self):
        return list(self.bit_values)
