"""
src/receiver.py

Defines the Receiver class using a Costas loop for blind carrier phase tracking.
Demodulated bits are sliced at symbol intervals and exposed to LinkEvaluator.
"""

from collections import deque
import numpy as np

from src.filter import Filter
from src.costas_loop import CostasLoop


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
        self.observation_window = float(observation_window)

        self.fft_window = float(fft_window)

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

        # Front-end RF Band-pass filter
        self.bandpass_filter = Filter(
            "butterworth",
            self.simulation_space.dt,
            order=4,
            filter_response="bandpass",
            low_cutoff_frequency=self.tuned_frequency - self.bit_rate,
            high_cutoff_frequency=self.tuned_frequency + self.bit_rate,
        )

        self.rrc_rolloff = float(rrc_rolloff)
        self.rrc_span = int(rrc_span)

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

        # Costas Loop for carrier phase tracking
        self.costas_loop = CostasLoop(
            dt=self.simulation_space.dt,
            carrier_frequency=self.tuned_frequency,
            bit_rate=self.bit_rate,
            rrc_rolloff=self.rrc_rolloff,
        )

        # Timing alignment state for symbol slicing
        self._propagation_delay_seconds = 0.0
        self._update_internal_filter_delays()

    def _update_internal_filter_delays(self):
        """
        Calculates the total delay from transmitted symbol impulse
        to the corresponding matched-filter symbol peak.

        Includes:
            - propagation delay
            - TX RRC group delay
            - RX RRC group delay
            - receiver BPF group delay
        """

        # ---------------------------------------------------------
        # TX RRC + RX RRC
        # ---------------------------------------------------------
        # Each RRC has span/2 symbol periods of group delay.
        # Therefore:
        #
        #     TX RRC + RX RRC = span symbol periods
        #
        rrc_pipeline_delay = (
            float(self.rrc_span)
            / self.bit_rate
        )

        # ---------------------------------------------------------
        # Receiver RF BPF
        # ---------------------------------------------------------
        bpf_delay_samples = (
            self.bandpass_filter.get_group_delay_samples(
                self.tuned_frequency
            )
        )

        bpf_delay_seconds = (
            bpf_delay_samples
            * self.simulation_space.dt
        )

        # ---------------------------------------------------------
        # Total delay to the matched-filter symbol peak
        # ---------------------------------------------------------
        self._total_delay_seconds = (
            self._propagation_delay_seconds
            + rrc_pipeline_delay
            + bpf_delay_seconds
        )

        self._total_delay_samples = int(
            round(
                self._total_delay_seconds
                / self.simulation_space.dt
            )
        )
        
    def set_estimated_propagation_delay(self, prop_delay_seconds):
        """Allows LinkEvaluator to pass channel delay for symbol-clock alignment."""
        self._propagation_delay_seconds = float(prop_delay_seconds)
        self._update_internal_filter_delays()

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

    def _is_symbol_sampling_instant(self, current_time):
        step_index = int(
            round(
                current_time / self.simulation_space.dt
            )
        )

        if step_index < self._total_delay_samples:
            return False

        offset = (
            step_index
            - self._total_delay_samples
        )

        return (
            offset % self._samples_per_symbol
        ) == 0

    def _decide_bit(self, baseband_value):
        """
        Thresholds matched filter output.
        Flipped to account for 180-degree carrier phase / coordinate sign inversion.
        """
        decoded_bit = 0 if baseband_value >= 0.0 else 1
        self.current_bit = decoded_bit
        self.demodulated_bits.append(decoded_bit)

    def receive(self):
        """Processes one simulation timestep."""
        current_time, received_value = self._sample_field()
        filtered_value = self._filter_signal(received_value)

        # Costas Loop performs carrier downconversion
        mixed_value, _ = self.costas_loop.process(filtered_value, current_time)
        self.mixed_values.append(mixed_value)

        # Matched filter
        baseband_value = self._matched_filter_stage(mixed_value)

        # Slice bits at symbol peaks
        if self._is_symbol_sampling_instant(current_time):
            self._decide_bit(baseband_value)

        # Store the latest decoded bit for every simulation timestep
        self.bit_values.append(self.current_bit)

    def _design_filter(self):
        self.bandpass_filter.set_parameters(
            order=4,
            filter_response="bandpass",
            low_cutoff_frequency=self.tuned_frequency - self.bit_rate,
            high_cutoff_frequency=self.tuned_frequency + self.bit_rate,
        )
        self._update_internal_filter_delays()

    def compute_filtered_fft(self):
        """
        Computes the FFT of the signal after the RF band-pass filter.

        Returns
        -------
        frequencies : np.ndarray
            Non-negative frequency values in Hz.

        amplitudes : np.ndarray
            Single-sided FFT amplitude spectrum.
        """

        samples = np.asarray(
            self.filtered_fft_values,
            dtype=np.float64,
        )

        if len(samples) < 8:
            return np.array([]), np.array([])

        n = len(samples)

        fft_values = np.fft.rfft(samples)

        frequencies = np.fft.rfftfreq(
            n,
            d=self.simulation_space.dt,
        )

        amplitudes = (
            2.0 * np.abs(fft_values) / n
        )

        # DC is not doubled.
        amplitudes[0] /= 2.0

        return frequencies, amplitudes

    def set_position(self, x, y):
        self.x = float(x)
        self.y = float(y)

    def get_position(self):
        return self.x, self.y

    def set_tuned_frequency(self, value):
        self.tuned_frequency = float(value)
        self._design_filter()

    def get_tuned_frequency(self):
        return self.tuned_frequency

    def set_bit_rate(self, value):
        self.bit_rate = float(value)
        self._design_filter()
        self._samples_per_symbol = max(
            1,
            int(round((1.0 / self.bit_rate) / self.simulation_space.dt)),
        )
        self.matched_filter.set_parameters(
            rolloff=self.rrc_rolloff,
            samples_per_symbol=self._samples_per_symbol,
            span=self.rrc_span,
            normalize="energy",
        )
        self._update_internal_filter_delays()

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

    def get_baseband_values(self):
        return list(self.baseband_values)

    def get_observation_times(self):
        return list(self.time_values)

    def get_estimated_total_delay_seconds(self):
        return self._total_delay_seconds

    def get_filtered_fft_values(self):
        return list(self.filtered_fft_values)

    def get_bit_values(self):
        return list(self.bit_values)