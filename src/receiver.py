"""
src/receiver.py

Defines the Receiver class.

Performs coherent BPSK demodulation: band-pass filtering, carrier
mixing, RRC matched filtering, and symbol-timing-aligned bit decisions.
Bit comparison and BER metrics are delegated to LinkEvaluator.
"""

from collections import deque
import numpy as np

from src.filter import Filter


class Receiver:
    """
    Coherent BPSK receiver: band-pass filter, carrier mixing, RRC
    matched filter, and symbol-timing-aligned bit decisions.
    """

    def __init__(
        self,
        simulation_space,
        x,
        y,
        tuned_frequency,
        bit_rate,
        observation_window=10e-9,
        rrc_rolloff=0.35,
        rrc_span=8,
        bandpass_order=4,
        **kwargs,
    ):
        self.simulation_space = simulation_space

        self.x = float(x)
        self.y = float(y)

        if not self.simulation_space.is_inside(self.x, self.y):
            raise ValueError(
                f"Receiver position ({self.x} m, {self.y} m) "
                f"is outside the simulation space "
                f"(0..{self.simulation_space.width} m, "
                f"0..{self.simulation_space.height} m)."
            )

        self.tuned_frequency = float(tuned_frequency)
        self.bit_rate = float(bit_rate)
        self.observation_window = float(observation_window)

        if self.bit_rate <= 0:
            raise ValueError("Bit rate must be greater than zero.")

        # ---------------------------------------------------------
        # Rolling observation buffers (for UI oscilloscope only)
        # ---------------------------------------------------------
        max_samples = max(
            1,
            int(round(self.observation_window / self.simulation_space.dt)),
        )

        self.time_values = deque(maxlen=max_samples)
        self.received_values = deque(maxlen=max_samples)
        self.filtered_values = deque(maxlen=max_samples)
        self.mixed_values = deque(maxlen=max_samples)
        self.baseband_values = deque(maxlen=max_samples)

        # Append-only stream of demodulated bits for LinkEvaluator
        self.demodulated_bits = []

        # ---------------------------------------------------------
        # Band-pass filter (shared Filter class)
        # ---------------------------------------------------------
        self.bandpass_order = int(bandpass_order)

        self._bandpass_filter = Filter(
            filter_type="butterworth",
            dt=self.simulation_space.dt,
            order=self.bandpass_order,
            filter_response="bandpass",
            low_cutoff_frequency=self.tuned_frequency - self.bit_rate,
            high_cutoff_frequency=self.tuned_frequency + self.bit_rate,
        )

        # ---------------------------------------------------------
        # RRC matched filter (shared Filter class, energy-normalized)
        # ---------------------------------------------------------
        self.rrc_rolloff = float(rrc_rolloff)
        self.rrc_span = int(rrc_span)

        self._samples_per_symbol = self._compute_samples_per_symbol()

        self._matched_filter = Filter(
            filter_type="rrc",
            dt=self.simulation_space.dt,
            rolloff=self.rrc_rolloff,
            samples_per_symbol=self._samples_per_symbol,
            span=self.rrc_span,
            normalize="energy",
        )

        # ---------------------------------------------------------
        # Timing state
        # ---------------------------------------------------------
        self._propagation_delay_seconds = 0.0
        self._total_delay_seconds = 0.0
        self._total_delay_samples = 0

        self._update_internal_filter_delays()

    # =============================================================
    # DELAY / TIMING CALCULATION
    # =============================================================

    def _update_internal_filter_delays(self):
        """
        Calculates group delay of the full pipeline:
        Transmitter pulse-shaping delay (span/2 symbols)
        + Bandpass group delay
        + Matched filter group delay (span/2 symbols)
        """
        rx_bandpass_group_delay = (
            self._bandpass_filter.get_group_delay_samples(
                frequency=self.tuned_frequency,
            )
            * self.simulation_space.dt
        )

        rx_matched_group_delay = (
            self._matched_filter.get_group_delay_samples()
            * self.simulation_space.dt
        )

        # TX pulse shaping filter delay = (span / 2) * Ts
        tx_shaping_group_delay = (self.rrc_span / 2.0) * (1.0 / self.bit_rate)

        self._filter_delay_seconds = (
            tx_shaping_group_delay + rx_bandpass_group_delay + rx_matched_group_delay
        )
        self._total_delay_seconds = self._propagation_delay_seconds + self._filter_delay_seconds
        self._total_delay_samples = int(round(self._total_delay_seconds / self.simulation_space.dt))

    def set_estimated_propagation_delay(self, prop_delay_seconds):
        """Hook for LinkEvaluator or runtime to pass physical propagation delay."""
        self._propagation_delay_seconds = float(prop_delay_seconds)
        self._update_internal_filter_delays()

    # =============================================================
    # PIPELINE STAGES
    # =============================================================

    def _sample_field(self):
        t = self.simulation_space.time
        received_value = self.simulation_space.get_field(self.x, self.y)

        self.time_values.append(t)
        self.received_values.append(received_value)

        return t, received_value

    def _filter_signal(self, received_value):
        filtered_value = self._bandpass_filter.filter(received_value)
        self.filtered_values.append(filtered_value)
        return filtered_value

    def _mix_signal(self, filtered_value, current_time):
        compensated_time = current_time - self._propagation_delay_seconds

        local_carrier = np.cos(
            2.0 * np.pi * self.tuned_frequency * compensated_time
        )

        mixed_value = 2.0 * filtered_value * local_carrier
        self.mixed_values.append(mixed_value)
        return mixed_value

    def _matched_filter_stage(self, mixed_value):
        baseband_value = self._matched_filter.filter(mixed_value)
        self.baseband_values.append(baseband_value)
        return baseband_value

    # =============================================================
    # MAIN ENTRY POINT
    # =============================================================

    def receive(self):
        """
        Processes one simulation timestep: sampling, filtering,
        mixing, matched filtering, and bit slicing at symbol intervals.
        """
        current_time, received_value = self._sample_field()
        filtered_value = self._filter_signal(received_value)
        mixed_value = self._mix_signal(filtered_value, current_time)
        baseband_value = self._matched_filter_stage(mixed_value)

        if self._is_symbol_sampling_instant(current_time):
            self._decide_bit(baseband_value)

    def _is_symbol_sampling_instant(self, current_time):
        step_index = int(round(current_time / self.simulation_space.dt))

        if step_index < self._total_delay_samples:
            return False

        offset = step_index - self._total_delay_samples
        return (offset % self._samples_per_symbol) == 0

    def _decide_bit(self, baseband_value):
        """
        Thresholds matched filter output sample (>= 0 -> bit 0, < 0 -> bit 1)
        and logs the recovered bit.
        """
        decoded_bit = 0 if baseband_value >= 0.0 else 1
        self.demodulated_bits.append(decoded_bit)

    # =============================================================
    # CONFIGURATION & REPOSITIONING
    # =============================================================

    def set_position(self, x, y):
        x = float(x)
        y = float(y)

        if not self.simulation_space.is_inside(x, y):
            raise ValueError(f"Point ({x}, {y}) is outside the simulation space.")

        self.x = x
        self.y = y

    def get_position(self):
        return self.x, self.y

    def set_tuned_frequency(self, value):
        self.tuned_frequency = float(value)

        self._bandpass_filter.set_parameters(
            low_cutoff_frequency=self.tuned_frequency - self.bit_rate,
            high_cutoff_frequency=self.tuned_frequency + self.bit_rate,
        )

        self._update_internal_filter_delays()

    def get_tuned_frequency(self):
        return self.tuned_frequency

    def set_bit_rate(self, value):
        self.bit_rate = float(value)

        self._bandpass_filter.set_parameters(
            low_cutoff_frequency=self.tuned_frequency - self.bit_rate,
            high_cutoff_frequency=self.tuned_frequency + self.bit_rate,
        )

        self._samples_per_symbol = self._compute_samples_per_symbol()

        self._matched_filter.set_parameters(
            samples_per_symbol=self._samples_per_symbol,
        )

        self._update_internal_filter_delays()

    def get_bit_rate(self):
        return self.bit_rate

    def _compute_samples_per_symbol(self):
        return max(
            1,
            int(round((1.0 / self.bit_rate) / self.simulation_space.dt)),
        )

    # =============================================================
    # VISUALIZATION & TELEMETRY ACCESSORS
    # =============================================================

    def get_current_received_value(self):
        if not self.received_values:
            return None
        return self.received_values[-1]

    def get_received_values(self):
        return list(self.received_values)

    def get_filtered_values(self):
        return list(self.filtered_values)

    def get_mixed_values(self):
        return list(self.mixed_values)

    def get_baseband_values(self):
        return list(self.baseband_values)

    def get_demodulated_bits(self):
        return self.demodulated_bits

    def get_observation_times(self):
        return list(self.time_values)

    def get_estimated_total_delay_seconds(self):
        return self._total_delay_seconds

    def get_estimated_propagation_delay_seconds(self):
        return self._propagation_delay_seconds