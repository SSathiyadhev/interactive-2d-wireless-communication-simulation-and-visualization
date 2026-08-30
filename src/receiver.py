"""
receiver.py

Defines the Receiver class.

Performs coherent BPSK demodulation: band-pass filtering, carrier
mixing, RRC matched filtering, symbol-timing-aligned bit decisions,
and cumulative Bit Error Rate (BER) measurement.

Oracle simplifications (deliberate, documented)
------------------------------------------------
Real BPSK receivers must recover carrier phase and symbol timing
blindly from the received signal alone (carrier recovery loops,
timing recovery loops). This Receiver instead is given a direct
reference to its Transmitter and uses the simulator's own known
geometry to compute:

  1. Local-oscillator phase compensation, from the true
     propagation delay (distance / wave speed).
  2. Symbol-sampling-instant timing, from propagation delay plus
     both filters' (TX pulse-shaping + RX matched-filter) group
     delays.
  3. Ground-truth bits for BER, from the transmitter's own
     recorded transmit history (never regenerated).

This is appropriate for measuring communication-quality metrics
in a simulator where both ends are known to the observer, but it
is not how a real, blind receiver would work. Real carrier/timing
recovery (e.g. a Costas loop) is a documented future upgrade, not
implemented here.
"""

from collections import deque
import math

import numpy as np

from src.filter import Filter


class Receiver:
    """
    Coherent BPSK receiver: band-pass filter, carrier mixing, RRC
    matched filter, symbol-timing-aligned bit decisions, and
    cumulative BER measurement.
    """

    def __init__(
        self,
        simulation_space,
        x,
        y,
        tuned_frequency,
        bit_rate,
        transmitter,
        observation_window=10e-9,
        rrc_rolloff=0.35,
        rrc_span=8,
        bandpass_order=4,
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
            raise ValueError(
                "Bit rate must be greater than zero."
            )

        # Oracle reference -- used ONLY for phase/timing
        # compensation and BER ground truth (see module docstring).
        self.transmitter = transmitter

        # ---------------------------------------------------------
        # Rolling observation buffers (visualization only).
        # ---------------------------------------------------------

        max_samples = max(
            1,
            int(round(
                self.observation_window / self.simulation_space.dt
            )),
        )

        self.time_values = deque(maxlen=max_samples)
        self.received_values = deque(maxlen=max_samples)
        self.filtered_values = deque(maxlen=max_samples)
        self.mixed_values = deque(maxlen=max_samples)
        self.baseband_values = deque(maxlen=max_samples)
        self.demodulated_bits = deque(maxlen=max_samples)

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
        # BER bookkeeping
        # ---------------------------------------------------------

        self._bit_errors = 0
        self._total_bits_compared = 0

        # Direct symbol counter: the receiver's Nth decision
        # corresponds, by construction of the sampling schedule
        # below, to the Nth transmitted symbol (index 0, 1, 2, ...).
        # This avoids converting the decision instant back into a
        # time and re-deriving which bit period that time falls in,
        # which is fragile: that instant lands essentially exactly on
        # a bit-period boundary by design, where sub-sample
        # floating-point rounding noise can flip which side of the
        # boundary a time-based lookup lands on.
        self._next_symbol_index = 0

        # ---------------------------------------------------------
        # Oracle delay/phase estimate (computed once here; call
        # _update_delay_estimate() again if geometry changes).
        # ---------------------------------------------------------

        self._update_delay_estimate()

    # =============================================================
    # ORACLE DELAY / PHASE ESTIMATION
    # =============================================================

    def _update_delay_estimate(self):
        """
        Computes:
          - self._propagation_delay_seconds : true channel delay,
            used ONLY to phase-align the local oscillator.
          - self._total_delay_seconds : propagation delay + both
            filters' group delays, used to know WHEN a given
            symbol's decision-worthy peak arrives at the output of
            this receiver's processing chain.
          - self._total_delay_samples : the same, in samples.
        """

        tx_x, tx_y = self.transmitter.get_position()

        distance = math.hypot(self.x - tx_x, self.y - tx_y)

        # Oracle simplification: assumes a straight-line path at a
        # single wave speed (the speed at the receiver's own grid
        # cell). Correct for homogeneous media; a real obstacle-laden
        # path would need a proper path integral, not implemented.
        wave_speed = self.simulation_space.get_wave_speed(
            self.x,
            self.y,
        )

        self._propagation_delay_seconds = distance / wave_speed

        tx_group_delay_seconds = (
            self.transmitter.get_pulse_shaping_group_delay_seconds()
        )

        rx_bandpass_group_delay_seconds = (
            self._bandpass_filter.get_group_delay_samples(
                frequency=self.tuned_frequency,
            )
            * self.simulation_space.dt
        )

        rx_matched_group_delay_seconds = (
            self._matched_filter.get_group_delay_samples()
            * self.simulation_space.dt
        )

        self._total_delay_seconds = (
            self._propagation_delay_seconds
            + tx_group_delay_seconds
            + rx_bandpass_group_delay_seconds
            + rx_matched_group_delay_seconds
        )

        self._total_delay_samples = int(round(
            self._total_delay_seconds / self.simulation_space.dt
        ))

    # =============================================================
    # PIPELINE STAGES
    # =============================================================

    def _sample_field(self):

        t = self.simulation_space.time

        received_value = self.simulation_space.get_field(
            self.x,
            self.y,
        )

        self.time_values.append(t)
        self.received_values.append(received_value)

        return t, received_value

    def _filter_signal(self, received_value):

        filtered_value = self._bandpass_filter.filter(received_value)

        self.filtered_values.append(filtered_value)

        return filtered_value

    def _mix_signal(self, filtered_value, current_time):
        """
        Down-converts using a local oscillator phase-compensated
        for the TRUE PROPAGATION DELAY ONLY (not filter delays --
        see module docstring for why those are handled separately,
        in symbol-timing rather than here).

        Multiplied by 2.0 to correct the standard real-mixer
        conversion loss: A*cos(wt) * cos(wt) = A/2 + (A/2)*cos(2wt),
        so once the matched filter removes the 2*fc term, what's
        left is A/2, not A. This restores that specific factor --
        it does NOT make the full RX chain unity-gain end-to-end
        (the matched filter's own peak gain still depends on the
        transmitted pulse's autocorrelation, since it's
        energy-normalized rather than peak-normalized). Bit
        decisions threshold on sign only, so this scaling has never
        affected BER -- it matters once RSSI/SNR (magnitude-based)
        are implemented.
        """

        compensated_time = (
            current_time - self._propagation_delay_seconds
        )

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
        mixing, matched filtering, and -- at the correct
        symbol-timing instant only -- a bit decision plus a BER
        update against the transmitter's recorded ground truth.
        """

        current_time, received_value = self._sample_field()

        filtered_value = self._filter_signal(received_value)

        mixed_value = self._mix_signal(filtered_value, current_time)

        baseband_value = self._matched_filter_stage(mixed_value)

        if self._is_symbol_sampling_instant(current_time):
            self._decide_bit(current_time, baseband_value)

    def _is_symbol_sampling_instant(self, current_time):
        """
        Returns True if `current_time` is the correct instant to
        sample a symbol decision -- i.e. `total_delay_samples`
        after a transmitted symbol boundary, spaced one symbol
        period apart.
        """

        step_index = int(round(
            current_time / self.simulation_space.dt
        ))

        if step_index < self._total_delay_samples:
            return False

        offset = step_index - self._total_delay_samples

        return (offset % self._samples_per_symbol) == 0

    def _decide_bit(self, current_time, baseband_value):
        """
        Thresholds one matched-filter sample into a bit (matching
        the transmitter's bit 0 -> +1, bit 1 -> -1 mapping), records
        it, and -- if ground truth is available -- updates the
        cumulative BER.

        Ground truth is looked up by direct symbol-index counting
        (see _next_symbol_index), not by converting back to a time --
        see the comment where _next_symbol_index is created.
        """

        decoded_bit = 0 if baseband_value >= 0.0 else 1

        self.demodulated_bits.append(decoded_bit)

        expected_bit = self.transmitter.get_bit_by_symbol_index(
            self._next_symbol_index
        )

        self._next_symbol_index += 1

        # expected_bit is None if this symbol index has already aged
        # out of the transmitter's recorded window, or hasn't been
        # transmitted yet -- skip counting rather than guessing.
        if expected_bit is not None:

            self._total_bits_compared += 1

            if decoded_bit != expected_bit:
                self._bit_errors += 1

    # =============================================================
    # RESULTS
    # =============================================================

    def get_bit_error_rate(self):
        """
        Returns the cumulative Bit Error Rate, or None if no bits
        have been compared against ground truth yet.
        """

        if self._total_bits_compared == 0:
            return None

        return self._bit_errors / self._total_bits_compared

    def get_bit_errors(self):
        return self._bit_errors

    def get_total_bits_compared(self):
        return self._total_bits_compared

    def get_estimated_total_delay_seconds(self):
        return self._total_delay_seconds

    def get_estimated_propagation_delay_seconds(self):
        return self._propagation_delay_seconds

    # =============================================================
    # CONFIGURATION
    # =============================================================

    def set_position(self, x, y):

        x = float(x)
        y = float(y)

        if not self.simulation_space.is_inside(x, y):
            raise ValueError(
                f"Point ({x}, {y}) is outside "
                f"the simulation space."
            )

        self.x = x
        self.y = y

        self._update_delay_estimate()

    def get_position(self):
        return self.x, self.y

    def set_tuned_frequency(self, value):

        self.tuned_frequency = float(value)

        self._bandpass_filter.set_parameters(
            low_cutoff_frequency=self.tuned_frequency - self.bit_rate,
            high_cutoff_frequency=self.tuned_frequency + self.bit_rate,
        )

        self._update_delay_estimate()

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

        self._update_delay_estimate()

    def get_bit_rate(self):
        return self.bit_rate

    def _compute_samples_per_symbol(self):

        return max(
            1,
            int(round(
                (1.0 / self.bit_rate) / self.simulation_space.dt
            )),
        )

    # =============================================================
    # VISUALIZATION ACCESSORS
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
        return list(self.demodulated_bits)

    def get_observation_times(self):
        return list(self.time_values)