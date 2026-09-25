"""
transmitter.py

Defines the Transmitter class.

Generates BPSK modulated signals (via RRC pulse shaping, delegated
to the shared Filter class) and injects them into SimulationSpace
using soft-source injection.
"""

from collections import deque
import random
import numpy as np

from src.filter import Filter


class Transmitter:
    """
    Generates BPSK modulated signals and injects them into SimulationSpace.
    """

    def __init__(
        self,
        simulation_space,
        x,
        y,
        carrier_frequency,
        carrier_amplitude,
        bit_rate,
        window_duration=10e-6,
        fft_window=30e-9,
        custom_bit_sequence=None,
        rrc_rolloff=0.35,
        rrc_span=8,
        **kwargs,
    ):
        self.simulation_space = simulation_space

        # ---------------------------------------------------------
        # Position
        # ---------------------------------------------------------
        self.x = float(x)
        self.y = float(y)

        if not self.simulation_space.is_inside(self.x, self.y):
            raise ValueError(
                f"Transmitter position ({self.x} m, {self.y} m) "
                f"is outside the simulation space "
                f"(0..{self.simulation_space.width} m, "
                f"0..{self.simulation_space.height} m)."
            )

        # ---------------------------------------------------------
        # Transmission parameters
        # ---------------------------------------------------------
        self.fc = float(carrier_frequency)
        self.Ac = float(carrier_amplitude)
        self.bit_rate = float(bit_rate)
        self.window_duration = float(window_duration)

        self.fft_window = float(fft_window)

        if self.fft_window <= 0:
            raise ValueError("FFT window must be strictly positive.")

        if self.window_duration <= 0:
            raise ValueError("Observation window duration must be greater than zero.")

        if self.bit_rate <= 0:
            raise ValueError("Bit rate must be greater than zero.")

        # ---------------------------------------------------------
        # Rolling observation buffers (for UI oscilloscope only)
        # ---------------------------------------------------------
        max_samples = max(
            1,
            int(round(self.window_duration / self.simulation_space.dt)),
        )

        self.time_values = deque(maxlen=max_samples)
        self.bit_values = deque(maxlen=max_samples)
        self.carrier_values = deque(maxlen=max_samples)
        self.shaped_values = deque(maxlen=max_samples)
        self.bpsk_values = deque(maxlen=max_samples)

        fft_samples = max(
            8,
            int(np.ceil(
                self.fft_window / self.simulation_space.dt
            )),
        )

        self.bpsk_fft_values = deque(maxlen=fft_samples)

        # ---------------------------------------------------------
        # Bit generation state & Ground-Truth Reference
        # ---------------------------------------------------------
        self.custom_bit_sequence = (
            list(custom_bit_sequence)
            if custom_bit_sequence is not None
            else None
        )

        self.current_bit = None
        self.last_bit_index = -1
        self._last_symbol_index = -1

        # Append-only ground-truth log for LinkEvaluator
        self.generated_bits = []

        # ---------------------------------------------------------
        # RRC pulse-shaping filter
        # ---------------------------------------------------------
        self.rrc_rolloff = float(rrc_rolloff)
        self.rrc_span = int(rrc_span)
        self._samples_per_symbol = self._compute_samples_per_symbol()

        self._pulse_filter = Filter(
            filter_type="rrc",
            dt=self.simulation_space.dt,
            rolloff=self.rrc_rolloff,
            samples_per_symbol=self._samples_per_symbol,
            span=self.rrc_span,
            normalize="peak",
        )

    # =============================================================
    # BIT GENERATION
    # =============================================================

    def get_bit_at_time(self, current_time):
        """
        Returns the bit active at the specified simulation time.
        A random bit is generated only when entering a new bit period.
        """
        if self.bit_rate <= 0:
            raise ValueError("Bit rate must be greater than zero.")

        current_sample = int(
            round(current_time / self.simulation_space.dt)
        )
        current_bit_index = current_sample // self._samples_per_symbol

        if current_bit_index != self.last_bit_index or self.current_bit is None:
            self.last_bit_index = current_bit_index

            if self.custom_bit_sequence is not None and len(self.custom_bit_sequence) > 0:
                sequence_index = current_bit_index % len(self.custom_bit_sequence)
                self.current_bit = int(self.custom_bit_sequence[sequence_index])
            else:
                self.current_bit = random.choice([0, 1])

            self.generated_bits.append(self.current_bit)

        return self.current_bit

    def _get_symbol_impulse(self, current_time):
        current_sample = int(
            round(current_time / self.simulation_space.dt)
        )
        current_symbol_index = current_sample // self._samples_per_symbol

        if current_symbol_index != self._last_symbol_index:
            self._last_symbol_index = current_symbol_index
            bit = self.get_bit_at_time(current_time)
            return 1.0 if bit == 0 else -1.0

        return 0.0

    def _shape_symbol(self, current_time):
        symbol_impulse = self._get_symbol_impulse(current_time)
        return self._pulse_filter.filter(symbol_impulse)

    # =============================================================
    # CARRIER GENERATION
    # =============================================================

    def get_current_carrier_value(self, current_time):
        return self.Ac * np.cos(2.0 * np.pi * self.fc * current_time)

    # =============================================================
    # BPSK SIGNAL GENERATION
    # =============================================================

    def get_current_transmitted_value(self, current_time):
        bit = self.get_bit_at_time(current_time)
        carrier = self.get_current_carrier_value(current_time)
        shaped_value = self._shape_symbol(current_time)
        bpsk = shaped_value * carrier

        self.time_values.append(current_time)
        self.bit_values.append(bit)
        self.carrier_values.append(carrier)
        self.shaped_values.append(shaped_value)
        self.bpsk_values.append(bpsk)

        self.bpsk_fft_values.append(bpsk)

        return bpsk

    # =============================================================
    # TRANSMISSION
    # =============================================================

    def transmit(self, current_time=None):
        """Soft-source injection into SimulationSpace."""
        if current_time is None:
            current_time = self.simulation_space.time

        voltage = self.get_current_transmitted_value(current_time)
        existing_field_value = self.simulation_space.get_field(self.x, self.y)

        self.simulation_space.set_field(
            self.x,
            self.y,
            existing_field_value + voltage,
        )

        return voltage

    # =============================================================
    # CUSTOM BIT SEQUENCE
    # =============================================================

    def set_custom_bit_sequence(self, bit_sequence):
        if bit_sequence is None:
            self.clear_custom_bit_sequence()
            return

        if len(bit_sequence) == 0:
            raise ValueError("Custom bit sequence cannot be empty.")

        for bit in bit_sequence:
            if bit not in (0, 1):
                raise ValueError("Custom bit sequence must contain only 0 and 1.")

        self.custom_bit_sequence = list(bit_sequence)
        self._reset_bit_state()

    def clear_custom_bit_sequence(self):
        self.custom_bit_sequence = None
        self._reset_bit_state()

    def _reset_bit_state(self):
        self.current_bit = None
        self.last_bit_index = -1
        self._last_symbol_index = -1
        self.generated_bits.clear()
        self._pulse_filter.reset()

    # =============================================================
    # POSITION
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

    # =============================================================
    # CARRIER FREQUENCY & AMPLITUDE
    # =============================================================

    def set_carrier_frequency(self, fc):
        self.fc = float(fc)

    def get_carrier_frequency(self):
        return self.fc

    def set_carrier_amplitude(self, Ac):
        self.Ac = float(Ac)

    def get_carrier_amplitude(self):
        return self.Ac

    # =============================================================
    # BIT RATE
    # =============================================================

    def set_bit_rate(self, bit_rate):
        bit_rate = float(bit_rate)

        if bit_rate <= 0:
            raise ValueError("Bit rate must be greater than zero.")

        self.bit_rate = bit_rate
        self._reset_bit_state()
        self._samples_per_symbol = self._compute_samples_per_symbol()
        self._pulse_filter.set_parameters(
            samples_per_symbol=self._samples_per_symbol,
        )

    def get_bit_rate(self):
        return self.bit_rate

    def _compute_samples_per_symbol(self):
        return max(
            1,
            int(round((1.0 / self.bit_rate) / self.simulation_space.dt)),
        )

    # =============================================================
    # OBSERVATION WINDOW
    # =============================================================

    def set_window_duration(self, window_duration):
        window_duration = float(window_duration)

        if window_duration <= 0:
            raise ValueError("Observation window duration must be greater than zero.")

        self.window_duration = window_duration

        max_samples = max(
            1,
            int(round(self.window_duration / self.simulation_space.dt)),
        )

        self.time_values = deque(maxlen=max_samples)
        self.bit_values = deque(maxlen=max_samples)
        self.carrier_values = deque(maxlen=max_samples)
        self.shaped_values = deque(maxlen=max_samples)
        self.bpsk_values = deque(maxlen=max_samples)

    def get_window_duration(self):
        return self.window_duration


    def compute_bpsk_fft(self):
        """
        Computes the FFT of the transmitted RRC-shaped BPSK RF waveform.
        """

        samples = np.asarray(
            self.bpsk_fft_values,
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

        amplitudes[0] /= 2.0

        return frequencies, amplitudes

    # =============================================================
    # BUFFER ACCESSORS (for UI Oscilloscope & LinkEvaluator)
    # =============================================================

    def get_time_values(self):
        return list(self.time_values)

    def get_bit_values(self):
        return list(self.bit_values)

    def get_carrier_values(self):
        return list(self.carrier_values)

    def get_shaped_values(self):
        return list(self.shaped_values)

    def get_bpsk_values(self):
        return list(self.bpsk_values)

    def get_generated_bits(self):
        """Returns the full ground-truth bit list for LinkEvaluator."""
        return self.generated_bits

    def get_bpsk_fft_values(self):
        return list(self.bpsk_fft_values)
