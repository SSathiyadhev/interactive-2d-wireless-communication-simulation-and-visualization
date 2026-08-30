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

        if self.window_duration <= 0:
            raise ValueError(
                "Observation window duration must be greater than zero."
            )

        if self.bit_rate <= 0:
            raise ValueError(
                "Bit rate must be greater than zero."
            )

        # ---------------------------------------------------------
        # Rolling observation buffers (for visualization only --
        # these are NOT used for any filtering computation; the
        # RRC Filter below carries its own internal streaming
        # state independently of these buffers).
        # ---------------------------------------------------------

        max_samples = max(
            1,
            int(round(
                self.window_duration / self.simulation_space.dt
            )),
        )

        self.time_values = deque(maxlen=max_samples)
        self.bit_values = deque(maxlen=max_samples)
        self.carrier_values = deque(maxlen=max_samples)
        self.shaped_values = deque(maxlen=max_samples)
        self.bpsk_values = deque(maxlen=max_samples)

        # ---------------------------------------------------------
        # Bit generation state
        # ---------------------------------------------------------

        self.custom_bit_sequence = (
            list(custom_bit_sequence)
            if custom_bit_sequence is not None
            else None
        )

        self.current_bit = None
        self.last_bit_index = -1
        self._last_symbol_index = -1

        # ---------------------------------------------------------
        # Per-symbol ground-truth history (independent of the
        # per-sample visualization deques above). Sized by number
        # of SYMBOLS that fit in the window, not simulation steps --
        # far smaller, since one symbol spans many steps.
        # ---------------------------------------------------------

        max_symbol_history = max(
            1,
            int(round(
                self.window_duration * self.bit_rate
            )),
        )

        self._symbol_history = deque(maxlen=max_symbol_history)
        self._symbol_history_start_index = 0

        # ---------------------------------------------------------
        # RRC pulse-shaping filter (shared Filter class)
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

        A random bit is generated only when the simulation enters
        a new bit period. The same random bit is maintained
        throughout its complete bit duration.
        """

        if self.bit_rate <= 0:
            raise ValueError(
                "Bit rate must be greater than zero."
            )

        bit_period = 1.0 / self.bit_rate

        current_bit_index = int(current_time // bit_period)

        if (
            current_bit_index != self.last_bit_index
            or self.current_bit is None
        ):

            self.last_bit_index = current_bit_index

            if (
                self.custom_bit_sequence is not None
                and len(self.custom_bit_sequence) > 0
            ):
                sequence_index = (
                    current_bit_index
                    % len(self.custom_bit_sequence)
                )

                self.current_bit = int(
                    self.custom_bit_sequence[sequence_index]
                )

            else:
                self.current_bit = random.choice([0, 1])

            self._record_symbol_history(self.current_bit)

        return self.current_bit

    def _get_symbol_impulse(self, current_time):
        """
        Returns +1 or -1 once at the start of each symbol.
        Returns 0 for the remaining samples in that symbol.
        """

        bit_period = 1.0 / self.bit_rate

        current_symbol_index = int(current_time // bit_period)

        if current_symbol_index != self._last_symbol_index:

            self._last_symbol_index = current_symbol_index

            bit = self.get_bit_at_time(current_time)

            return 1.0 if bit == 0 else -1.0

        return 0.0

    def _shape_symbol(self, current_time):
        """
        Generates the symbol impulse and applies RRC pulse shaping
        via the shared streaming Filter.
        """

        symbol_impulse = self._get_symbol_impulse(current_time)

        return self._pulse_filter.filter(symbol_impulse)

    # =============================================================
    # CARRIER GENERATION
    # =============================================================

    def get_current_carrier_value(self, current_time):
        """
        carrier(t) = Ac * cos(2*pi*fc*t)
        """

        return self.Ac * np.cos(2.0 * np.pi * self.fc * current_time)

    # =============================================================
    # BPSK SIGNAL GENERATION
    # =============================================================

    def get_current_transmitted_value(self, current_time):
        """
        Calculates the current BPSK signal and records exactly
        one synchronized observation sample (for visualization).
        """

        bit = self.get_bit_at_time(current_time)

        carrier = self.get_current_carrier_value(current_time)

        shaped_value = self._shape_symbol(current_time)

        bpsk = shaped_value * carrier

        self.time_values.append(current_time)
        self.bit_values.append(bit)
        self.carrier_values.append(carrier)
        self.shaped_values.append(shaped_value)
        self.bpsk_values.append(bpsk)

        return bpsk

    # =============================================================
    # TRANSMISSION
    # =============================================================

    def transmit(self, current_time=None):
        """
        Generates the current BPSK value and injects it into
        SimulationSpace using SOFT-SOURCE injection: the
        transmitter's contribution is ADDED to whatever field
        value already exists at its location, rather than
        overwriting it.

        This preserves the effect of neighboring cells already
        computed by the WaveSolver (from other transmitters, or
        from waves reflecting back toward this location), instead
        of clamping this grid point to a fixed value every step
        (which behaves like an unintended Dirichlet boundary and
        causes non-physical reflections once multiple transmitters
        or obstacles are present).
        """

        if current_time is None:
            current_time = self.simulation_space.time

        voltage = self.get_current_transmitted_value(current_time)

        existing_field_value = self.simulation_space.get_field(
            self.x,
            self.y,
        )

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
            raise ValueError(
                "Custom bit sequence cannot be empty."
            )

        for bit in bit_sequence:
            if bit not in (0, 1):
                raise ValueError(
                    "Custom bit sequence must contain only 0 and 1."
                )

        self.custom_bit_sequence = list(bit_sequence)

        self._reset_bit_state()

    def clear_custom_bit_sequence(self):

        self.custom_bit_sequence = None

        self._reset_bit_state()

    def _reset_bit_state(self):

        self.current_bit = None
        self.last_bit_index = -1
        self._last_symbol_index = -1

        self._symbol_history.clear()
        self._symbol_history_start_index = 0

        self._pulse_filter.reset()

    def _record_symbol_history(self, bit):
        """
        Appends `bit` to the per-symbol ground-truth history,
        keeping `_symbol_history_start_index` in sync with
        whatever the oldest remaining entry's symbol index is
        (the deque silently discards the oldest entry once full).
        """

        if len(self._symbol_history) == self._symbol_history.maxlen:
            self._symbol_history_start_index += 1

        self._symbol_history.append(bit)

    # =============================================================
    # POSITION
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

    def get_position(self):

        return self.x, self.y

    # =============================================================
    # CARRIER FREQUENCY
    # =============================================================

    def set_carrier_frequency(self, fc):

        self.fc = float(fc)

    def get_carrier_frequency(self):

        return self.fc

    # =============================================================
    # CARRIER AMPLITUDE
    # =============================================================

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
            raise ValueError(
                "Bit rate must be greater than zero."
            )

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
            int(round(
                (1.0 / self.bit_rate) / self.simulation_space.dt
            )),
        )

    # =============================================================
    # OBSERVATION WINDOW
    # =============================================================

    def set_window_duration(self, window_duration):

        window_duration = float(window_duration)

        if window_duration <= 0:
            raise ValueError(
                "Observation window duration must be greater than zero."
            )

        self.window_duration = window_duration

        max_samples = max(
            1,
            int(round(
                self.window_duration / self.simulation_space.dt
            )),
        )

        self.time_values = deque(maxlen=max_samples)
        self.bit_values = deque(maxlen=max_samples)
        self.carrier_values = deque(maxlen=max_samples)
        self.shaped_values = deque(maxlen=max_samples)
        self.bpsk_values = deque(maxlen=max_samples)

    def get_window_duration(self):

        return self.window_duration

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

    # =============================================================
    # RECEIVER-FACING HELPERS (oracle-style lookups)
    # =============================================================

    def get_pulse_shaping_group_delay_seconds(self):
        """
        Returns the group delay introduced by this transmitter's
        RRC pulse-shaping filter, in seconds.
        """

        return (
            self._pulse_filter.get_group_delay_samples()
            * self.simulation_space.dt
        )

    def get_bit_at_recorded_time(self, query_time):
        """
        Returns the bit that was ACTUALLY transmitted at
        `query_time`, read from recorded history.

        This is intentionally different from get_bit_at_time():
        that method is stateful and advances the bit-generation
        sequence whenever it sees a new bit-period index -- calling
        it with an arbitrary past `query_time` would incorrectly
        generate a brand-new random bit instead of reporting what
        was really sent, corrupting the transmitter's own sequence.

        This method never mutates state -- it only looks up
        `query_time` in the rolling observation history. Returns
        None if `query_time` falls outside the currently recorded
        window (either because it's too far in the past and has
        aged out, or hasn't happened yet).

        NOTE: prefer get_bit_by_symbol_index() over this method for
        BER ground truth. A receiver's correct decision instant for
        symbol k lands (by construction) essentially exactly on the
        boundary between bit period k and k+1, so converting that
        instant back into a time and re-deriving "which bit period
        is this" is fragile to sub-sample floating-point rounding
        noise -- it can flip to the wrong side of the boundary. Direct
        symbol-index counting (see get_bit_by_symbol_index) sidesteps
        this entirely. This method is kept for cases where a time
        value -- not a symbol index -- is genuinely what's available.
        """

        if not self.time_values:
            return None

        times = np.fromiter(self.time_values, dtype=float)

        if query_time < times[0] or query_time > times[-1]:
            return None

        index = int(np.searchsorted(times, query_time))
        index = min(index, len(times) - 1)

        return self.bit_values[index]

    def get_bit_by_symbol_index(self, symbol_index):
        """
        Returns the bit actually transmitted during bit period
        `symbol_index` (0-based, counting bit periods from t=0),
        read from a dedicated per-symbol history that is immune to
        the floating-point boundary ambiguity described above.

        Returns None if `symbol_index` is not in the recorded
        window (too far in the past and aged out, or hasn't
        occurred yet).
        """

        if not self._symbol_history:
            return None

        offset = symbol_index - self._symbol_history_start_index

        if offset < 0 or offset >= len(self._symbol_history):
            return None

        return self._symbol_history[offset]