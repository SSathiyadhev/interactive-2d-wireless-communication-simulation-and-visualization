from collections import deque
import random
import numpy as np


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

        # Observation window duration in seconds
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
        # Calculate maximum number of samples in observation window
        # ---------------------------------------------------------

        max_samples = max(
            1,
            int(
                round(
                    self.window_duration
                    / self.simulation_space.dt
                )
            ),
        )

        # ---------------------------------------------------------
        # Rolling observation buffers
        #
        # All four deques have the same maxlen so that the samples
        # remain synchronized.
        # ---------------------------------------------------------

        self.time_values = deque(maxlen=max_samples)
        self.bit_values = deque(maxlen=max_samples)
        self.carrier_values = deque(maxlen=max_samples)
        self.bpsk_values = deque(maxlen=max_samples)

        # ---------------------------------------------------------
        # Bit generation state
        # ---------------------------------------------------------

        self.custom_bit_sequence = (
            list(custom_bit_sequence)
            if custom_bit_sequence is not None
            else None
        )

        # Current random/custom bit
        self.current_bit = None

        # Bit period index of current_bit
        self.last_bit_index = -1

    # =============================================================
    # BIT GENERATION
    # =============================================================

    def get_bit_at_time(self, current_time):
        """
        Returns the bit active at the specified simulation time.

        A random bit is generated only when the simulation enters
        a new bit period.

        Therefore, the same random bit is maintained throughout
        its complete bit duration.
        """

        if self.bit_rate <= 0:
            raise ValueError(
                "Bit rate must be greater than zero."
            )

        # Duration of one bit
        bit_period = 1.0 / self.bit_rate

        # Determine which bit period the current time belongs to
        current_bit_index = int(
            current_time // bit_period
        )

        # Generate/select a new bit only when entering
        # a new bit period.
        if (
            current_bit_index != self.last_bit_index
            or self.current_bit is None
        ):

            self.last_bit_index = current_bit_index

            # -----------------------------------------------------
            # Custom bit sequence
            # -----------------------------------------------------

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

            # -----------------------------------------------------
            # Random bit
            # -----------------------------------------------------

            else:
                self.current_bit = random.choice([0, 1])

        return self.current_bit

    # =============================================================
    # CARRIER GENERATION
    # =============================================================

    def get_current_carrier_value(self, current_time):
        """
        Calculates the carrier value at the specified time.

            carrier(t) = Ac * cos(2*pi*fc*t)
        """

        return (
            self.Ac
            * np.cos(
                2.0
                * np.pi
                * self.fc
                * current_time
            )
        )

    # =============================================================
    # BPSK SIGNAL GENERATION
    # =============================================================

    def get_current_transmitted_value(self, current_time):
        """
        Calculates the current BPSK signal and records exactly
        one synchronized observation sample.

        One call produces:

            time[i]
            bit[i]
            carrier[i]
            bpsk[i]
        """

        # ---------------------------------------------------------
        # Current bit
        # ---------------------------------------------------------

        bit = self.get_bit_at_time(current_time)

        # ---------------------------------------------------------
        # Current carrier
        # ---------------------------------------------------------

        carrier = self.get_current_carrier_value(
            current_time
        )

        # ---------------------------------------------------------
        # BPSK mapping
        #
        # Bit 0 -> +carrier
        # Bit 1 -> -carrier
        # ---------------------------------------------------------

        polarity = 1.0 if bit == 0 else -1.0

        bpsk = polarity * carrier

        # ---------------------------------------------------------
        # Store ONE synchronized sample
        # ---------------------------------------------------------

        self.time_values.append(current_time)
        self.bit_values.append(bit)
        self.carrier_values.append(carrier)
        self.bpsk_values.append(bpsk)

        # ---------------------------------------------------------
        # deque automatically removes the oldest samples when
        # maxlen is exceeded.
        # ---------------------------------------------------------

        return bpsk

    # =============================================================
    # TRANSMISSION
    # =============================================================

    def transmit(self, current_time=None):
        """
        Generates the current BPSK value and injects it into
        SimulationSpace.

        The observation sample is already recorded by
        get_current_transmitted_value(), so this function does
        not record another sample.
        """

        if current_time is None:
            current_time = self.simulation_space.time

        voltage = self.get_current_transmitted_value(
            current_time
        )

        self.simulation_space.set_field(
            self.x,
            self.y,
            voltage
        )

        return voltage

    # =============================================================
    # CUSTOM BIT SEQUENCE
    # =============================================================

    def set_custom_bit_sequence(self, bit_sequence):
        """
        Sets a custom bit sequence.

        Example:
            [1, 0, 1, 1, 0]
        """

        if bit_sequence is None:
            self.clear_custom_bit_sequence()
            return

        if len(bit_sequence) == 0:
            raise ValueError(
                "Custom bit sequence cannot be empty."
            )

        # Validate that bits are only 0 or 1
        for bit in bit_sequence:
            if bit not in (0, 1):
                raise ValueError(
                    "Custom bit sequence must contain only 0 and 1."
                )

        self.custom_bit_sequence = list(bit_sequence)

        # Reset current-bit state
        self.current_bit = None
        self.last_bit_index = -1

    def clear_custom_bit_sequence(self):
        """
        Removes the custom sequence and switches back to
        random bit generation.
        """

        self.custom_bit_sequence = None

        # Reset current-bit state
        self.current_bit = None
        self.last_bit_index = -1

    # =============================================================
    # POSITION
    # =============================================================

    def set_position(self, x, y):
        """
        Changes transmitter position.
        """

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
        """
        Returns transmitter position.
        """

        return self.x, self.y

    # =============================================================
    # CARRIER FREQUENCY
    # =============================================================

    def set_carrier_frequency(self, fc):
        """
        Sets carrier frequency.
        """

        self.fc = float(fc)

    def get_carrier_frequency(self):
        """
        Returns carrier frequency.
        """

        return self.fc

    # =============================================================
    # CARRIER AMPLITUDE
    # =============================================================

    def set_carrier_amplitude(self, Ac):
        """
        Sets carrier amplitude.
        """

        self.Ac = float(Ac)

    def get_carrier_amplitude(self):
        """
        Returns carrier amplitude.
        """

        return self.Ac

    # =============================================================
    # BIT RATE
    # =============================================================

    def set_bit_rate(self, bit_rate):
        """
        Sets bit rate.
        """

        bit_rate = float(bit_rate)

        if bit_rate <= 0:
            raise ValueError(
                "Bit rate must be greater than zero."
            )

        self.bit_rate = bit_rate

        # Reset current-bit state because the bit periods
        # have changed.
        self.current_bit = None
        self.last_bit_index = -1

    def get_bit_rate(self):
        """
        Returns bit rate.
        """

        return self.bit_rate

    # =============================================================
    # OBSERVATION WINDOW
    # =============================================================

    def set_window_duration(self, window_duration):
        """
        Changes the duration of the rolling observation window.

        The existing observation data is cleared because the
        buffer size changes.
        """

        window_duration = float(window_duration)

        if window_duration <= 0:
            raise ValueError(
                "Observation window duration must be greater than zero."
            )

        self.window_duration = window_duration

        max_samples = max(
            1,
            int(
                round(
                    self.window_duration
                    / self.simulation_space.dt
                )
            ),
        )

        # Recreate rolling buffers with the new size
        self.time_values = deque(maxlen=max_samples)
        self.bit_values = deque(maxlen=max_samples)
        self.carrier_values = deque(maxlen=max_samples)
        self.bpsk_values = deque(maxlen=max_samples)

    def get_window_duration(self):
        """
        Returns observation window duration.
        """

        return self.window_duration