from collections import deque
import numpy as np

from scipy.signal import butter, sosfilt, lfilter


class Receiver:

    def __init__(
        self,
        simulation_space,     # SimulationSpace used by the receiver
        x,                    # Receiver physical x-coordinate
        y,                    # Receiver physical y-coordinate
        tuned_frequency,      # Carrier frequency to receive
        bit_rate,             # Transmitted bit rate in bits/second
        observation_window,   # Duration of stored observation data
    ):
        self.simulation_space = simulation_space
        self.x = float(x)
        self.y = float(y)
        self.tuned_frequency = float(tuned_frequency)
        self.bit_rate = float(bit_rate)
        self.observation_window = float(observation_window)
    
        # Number of simulation samples that fit in the observation window
        max_samples = int(
            np.ceil(
                self.observation_window / self.simulation_space.dt
            )
        )

        # Rolling sample-rate data
        self.time_values = deque(maxlen=max_samples)
        self.received_values = deque(maxlen=max_samples)
        self.filtered_values = deque(maxlen=max_samples)
        self.mixed_values = deque(maxlen=max_samples)
        self.demodulated_bits = deque(maxlen=max_samples)
        self.baseband_values = deque(maxlen=max_samples)

        # Digital sampling frequency
        self._sample_frequency = 1.0 / self.simulation_space.dt

        # BPSK band-pass limits
        f_low = self.tuned_frequency - self.bit_rate
        f_high = self.tuned_frequency + self.bit_rate

        # 4th-order Butterworth band-pass filter
        self.filter = butter(
            4,
            [f_low, f_high],
            btype="bandpass",
            fs=self._sample_frequency,
            output="sos",
        )

        # Internal IIR filter state
        self._filter_state = np.zeros(
            (self.filter.shape[0], 2)
        )

        # RRC matched-filter parameters
        self.rrc_rolloff = 0.35
        self.rrc_span = 8

        self._samples_per_symbol = max(
            1,
            int(round(
                (1.0 / self.bit_rate)
                / self.simulation_space.dt
            ))
        )

        self._design_rrc_filter()

        self._rrc_state = np.zeros(
            len(self._rrc_coefficients) - 1
        )

    def _sample_field(self):
        """
        Reads and stores the current field value at the
        receiver position.
        """

        t = self.simulation_space.time

        received_value = self.simulation_space.get_field(
            self.x,
            self.y,
        )

        self.time_values.append(t)
        self.received_values.append(received_value)

        return received_value

    def _filter_signal(self, received_value):
        """
        Filters the current received sample and stores the result.
        """

        filtered_value, self._filter_state = sosfilt(
            self.filter,
            [received_value],
            zi=self._filter_state,
        )

        filtered_value = float(filtered_value[0])

        self.filtered_values.append(filtered_value)

        return filtered_value

    def _mix_signal(self, filtered_value):
        """
        Mixes the filtered signal with the local carrier
        and stores the mixed sample.
        """

        t = self.simulation_space.time

        local_carrier = np.cos(
            2.0 * np.pi * self.tuned_frequency * t
        )

        mixed_value = filtered_value * local_carrier

        self.mixed_values.append(mixed_value)

        return mixed_value


    def _design_rrc_filter(self):
        """
        Generates the RRC matched-filter coefficients.
        """

        alpha = self.rrc_rolloff
        sps = self._samples_per_symbol
        span = self.rrc_span

        number_of_taps = span * sps + 1

        time_values = (
            np.arange(number_of_taps)
            - number_of_taps // 2
        ) / sps

        h = np.zeros_like(time_values, dtype=float)

        for i, t in enumerate(time_values):

            if np.isclose(t, 0.0):

                h[i] = (
                    1.0
                    + alpha
                    * (4.0 / np.pi - 1.0)
                )

            elif alpha != 0 and np.isclose(
                abs(t),
                1.0 / (4.0 * alpha),
            ):

                h[i] = (
                    alpha
                    / np.sqrt(2.0)
                ) * (
                    (1.0 + 2.0 / np.pi)
                    * np.sin(np.pi / (4.0 * alpha))
                    +
                    (1.0 - 2.0 / np.pi)
                    * np.cos(np.pi / (4.0 * alpha))
                )

            else:

                numerator = (
                    np.sin(
                        np.pi * t * (1.0 - alpha)
                    )
                    +
                    4.0
                    * alpha
                    * t
                    * np.cos(
                        np.pi * t * (1.0 + alpha)
                    )
                )

                denominator = (
                    np.pi
                    * t
                    * (
                        1.0
                        - (4.0 * alpha * t) ** 2
                    )
                )

                h[i] = numerator / denominator

        h /= np.sqrt(np.sum(h ** 2))

        self._rrc_coefficients = h

    def _matched_filter(self, mixed_value):
        """
        Applies the RRC matched filter to the mixed signal.
        """

        baseband_value, self._rrc_state = lfilter(
            self._rrc_coefficients,
            1.0,
            [mixed_value],
            zi=self._rrc_state,
        )

        baseband_value = float(baseband_value[0])

        self.baseband_values.append(baseband_value)

        return baseband_value

    def receive(self):
        """
        Processes one simulation timestep.
        """

        # 1. Get the current field sample.
        received_value = self._sample_field()

        # 2. Band-pass filter around the tuned frequency.
        filtered_value = self._filter_signal(
            received_value
        )

        # 3. Mix with the local carrier.
        mixed_value = self._mix_signal(
            filtered_value
        )
        # 4. Apply the RRC matched filter.
        baseband_value = self._matched_filter(
            mixed_value
        )

    def _design_filter(self):
        """
        Recalculates the band-pass filter.
        """

        f_low = (
            self.tuned_frequency
            - self.bit_rate
        )

        f_high = (
            self.tuned_frequency
            + self.bit_rate
        )

        self.filter = butter(
            4,
            [f_low, f_high],
            btype="bandpass",
            fs=self._sample_frequency,
            output="sos",
        )

        self._filter_state = np.zeros(
            (self.filter.shape[0], 2)
        )

    def set_position(self, x, y):
        """
        Updates the receiver position.
        """

        self.x = float(x)
        self.y = float(y)

    def get_position(self):
        """
        Returns the current receiver position.
        """

        return self.x, self.y

    def set_tuned_frequency(self, value):
        """
        Updates the tuned frequency and recalculates
        the band-pass filter.
        """

        self.tuned_frequency = float(value)

        self._design_filter()

    def get_tuned_frequency(self):
        """
        Returns the current receiver tuned frequency.
        """

        return self.tuned_frequency

    def set_bit_rate(self, value):
        """
        Updates the bit rate and recalculates
        the band-pass filter and bit timing.
        """

        self.bit_rate = float(value)

        self._design_filter()
        self._samples_per_symbol = max(
            1,
            int(round(
                (1.0 / self.bit_rate)
                / self.simulation_space.dt
            ))
        )

        self._design_rrc_filter()

        self._rrc_state = np.zeros(
            len(self._rrc_coefficients) - 1
        )

    def get_bit_rate(self):
        """
        Returns the current bit rate.
        """

        return self.bit_rate

    def get_current_received_value(self):
        """
        Returns the most recent received field value.
        """

        if not self.received_values:
            return None

        return self.received_values[-1]

    def get_received_values(self):
        """
        Returns the rolling received signal values.
        """

        return list(self.received_values)

    def get_filtered_values(self):
        """
        Returns the rolling filtered signal values.
        """

        return list(self.filtered_values)

    def get_mixed_values(self):
        """
        Returns the rolling mixed signal values.
        """

        return list(self.mixed_values)

    def get_demodulated_bits(self):
        """
        Returns the rolling demodulated bit values.
        """

        return list(self.demodulated_bits)

    def get_baseband_values(self):
        """
        Returns the rolling baseband signal values.
        """

        return list(self.baseband_values)

    def get_observation_times(self):
        """
        Returns the rolling observation time values.
        """

        return list(self.time_values)

