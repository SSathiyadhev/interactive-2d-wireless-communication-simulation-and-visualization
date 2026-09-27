"""
gardner_loop.py

Streaming Gardner symbol-timing recovery for an oversampled,
RRC-matched baseband signal.

The timing loop operates once per recovered symbol.

Processing chain:

    RRC matched baseband
            |
            v
    fractional interpolation
            |
            v
       Gardner TED
            |
            v
       PI timing loop
            |
            v
       timing NCO
            |
            v
    recovered symbol
"""

import numpy as np


class GardnerLoop:

    def __init__(
        self,
        dt,
        symbol_rate,
        loop_bandwidth_ratio=0.05,
        damping_factor=0.707,
        initial_timing=None,
        buffer_size=1024,
    ):
        self.dt = float(dt)
        self.symbol_rate = float(symbol_rate)

        self.loop_bandwidth_ratio = float(
            loop_bandwidth_ratio
        )

        self.damping_factor = float(
            damping_factor
        )

        if self.dt <= 0:
            raise ValueError(
                "dt must be greater than zero."
            )

        if self.symbol_rate <= 0:
            raise ValueError(
                "symbol_rate must be greater than zero."
            )

        if self.loop_bandwidth_ratio <= 0:
            raise ValueError(
                "loop_bandwidth_ratio must be greater "
                "than zero."
            )

        if self.damping_factor <= 0:
            raise ValueError(
                "damping_factor must be greater "
                "than zero."
            )

        # =========================================================
        # TIMING PARAMETERS
        # =========================================================

        # Symbol period.
        self.symbol_period = (
            1.0 / self.symbol_rate
        )

        # Nominal number of input samples per symbol.
        #
        # This does not have to be an integer.
        self.nominal_samples_per_symbol = (
            self.symbol_period / self.dt
        )

        # Current timing-rate estimate.
        self.samples_per_symbol = (
            self.nominal_samples_per_symbol
        )

        # =========================================================
        # TIMING LOOP BANDWIDTH
        # =========================================================

        # Timing-loop bandwidth in Hz.
        self.loop_bandwidth = (
            self.loop_bandwidth_ratio
            * self.symbol_rate
        )

        # Convert loop bandwidth to angular frequency.
        self.natural_frequency = (
            2.0
            * np.pi
            * self.loop_bandwidth
        )

        # The timing loop is updated once per recovered symbol,
        # therefore the loop update interval is the symbol period,
        # NOT the simulation dt.
        self.loop_update_period = (
            self.symbol_period
        )

        # =========================================================
        # DISCRETE TIMING PI COEFFICIENTS
        # =========================================================
        #
        # The Gardner timing error is normalized before entering
        # this loop, so the timing-detector gain is treated as
        # unity for the loop design.
        #
        # The coefficients below are the standard second-order
        # discrete PI form used for a normalized type-II loop.
        # =========================================================

        theta = (
            self.natural_frequency
            * self.loop_update_period
            / (
                self.damping_factor
                + 0.25
                / self.damping_factor
            )
        )

        denominator = (
            1.0
            + 2.0
            * self.damping_factor
            * theta
            + theta * theta
        )

        self.kp = (
            4.0
            * self.damping_factor
            * theta
            / denominator
        )

        self.ki = (
            4.0
            * theta
            * theta
            / denominator
        )

        # =========================================================
        # TIMING POSITION
        # =========================================================

        # If not supplied, timing is initialized when processing
        # begins.
        if initial_timing is None:
            self.timing_position = None
        else:
            self.timing_position = float(
                initial_timing
            )

        # =========================================================
        # TIMING LOOP STATE
        # =========================================================

        self.integrator_state = 0.0

        # =========================================================
        # PREVIOUS SYMBOL
        # =========================================================

        self.previous_symbol = None

        # =========================================================
        # SIGNAL POWER ESTIMATE
        # =========================================================

        # Used to normalize the Gardner TED output so that loop
        # gains do not depend strongly on signal amplitude.
        self.signal_power = 0.0

        # Exponential averaging coefficient.
        self.power_alpha = 0.01

        # =========================================================
        # STREAMING INPUT BUFFER
        # =========================================================

        self.buffer = []

        # Absolute input-sample index represented by buffer[0].
        self.buffer_start_index = 0

        self.buffer_size = int(buffer_size)

        if self.buffer_size < 16:
            raise ValueError(
                "buffer_size must be at least 16."
            )

        # =========================================================
        # DIAGNOSTIC STATE
        # =========================================================

        self.timing_error = 0.0
        self.timing_correction = 0.0

    # =============================================================
    # TIMING INITIALIZATION
    # =============================================================

    def _initialize_timing(self):
        """
        Initializes the nominal symbol timing.

        The initial estimate is one nominal symbol period into
        the matched-filter output stream.

        Gardner subsequently corrects the timing phase.
        """

        if self.timing_position is not None:
            return

        self.timing_position = (
            self.nominal_samples_per_symbol
        )

    # =============================================================
    # STREAM INPUT
    # =============================================================

    def _append_sample(self, sample):
        """
        Appends one RRC matched-filter output sample.
        """

        self.buffer.append(
            float(sample)
        )

    # =============================================================
    # SIGNAL POWER
    # =============================================================

    def _update_signal_power(self, sample):
        """
        Updates the exponentially averaged baseband signal power.
        """

        sample_power = sample * sample

        if self.signal_power == 0.0:
            self.signal_power = sample_power
            return

        self.signal_power = (
            (1.0 - self.power_alpha)
            * self.signal_power
            +
            self.power_alpha
            * sample_power
        )

    # =============================================================
    # BUFFER MANAGEMENT
    # =============================================================

    def _has_required_samples(self):
        """
        Checks whether the input buffer contains all samples
        required for the current Gardner interpolation points.
        """

        if self.timing_position is None:
            return False

        current_position = (
            self.timing_position
        )

        midpoint_position = (
            current_position
            - self.samples_per_symbol / 2.0
        )

        minimum_position = min(
            current_position,
            midpoint_position,
        )

        maximum_position = max(
            current_position,
            midpoint_position,
        )

        minimum_index = int(
            np.floor(minimum_position)
        )

        maximum_index = (
            int(
                np.floor(maximum_position)
            )
            + 1
        )

        buffer_end_index = (
            self.buffer_start_index
            + len(self.buffer)
            - 1
        )

        return (
            minimum_index
            >= self.buffer_start_index
            and
            maximum_index
            <= buffer_end_index
        )

    def _trim_buffer(self):
        """
        Removes samples that can no longer be required by
        the timing loop.
        """

        if self.timing_position is None:
            return

        # We need to retain enough history for the previous
        # symbol / timing interval.
        keep_from_position = (
            self.timing_position
            - self.samples_per_symbol
        )

        keep_from_index = int(
            np.floor(
                keep_from_position
            )
        )

        remove_count = (
            keep_from_index
            - self.buffer_start_index
        )

        if remove_count <= 0:
            return

        remove_count = min(
            remove_count,
            len(self.buffer),
        )

        del self.buffer[
            :remove_count
        ]

        self.buffer_start_index += (
            remove_count
        )

        # Safety limit.
        if len(self.buffer) > self.buffer_size:

            excess = (
                len(self.buffer)
                - self.buffer_size
            )

            del self.buffer[
                :excess
            ]

            self.buffer_start_index += (
                excess
            )

    # =============================================================
    # FRACTIONAL INTERPOLATION
    # =============================================================

    def _interpolate(self, position):
        """
        Linearly interpolates the streaming signal at a
        fractional input-sample position.
        """

        lower_index = int(
            np.floor(position)
        )

        fraction = (
            position
            - lower_index
        )

        lower_buffer_index = (
            lower_index
            - self.buffer_start_index
        )

        upper_buffer_index = (
            lower_buffer_index
            + 1
        )

        x0 = self.buffer[
            lower_buffer_index
        ]

        x1 = self.buffer[
            upper_buffer_index
        ]

        return (
            (1.0 - fraction) * x0
            + fraction * x1
        )

    # =============================================================
    # GARDNER SAMPLE EXTRACTION
    # =============================================================

    def _get_gardner_samples(self):
        """
        Obtains:

            previous symbol
            midpoint sample
            current symbol
        """

        current_position = (
            self.timing_position
        )

        midpoint_position = (
            current_position
            - self.samples_per_symbol / 2.0
        )

        current_symbol = (
            self._interpolate(
                current_position
            )
        )

        midpoint_sample = (
            self._interpolate(
                midpoint_position
            )
        )

        previous_symbol = (
            self.previous_symbol
        )

        return (
            previous_symbol,
            midpoint_sample,
            current_symbol,
        )

    # =============================================================
    # GARDNER TIMING ERROR DETECTOR
    # =============================================================

    def _calculate_timing_error(
        self,
        previous_symbol,
        midpoint_sample,
        current_symbol,
    ):
        """
        Calculates the Gardner timing error.

        For real BPSK:

            e[k] =
                x[k - 1/2]
                *
                (x[k - 1] - x[k])
        """

        if previous_symbol is None:
            return None

        raw_error = (
            midpoint_sample
            * (
                previous_symbol
                - current_symbol
            )
        )

        # ---------------------------------------------------------
        # Normalize the TED output by signal power.
        #
        # This prevents the loop gain from changing simply because
        # the received signal amplitude changes.
        # ---------------------------------------------------------

        power = max(
            self.signal_power,
            1e-12,
        )

        normalized_error = (
            raw_error / power
        )

        return normalized_error

    # =============================================================
    # TIMING PI LOOP
    # =============================================================

    def _update_timing_rate(
        self,
        timing_error,
    ):
        """
        Updates the discrete timing PI loop.

        I[k] = I[k-1] + Ki * e[k]

        correction[k] =
            Kp * e[k] + I[k]

        omega[k] =
            nominal_samples_per_symbol
            + correction[k]
        """

        self.integrator_state += (
            self.ki
            * timing_error
        )

        self.timing_correction = (
            self.kp
            * timing_error
            + self.integrator_state
        )

        self.samples_per_symbol = (
            self.nominal_samples_per_symbol
            + self.timing_correction
        )

        # Prevent the timing-rate estimate from becoming
        # physically invalid.
        minimum_samples_per_symbol = (
            0.5
            * self.nominal_samples_per_symbol
        )

        maximum_samples_per_symbol = (
            1.5
            * self.nominal_samples_per_symbol
        )

        self.samples_per_symbol = np.clip(
            self.samples_per_symbol,
            minimum_samples_per_symbol,
            maximum_samples_per_symbol,
        )

        return self.samples_per_symbol

    # =============================================================
    # TIMING NCO
    # =============================================================

    def _advance_timing(self):
        """
        Advances the next recovered-symbol timing position.

            tau[k+1] =
                tau[k] + omega[k]
        """

        self.timing_position += (
            self.samples_per_symbol
        )

    # =============================================================
    # SYMBOL STATE
    # =============================================================

    def _save_previous_symbol(
        self,
        current_symbol,
    ):
        """
        Saves the current recovered symbol for the next
        Gardner timing-error calculation.
        """

        self.previous_symbol = (
            current_symbol
        )

    # =============================================================
    # PUBLIC STREAMING API
    # =============================================================

    def process(self, sample):
        """
        Processes exactly one RRC matched-filter output sample.

        Parameters
        ----------
        sample : float
            One oversampled RRC matched-baseband sample.

        Returns
        -------
        float or None
            Recovered symbol sample when a symbol is available.

            None when more input samples are required.
        """

        # ---------------------------------------------------------
        # 1. Add incoming sample.
        # ---------------------------------------------------------

        self._append_sample(
            sample
        )

        # ---------------------------------------------------------
        # 2. Update signal-power estimate.
        # ---------------------------------------------------------

        self._update_signal_power(
            float(sample)
        )

        # ---------------------------------------------------------
        # 3. Initialize timing.
        # ---------------------------------------------------------

        self._initialize_timing()

        # ---------------------------------------------------------
        # 4. Check whether enough samples are available.
        # ---------------------------------------------------------

        if not self._has_required_samples():
            return None

        # ---------------------------------------------------------
        # 5. Obtain Gardner samples.
        # ---------------------------------------------------------

        (
            previous_symbol,
            midpoint_sample,
            current_symbol,
        ) = self._get_gardner_samples()

        # ---------------------------------------------------------
        # 6. Calculate Gardner timing error.
        # ---------------------------------------------------------

        timing_error = (
            self._calculate_timing_error(
                previous_symbol,
                midpoint_sample,
                current_symbol,
            )
        )

        # ---------------------------------------------------------
        # 7. First recovered symbol.
        #
        # There is no Gardner error yet because there is no
        # previous recovered symbol.
        # ---------------------------------------------------------

        if timing_error is None:

            self._save_previous_symbol(
                current_symbol
            )

            self._advance_timing()

            self._trim_buffer()

            return current_symbol

        # ---------------------------------------------------------
        # 8. Store timing error for diagnostics.
        # ---------------------------------------------------------

        self.timing_error = (
            timing_error
        )

        # ---------------------------------------------------------
        # 9. Update timing PI loop.
        # ---------------------------------------------------------

        self._update_timing_rate(
            timing_error
        )

        # ---------------------------------------------------------
        # 10. Save current symbol.
        # ---------------------------------------------------------

        self._save_previous_symbol(
            current_symbol
        )

        # ---------------------------------------------------------
        # 11. Advance timing NCO.
        # ---------------------------------------------------------

        self._advance_timing()

        # ---------------------------------------------------------
        # 12. Remove old samples.
        # ---------------------------------------------------------

        self._trim_buffer()

        # ---------------------------------------------------------
        # 13. Return recovered symbol.
        # ---------------------------------------------------------

        return current_symbol

    # =============================================================
    # DIAGNOSTIC GETTERS
    # =============================================================

    def get_timing_error(self):
        """
        Returns the latest normalized Gardner timing error.
        """

        return self.timing_error

    def get_timing_position(self):
        """
        Returns the current timing position in input samples.
        """

        return self.timing_position

    def get_samples_per_symbol(self):
        """
        Returns the current timing-rate estimate.
        """

        return self.samples_per_symbol

    def get_nominal_samples_per_symbol(self):
        """
        Returns the nominal input samples per symbol.
        """

        return self.nominal_samples_per_symbol

    def get_loop_bandwidth(self):
        """
        Returns the timing-loop bandwidth in Hz.
        """

        return self.loop_bandwidth

    def get_kp(self):
        """
        Returns the calculated proportional gain.
        """

        return self.kp

    def get_ki(self):
        """
        Returns the calculated integral gain.
        """

        return self.ki
