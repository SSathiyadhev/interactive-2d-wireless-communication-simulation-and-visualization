"""
gardner_loop.py

Streaming Gardner symbol-timing recovery for an oversampled,
RRC-matched real BPSK baseband signal.

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
    normalized timing error
            |
            v
       2nd-order PI loop
            |
            +----------------------+
            |                      |
            v                      v
      timing-rate             timing-phase
       correction              correction
            |                      |
            +----------+-----------+
                       |
                       v
                  timing NCO
                       |
                       v
              fractional timing
                       |
                       v
                recovered symbol


Important
---------
The receiver does NOT need to know:

    - transmitter distance
    - propagation delay
    - transmit start time

The Gardner loop detects the timing offset from the received
waveform itself.

For BPSK:

    symbol_rate = bit_rate

The Gardner loop operates on symbol timing, not RF carrier phase.

The Costas loop should handle carrier phase/frequency recovery.
"""


import numpy as np


class GardnerLoop:

    def __init__(
        self,
        dt,
        symbol_rate,
        loop_bandwidth_ratio=0.01,
        damping_factor=0.707,
        ted_gain=1.0,
        initial_timing=None,
        buffer_size=1024,
    ):
        # =========================================================
        # BASIC PARAMETERS
        # =========================================================

        self.dt = float(dt)
        self.symbol_rate = float(symbol_rate)

        self.loop_bandwidth_ratio = float(
            loop_bandwidth_ratio
        )

        self.damping_factor = float(
            damping_factor
        )

        # Effective gain of the normalized Gardner TED.
        #
        # Start with 1.0.
        #
        # It is kept explicit because the loop-filter gains
        # depend on detector gain.
        self.ted_gain = float(ted_gain)

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

        if self.ted_gain <= 0:
            raise ValueError(
                "ted_gain must be greater "
                "than zero."
            )

        if buffer_size < 16:
            raise ValueError(
                "buffer_size must be at least 16."
            )

        self.buffer_size = int(
            buffer_size
        )

        # =========================================================
        # SYMBOL TIMING
        # =========================================================

        # Symbol period:
        #
        #       Ts = 1 / Rs
        #
        self.symbol_period = (
            1.0
            / self.symbol_rate
        )

        # Nominal number of input samples per symbol:
        #
        #       SPS = Ts / dt
        #
        self.nominal_samples_per_symbol = (
            self.symbol_period
            /
            self.dt
        )

        # Current timing-rate estimate.
        #
        # This is allowed to change only through the integral
        # timing-rate path.
        self.samples_per_symbol = (
            self.nominal_samples_per_symbol
        )

        # =========================================================
        # TIMING LOOP BANDWIDTH
        # =========================================================

        # Timing bandwidth is scaled from symbol rate:
        #
        #       Bn = ratio * Rs
        #
        self.loop_bandwidth = (
            self.loop_bandwidth_ratio
            *
            self.symbol_rate
        )

        # Angular loop bandwidth.
        self.natural_frequency = (
            2.0
            *
            np.pi
            *
            self.loop_bandwidth
        )

        # The loop updates once per recovered symbol.
        self.loop_update_period = (
            self.symbol_period
        )

        # =========================================================
        # SECOND-ORDER LOOP COEFFICIENTS
        # =========================================================

        #
        # Discrete second-order timing loop design.
        #
        # theta:
        #
        #     wn * T
        #
        theta = (
            self.natural_frequency
            *
            self.loop_update_period
            /
            (
                self.damping_factor
                +
                0.25
                /
                self.damping_factor
            )
        )

        denominator = (
            1.0
            +
            2.0
            *
            self.damping_factor
            *
            theta
            +
            theta
            *
            theta
        )

        # Raw PI coefficients.
        kp_base = (
            4.0
            *
            self.damping_factor
            *
            theta
            /
            denominator
        )

        ki_base = (
            4.0
            *
            theta
            *
            theta
            /
            denominator
        )

        # =========================================================
        # TED GAIN COMPENSATION
        # =========================================================

        # The loop sees:
        #
        #       timing_error = Kd * timing_phase_error
        #
        # Therefore compensate the PI gains by Kd.
        #
        self.kp = (
            kp_base
            /
            self.ted_gain
        )

        self.ki = (
            ki_base
            /
            self.ted_gain
        )

        # =========================================================
        # TIMING NCO STATE
        # =========================================================

        # Absolute fractional position of the current recovered
        # symbol in the input sample stream.
        #
        # This is the actual timing phase state.
        if initial_timing is None:
            self.timing_position = None
        else:
            self.timing_position = float(
                initial_timing
            )

        # Position of the previously recovered symbol.
        self.previous_timing_position = None

        # =========================================================
        # TIMING LOOP STATES
        # =========================================================

        # Integral state.
        #
        # This controls average timing rate / SPS.
        self.integrator_state = 0.0

        # Immediate proportional timing correction.
        #
        # Units: input samples.
        self.phase_correction = 0.0

        # Current timing-rate correction.
        #
        # Units: samples/symbol.
        self.timing_rate_correction = 0.0

        # =========================================================
        # PREVIOUS SYMBOL
        # =========================================================

        self.previous_symbol = None

        # =========================================================
        # SIGNAL POWER
        # =========================================================

        self.signal_power = 0.0

        self.power_alpha = 0.01

        # =========================================================
        # INPUT BUFFER
        # =========================================================

        self.buffer = []

        # Absolute sample index represented by buffer[0].
        self.buffer_start_index = 0

        # =========================================================
        # DIAGNOSTICS
        # =========================================================

        self.timing_error = 0.0

    # =============================================================
    # TIMING INITIALIZATION
    # =============================================================

    def _initialize_timing(self):

        if self.timing_position is not None:
            return

        # Start at nominal symbol spacing.
        #
        # Gardner will subsequently move this timing position
        # toward the actual received-symbol timing.
        self.timing_position = (
            self.nominal_samples_per_symbol
        )

    # =============================================================
    # INPUT BUFFER
    # =============================================================

    def _append_sample(
        self,
        sample,
    ):

        self.buffer.append(
            float(sample)
        )

    # =============================================================
    # SIGNAL POWER ESTIMATION
    # =============================================================

    def _update_signal_power(
        self,
        sample,
    ):

        sample_power = (
            sample
            *
            sample
        )

        if self.signal_power == 0.0:

            self.signal_power = (
                sample_power
            )

            return

        self.signal_power = (
            (1.0 - self.power_alpha)
            *
            self.signal_power
            +
            self.power_alpha
            *
            sample_power
        )

    # =============================================================
    # REQUIRED SAMPLE CHECK
    # =============================================================

    def _has_required_samples(self):

        if self.timing_position is None:
            return False

        current_position = (
            self.timing_position
        )

        # First symbol:
        #
        # Only current interpolation point is required.
        if self.previous_timing_position is None:

            minimum_position = (
                current_position
            )

            maximum_position = (
                current_position
            )

        else:

            # Gardner midpoint is halfway between the actual
            # previous and current timing positions.
            midpoint_position = (
                0.5
                *
                (
                    self.previous_timing_position
                    +
                    current_position
                )
            )

            minimum_position = min(
                self.previous_timing_position,
                midpoint_position,
                current_position,
            )

            maximum_position = max(
                self.previous_timing_position,
                midpoint_position,
                current_position,
            )

        minimum_index = int(
            np.floor(
                minimum_position
            )
        )

        maximum_index = (
            int(
                np.floor(
                    maximum_position
                )
            )
            +
            1
        )

        buffer_end_index = (
            self.buffer_start_index
            +
            len(self.buffer)
            -
            1
        )

        return (
            minimum_index
            >= self.buffer_start_index
            and
            maximum_index
            <= buffer_end_index
        )

    # =============================================================
    # BUFFER TRIMMING
    # =============================================================

    def _trim_buffer(self):

        if self.timing_position is None:
            return

        # We must retain the previous timing position because
        # the next Gardner calculation needs it.
        if self.previous_timing_position is not None:

            keep_from_position = min(
                self.previous_timing_position,
                self.timing_position,
            )

        else:

            keep_from_position = (
                self.timing_position
            )

        keep_from_index = int(
            np.floor(
                keep_from_position
            )
        )

        remove_count = (
            keep_from_index
            -
            self.buffer_start_index
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
                -
                self.buffer_size
            )

            del self.buffer[
                :excess
            ]

            self.buffer_start_index += (
                excess
            )

    # =============================================================
    # FRACTIONAL INTERPOLATOR
    # =============================================================

    def _interpolate(
        self,
        position,
    ):
        """
        Linear fractional interpolation.
        """

        lower_index = int(
            np.floor(
                position
            )
        )

        fraction = (
            position
            -
            lower_index
        )

        lower_buffer_index = (
            lower_index
            -
            self.buffer_start_index
        )

        upper_buffer_index = (
            lower_buffer_index
            +
            1
        )

        # These should be guaranteed by
        # _has_required_samples().
        x0 = self.buffer[
            lower_buffer_index
        ]

        x1 = self.buffer[
            upper_buffer_index
        ]

        return (
            (1.0 - fraction)
            *
            x0
            +
            fraction
            *
            x1
        )

    # =============================================================
    # GARDNER SAMPLE EXTRACTION
    # =============================================================

    def _get_gardner_samples(self):
        """
        Returns:

            previous symbol
            midpoint sample
            current symbol
        """

        current_position = (
            self.timing_position
        )

        # Current recovered symbol.
        current_symbol = (
            self._interpolate(
                current_position
            )
        )

        # No previous symbol/timing position yet.
        if (
            self.previous_timing_position
            is None
        ):

            return (
                self.previous_symbol,
                None,
                current_symbol,
            )

        # ---------------------------------------------------------
        # Gardner midpoint
        # ---------------------------------------------------------
        #
        # The midpoint is NOT assumed to be exactly
        # nominal_sps / 2 away.
        #
        # It is calculated from the actual timing positions.
        #
        midpoint_position = (
            0.5
            *
            (
                self.previous_timing_position
                +
                current_position
            )
        )

        midpoint_sample = (
            self._interpolate(
                midpoint_position
            )
        )

        return (
            self.previous_symbol,
            midpoint_sample,
            current_symbol,
        )

    # =============================================================
    # GARDNER TED
    # =============================================================

    def _calculate_timing_error(
        self,
        previous_symbol,
        midpoint_sample,
        current_symbol,
    ):
        """
        Gardner timing-error detector.

        For real BPSK:

            e[k] =
                x_mid
                *
                (x_prev - x_current)

        The error is normalized by received signal power.
        """

        if previous_symbol is None:
            return None

        raw_error = (
            midpoint_sample
            *
            (
                previous_symbol
                -
                current_symbol
            )
        )

        # ---------------------------------------------------------
        # Normalize TED output
        # ---------------------------------------------------------

        power = max(
            self.signal_power,
            1e-12,
        )

        timing_error = (
            raw_error
            /
            power
        )

        return timing_error

    # =============================================================
    # TIMING LOOP UPDATE
    # =============================================================

    def _update_timing_loop(
        self,
        timing_error,
    ):
        """
        Second-order timing loop.

        Two paths are maintained:

            Integral path
                -> timing-rate correction

            Proportional path
                -> immediate timing-phase correction
        """

        # =========================================================
        # 1. INTEGRAL / RATE PATH
        # =========================================================

        new_integrator = (
            self.integrator_state
            +
            self.ki
            *
            timing_error
        )

        candidate_sps = (
            self.nominal_samples_per_symbol
            +
            new_integrator
        )

        # ---------------------------------------------------------
        # Reasonable timing-rate limits
        # ---------------------------------------------------------

        minimum_sps = (
            0.5
            *
            self.nominal_samples_per_symbol
        )

        maximum_sps = (
            1.5
            *
            self.nominal_samples_per_symbol
        )

        clipped_sps = np.clip(
            candidate_sps,
            minimum_sps,
            maximum_sps,
        )

        # ---------------------------------------------------------
        # Anti-windup
        # ---------------------------------------------------------

        if clipped_sps == candidate_sps:

            self.integrator_state = (
                new_integrator
            )

        else:

            self.integrator_state = (
                clipped_sps
                -
                self.nominal_samples_per_symbol
            )

        # ---------------------------------------------------------
        # Updated timing rate
        # ---------------------------------------------------------

        self.samples_per_symbol = (
            clipped_sps
        )

        self.timing_rate_correction = (
            self.samples_per_symbol
            -
            self.nominal_samples_per_symbol
        )

        # =========================================================
        # 2. PROPORTIONAL / PHASE PATH
        # =========================================================

        # Immediate correction to the timing trajectory.
        #
        # This is what allows Gardner to move the sampling phase
        # to the correct point when the received waveform has an
        # unknown fixed delay.
        self.phase_correction = (
            self.kp
            *
            timing_error
        )

    # =============================================================
    # TIMING NCO
    # =============================================================

    def _advance_timing(self):
        """
        Advance the receiver's timing trajectory.

        Current timing position:

            t[k]

        Next timing position:

            t[k+1] =
                t[k]
                + timing_rate
                + phase_correction

        The timing position itself is the accumulated timing
        phase state.
        """

        # Preserve current timing location.
        self.previous_timing_position = (
            self.timing_position
        )

        # Advance timing trajectory.
        self.timing_position += (
            self.samples_per_symbol
            +
            self.phase_correction
        )

    # =============================================================
    # PREVIOUS SYMBOL
    # =============================================================

    def _save_previous_symbol(
        self,
        current_symbol,
    ):

        self.previous_symbol = (
            current_symbol
        )

    # =============================================================
    # MAIN STREAMING PROCESSOR
    # =============================================================

    def process(
        self,
        sample,
    ):
        """
        Process one RRC matched-filter output sample.

        Returns
        -------
        float or None

            Recovered symbol when timing loop produces one.

        """

        # =========================================================
        # 1. Add input sample
        # =========================================================

        self._append_sample(
            sample
        )

        # =========================================================
        # 2. Update signal power
        # =========================================================

        self._update_signal_power(
            float(sample)
        )

        # =========================================================
        # 3. Initialize timing
        # =========================================================

        self._initialize_timing()

        # =========================================================
        # 4. Check interpolation availability
        # =========================================================

        if not self._has_required_samples():

            return None

        # =========================================================
        # 5. Extract Gardner samples
        # =========================================================

        (
            previous_symbol,
            midpoint_sample,
            current_symbol,
        ) = self._get_gardner_samples()

        # =========================================================
        # 6. Calculate timing error
        # =========================================================

        timing_error = (
            self._calculate_timing_error(
                previous_symbol,
                midpoint_sample,
                current_symbol,
            )
        )

        # =========================================================
        # 7. First recovered symbol
        # =========================================================

        if timing_error is None:

            self.timing_error = 0.0

            self._save_previous_symbol(
                current_symbol
            )

            self._advance_timing()

            self._trim_buffer()

            return current_symbol

        # =========================================================
        # 8. Save timing error
        # =========================================================

        self.timing_error = (
            timing_error
        )

        # =========================================================
        # 9. Update timing loop
        # =========================================================

        self._update_timing_loop(
            timing_error
        )

        # =========================================================
        # 10. Save current symbol
        # =========================================================

        self._save_previous_symbol(
            current_symbol
        )

        # =========================================================
        # 11. Advance timing NCO
        # =========================================================

        self._advance_timing()

        # =========================================================
        # 12. Remove samples no longer required
        # =========================================================

        self._trim_buffer()

        # =========================================================
        # 13. Return recovered symbol
        # =========================================================

        return current_symbol

    # =============================================================
    # DIAGNOSTIC GETTERS
    # =============================================================

    def get_timing_error(self):

        return self.timing_error

    def get_timing_position(self):

        return self.timing_position

    def get_samples_per_symbol(self):

        return self.samples_per_symbol

    def get_nominal_samples_per_symbol(self):

        return self.nominal_samples_per_symbol

    def get_timing_phase_correction(self):

        return self.phase_correction

    def get_timing_rate_correction(self):

        return self.timing_rate_correction

    def get_loop_bandwidth(self):

        return self.loop_bandwidth

    def get_kp(self):

        return self.kp

    def get_ki(self):

        return self.ki

    def get_ted_gain(self):

        return self.ted_gain
