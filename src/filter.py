"""
filter.py

Defines the Filter class.

Filter provides a single, reusable, sample-by-sample streaming
filter implementation used by both Transmitter (pulse shaping) and
Receiver (band-pass pre-filtering + matched filtering).

Supported filter types
-----------------------
"butterworth"
    Butterworth IIR filter (lowpass / highpass / bandpass),
    implemented as second-order sections (SOS) for numerical
    stability at higher orders.

"rrc"
    Root Raised Cosine FIR filter, implemented as a direct-form
    FIR filter with streaming state.

Both filter types expose the exact same streaming API:

    filter(sample) -> one output sample, given one input sample.

Neither filter type requires the full signal, future samples, or
manual bookkeeping of past samples by the caller -- all required
history is stored internally and carried between calls.
"""

import numpy as np
from scipy.signal import butter, sosfilt, sos2tf, group_delay


class Filter:
    """
    Generic streaming filter supporting Butterworth and RRC designs.
    """

    SUPPORTED_TYPES = (
        "butterworth",
        "rrc",
    )

    def __init__(
        self,
        filter_type,
        dt,
        **parameters,
    ):
        """
        Constructor Arguments
        ----------------------
        filter_type : "butterworth" or "rrc"

        dt          : Time between consecutive input samples
                      (seconds). Used internally to compute
                      sample_frequency = 1 / dt.

        **parameters :
            Butterworth:
                order                  : Filter order.
                filter_response        : "lowpass", "highpass",
                                          or "bandpass".
                cutoff_frequency       : Hz. Used for lowpass/highpass.
                low_cutoff_frequency   : Hz. Used for bandpass.
                high_cutoff_frequency  : Hz. Used for bandpass.

            RRC:
                rolloff             : RRC roll-off factor (0 < alpha <= 1).
                samples_per_symbol  : Number of input samples per symbol.
                span                : Filter span, in symbols.
                normalize           : "peak" (default) or "energy".
                                      "peak"   -> max(|h|) == 1
                                                  (typical for a
                                                  transmit pulse-shaping
                                                  filter).
                                      "energy" -> sum(h**2) == 1
                                                  (typical for a
                                                  receive matched
                                                  filter).
        """

        if filter_type not in self.SUPPORTED_TYPES:
            raise ValueError(
                f"Unsupported filter_type '{filter_type}'. "
                f"Supported types: {self.SUPPORTED_TYPES}"
            )

        self.filter_type = filter_type
        self.dt = float(dt)

        if self.dt <= 0:
            raise ValueError(
                "dt must be greater than zero."
            )

        self._sample_frequency = 1.0 / self.dt
        self._parameters = dict(parameters)

        self._design()

    # =============================================================
    # PUBLIC API
    # =============================================================

    def filter(self, sample):
        """
        Processes exactly one input sample and returns exactly one
        output sample. Internal filter state is carried forward
        automatically between calls.
        """

        if self.filter_type == "butterworth":

            output_value, self._state = sosfilt(
                self._sos,
                [sample],
                zi=self._state,
            )

            return float(output_value[0])

        # filter_type == "rrc"

        output_value, self._state = _lfilter_one(
            self._coefficients,
            sample,
            self._state,
        )

        return output_value

    def reset(self):
        """
        Resets the internal streaming filter state.

        Does not change the filter design (coefficients) -- only
        clears history, as if the filter were newly constructed
        and had never processed any samples.
        """

        self._state = self._initial_state()

    def set_parameters(self, **parameters):
        """
        Updates filter parameters, redesigns the filter, and resets
        internal state. Only the provided keyword parameters are
        changed -- any parameters not supplied keep their previous
        values.
        """

        self._parameters.update(parameters)
        self._design()

    def get_filter_type(self):
        """
        Returns the current filter type ("butterworth" or "rrc").
        """

        return self.filter_type

    def get_parameters(self):
        """
        Returns a copy of the current filter parameters.
        """

        return dict(self._parameters)

    def get_group_delay_samples(self, frequency=None):
        """
        Returns this filter's group delay, in samples.

        For "rrc" (a symmetric linear-phase FIR filter), the group
        delay is exact and frequency-independent:
            (number_of_taps - 1) / 2
        `frequency` is ignored for this filter type.

        For "butterworth" (an IIR filter), group delay is
        frequency-dependent and is computed exactly at the
        specified `frequency` (Hz) using scipy's digital filter
        group-delay analysis. `frequency` is required for this
        filter type.
        """

        if self.filter_type == "rrc":
            return (len(self._coefficients) - 1) / 2.0

        # filter_type == "butterworth"

        if frequency is None:
            raise ValueError(
                "frequency (Hz) is required to compute group delay "
                "for a butterworth filter, since IIR group delay is "
                "frequency-dependent."
            )

        b, a = sos2tf(self._sos)

        digital_frequency = (
            2.0 * np.pi * frequency / self._sample_frequency
        )

        _, group_delay_samples = group_delay(
            (b, a),
            w=[digital_frequency],
        )

        return float(group_delay_samples[0])

    # =============================================================
    # INTERNAL: DESIGN
    # =============================================================

    def _design(self):
        """
        (Re)designs the filter from self._parameters and resets
        internal state.
        """

        if self.filter_type == "butterworth":
            self._design_butterworth()
        else:
            self._design_rrc()

        self.reset()

    def _design_butterworth(self):

        order = self._parameters["order"]
        response = self._parameters["filter_response"]

        if response in ("lowpass", "highpass"):

            cutoff_frequency = self._parameters["cutoff_frequency"]
            band_edges = cutoff_frequency

        elif response == "bandpass":

            low_cutoff_frequency = self._parameters[
                "low_cutoff_frequency"
            ]
            high_cutoff_frequency = self._parameters[
                "high_cutoff_frequency"
            ]
            band_edges = [
                low_cutoff_frequency,
                high_cutoff_frequency,
            ]

        else:
            raise ValueError(
                f"Unsupported filter_response '{response}'. "
                "Supported: 'lowpass', 'highpass', 'bandpass'."
            )

        self._sos = butter(
            order,
            band_edges,
            btype=response,
            fs=self._sample_frequency,
            output="sos",
        )

    def _design_rrc(self):

        alpha = float(self._parameters["rolloff"])
        sps = int(self._parameters["samples_per_symbol"])
        span = int(self._parameters["span"])
        normalize = self._parameters.get("normalize", "peak")

        if not (0.0 < alpha <= 1.0):
            raise ValueError(
                "RRC rolloff must satisfy 0 < rolloff <= 1."
            )

        if sps < 1:
            raise ValueError(
                "samples_per_symbol must be a positive integer."
            )

        number_of_taps = span * sps + 1

        # Time axis, expressed in units of symbol periods (Ts = 1).
        t = (
            np.arange(number_of_taps)
            - number_of_taps // 2
        ) / sps

        h = np.zeros_like(t, dtype=float)

        # Sample spacing in the same (symbol-period) units used by
        # `t`. Any true singularity of the RRC formula falls exactly
        # on the denominator hitting zero -- rather than guessing a
        # tolerance on `t` itself (which depends on how closely a
        # sample happens to land near the singularity), we detect
        # the singularity directly from the denominator magnitude.
        # This is robust regardless of alpha / samples_per_symbol.
        singularity_epsilon = 1e-9

        for i, ti in enumerate(t):

            if np.isclose(ti, 0.0):

                h[i] = (
                    1.0
                    + alpha * (4.0 / np.pi - 1.0)
                )
                continue

            denominator = (
                np.pi
                * ti
                * (1.0 - (4.0 * alpha * ti) ** 2)
            )

            if abs(denominator) < singularity_epsilon:

                # Limiting value of the RRC formula as
                # t -> +/- 1 / (4 * alpha).
                h[i] = (
                    alpha / np.sqrt(2.0)
                ) * (
                    (1.0 + 2.0 / np.pi)
                    * np.sin(np.pi / (4.0 * alpha))
                    +
                    (1.0 - 2.0 / np.pi)
                    * np.cos(np.pi / (4.0 * alpha))
                )
                continue

            numerator = (
                np.sin(np.pi * ti * (1.0 - alpha))
                + 4.0 * alpha * ti
                * np.cos(np.pi * ti * (1.0 + alpha))
            )

            h[i] = numerator / denominator

        if normalize == "peak":
            h = h / np.max(np.abs(h))
        elif normalize == "energy":
            h = h / np.sqrt(np.sum(h ** 2))
        else:
            raise ValueError(
                f"Unsupported normalize option '{normalize}'. "
                "Supported: 'peak', 'energy'."
            )

        self._coefficients = h

    # =============================================================
    # INTERNAL: STATE
    # =============================================================

    def _initial_state(self):
        """
        Returns the correct zero-initial-condition state for the
        current filter design.

        Zero initial state is exact (not an approximation) here,
        because every signal in this simulator genuinely begins
        from silence at t = 0 -- there is no real "past" signal
        whose steady-state condition would need to be matched.
        """

        if self.filter_type == "butterworth":
            return np.zeros((self._sos.shape[0], 2))

        # filter_type == "rrc"
        return np.zeros(len(self._coefficients) - 1)


def _lfilter_one(coefficients, sample, state):
    """
    Applies a direct-form FIR filter to exactly one input sample,
    given the filter's current delay-line state, and returns the
    output sample together with the updated state.

    Implemented directly (rather than calling scipy.signal.lfilter
    once per sample) to avoid per-call function-call overhead when
    this runs once per simulation timestep.
    """

    # Direct-form FIR: y[n] = sum_k h[k] * x[n-k]
    #
    # `state` holds x[n-1], x[n-2], ..., x[n-(N-1)] from the
    # previous call (oldest last), where N = len(coefficients).

    output_value = coefficients[0] * sample

    if len(state) > 0:
        output_value += np.dot(coefficients[1:], state)

        new_state = np.empty_like(state)
        new_state[0] = sample
        new_state[1:] = state[:-1]
        state = new_state

    return float(output_value), state