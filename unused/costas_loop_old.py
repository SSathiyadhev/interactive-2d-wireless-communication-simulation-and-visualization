import numpy as np
from collections import deque
from scipy.signal import hilbert

from src.filter import Filter


class CostasLoop:

    def __init__(
        self,
        dt,
        carrier_frequency,
        bit_rate,
        rrc_rolloff=0.35,
    ):
        # Simulation time step
        self.dt = dt

        # Carrier frequency
        self.carrier_frequency = carrier_frequency

        # Bit/symbol rate
        self.bit_rate = bit_rate

        # RRC roll-off factor
        self.rrc_rolloff = rrc_rolloff

        # ---------------------------------------------------------
        # Costas loop design parameters
        # ---------------------------------------------------------

        # Fixed design ratios
        lpf_bandwidth_ratio = 1.2
        loop_bandwidth_ratio = 0.2

        # Critical damping
        self.damping_factor = 1.0

        # ---------------------------------------------------------
        # RRC bandwidth
        # ---------------------------------------------------------

        self.rrc_bandwidth = (
            self.bit_rate / 2.0
            * (1.0 + self.rrc_rolloff)
        )

        # ---------------------------------------------------------
        # I/Q low-pass filter cutoff
        # ---------------------------------------------------------

        self.lpf_cutoff_frequency = (
            lpf_bandwidth_ratio
            * self.rrc_bandwidth
        )

        # ---------------------------------------------------------
        # Costas loop bandwidth
        # ---------------------------------------------------------

        self.loop_bandwidth = (
            loop_bandwidth_ratio
            * self.bit_rate
        )

        # ---------------------------------------------------------
        # Loop natural frequency
        # ---------------------------------------------------------

        self.natural_frequency = (
            2.0 * np.pi * self.loop_bandwidth
        )

        # ---------------------------------------------------------
        # Phase detector gain
        # ---------------------------------------------------------

        self.detector_gain = 1.0

        # ---------------------------------------------------------
        # PI controller gains
        # ---------------------------------------------------------

        r = np.exp(
            -self.damping_factor
            * self.natural_frequency
            * self.dt
        )

        self.kp = (
            (1.0 - r**2)
            / (self.detector_gain * self.dt)
        )

        self.ki = (
            (1.0 - r) ** 2
            / (self.detector_gain * self.dt ** 2)
        )

        # ---------------------------------------------------------
        # Current estimated phase
        # ---------------------------------------------------------

        self.phase = 0.0

        # Current frequency correction from PI controller
        self.frequency_correction = 0.0

        # PI integrator state
        self.integrator_state = 0.0

        # ---------------------------------------------------------
        # I/Q low-pass filters
        # ---------------------------------------------------------

        # Low-pass filter for I branch
        self.i_filter = Filter(
            "butterworth",
            dt,
            order=4,
            filter_response="lowpass",
            cutoff_frequency=self.lpf_cutoff_frequency,
        )

        # Low-pass filter for Q branch
        self.q_filter = Filter(
            "butterworth",
            dt,
            order=4,
            filter_response="lowpass",
            cutoff_frequency=self.lpf_cutoff_frequency,
        )

        # ---------------------------------------------------------
        # Coarse frequency / phase estimation
        # ---------------------------------------------------------

        self.coarse_buffer_size = 250

        self.coarse_buffer = deque(
            maxlen=self.coarse_buffer_size
        )

        self.coarse_frequency = self.carrier_frequency
        self.coarse_phase = 0.0

        self._coarse_buffer_end_time = 0.0

    def _mix(self, sample, time):
        # Current local carrier phase
        phase = 2 * np.pi * self.carrier_frequency * time + self.phase

        # I and Q mixers
        i_mixed = sample * np.cos(phase)
        q_mixed = sample * -np.sin(phase)

        return i_mixed, q_mixed

    def _low_pass(self, i_mixed, q_mixed):
        # Remove the 2fc component from the I and Q signals
        i = self.i_filter.filter(i_mixed)
        q = self.q_filter.filter(q_mixed)

        return i, q

    def _phase_detector(self, i, q):
        # Estimate phase error from BPSK decision
        return np.sign(i) * q

    def _pi_controller(self, error):
        # Proportional term
        proportional = self.kp * error

        # Integral term
        self.integrator_state += self.ki * error * self.dt

        # Frequency correction
        self.frequency_correction = proportional + self.integrator_state

        return self.frequency_correction

    def _update_phase(self):
        # Update phase from estimated frequency
        self.phase += self.frequency_correction * self.dt

        # Keep phase within one cycle
        self.phase %= 2 * np.pi

        return self.phase

    def process(self, sample, time):

        # =========================================================
        # 1. Store sample for coarse estimation
        # =========================================================

        self.coarse_buffer.append(sample)
        self._coarse_buffer_end_time = time

        # =========================================================
        # 2. Normal Costas loop
        # =========================================================

        i_mixed, q_mixed = self._mix(
            sample,
            time,
        )

        i, q = self._low_pass(
            i_mixed,
            q_mixed,
        )

        error = self._phase_detector(
            i,
            q,
        )

        self._pi_controller(
            error,
        )

        # =========================================================
        # 3. Coarse correction when buffer is full
        # =========================================================

        if len(self.coarse_buffer) == self.coarse_buffer_size:
            (
                estimated_frequency,
                estimated_phase,
            ) = self._coarse_estimate()

            print(
                f"Coarse update: "
                f"t={time*1e9:.2f} ns, "
                f"f={estimated_frequency/1e9:.6f} GHz, "
                f"phase={np.degrees(estimated_phase):.2f}°"
            )

            self.carrier_frequency = estimated_frequency
            self.phase = estimated_phase

            self.coarse_frequency = estimated_frequency
            self.coarse_phase = estimated_phase

            self.coarse_buffer.clear()

        # =========================================================
        # 4. Update Costas phase ONCE
        # =========================================================

        phase = self._update_phase()

        return i_mixed, phase

    def _coarse_estimate(self):
        """
        Estimates carrier frequency and phase from
        the samples accumulated in the coarse buffer.
        """

        samples = np.asarray(
            self.coarse_buffer,
            dtype=float,
        )

        if len(samples) < 2:
            return (
                self.carrier_frequency,
                self.phase,
            )

        # ---------------------------------------------------------
        # Sampling frequency
        # ---------------------------------------------------------

        fs = 1.0 / self.dt

        # ---------------------------------------------------------
        # Convert real signal to analytic signal
        # ---------------------------------------------------------

        analytic = hilbert(samples)

        # ---------------------------------------------------------
        # Square to remove BPSK +/-180 degree modulation
        # ---------------------------------------------------------

        squared = analytic ** 2

        # ---------------------------------------------------------
        # FFT
        # ---------------------------------------------------------

        spectrum = np.fft.fft(squared)
        frequencies = np.fft.fftfreq(
            len(squared),
            d=self.dt,
        )

        # Only positive frequencies
        positive = frequencies > 0

        positive_spectrum = spectrum[positive]
        positive_frequencies = frequencies[positive]

        if len(positive_spectrum) == 0:
            return (
                self.carrier_frequency,
                self.phase,
            )

        # ---------------------------------------------------------
        # Find strongest spectral component
        # ---------------------------------------------------------

        peak_index = np.argmax(
            np.abs(positive_spectrum)
        )

        estimated_double_frequency = (
            positive_frequencies[peak_index]
        )

        estimated_frequency = (
            estimated_double_frequency / 2.0
        )

        # ---------------------------------------------------------
        # Actual time corresponding to each sample
        # ---------------------------------------------------------

        n = np.arange(len(samples))

        t0 = self._coarse_buffer_end_time - (
            len(samples) - 1
        ) * self.dt

        sample_times = (
            t0 + n * self.dt
        )

        # ---------------------------------------------------------
        # Estimate phase from squared analytic signal
        # ---------------------------------------------------------

        reference = np.exp(
            -1j
            * 2.0
            * np.pi
            * (2.0 * estimated_frequency)
            * sample_times
        )

        phasor = np.mean(
            squared * reference
        )

        # Squaring doubled the phase
        estimated_phase = (
            0.5 * np.angle(phasor)
        )

        # Keep phase in [0, 2*pi)
        estimated_phase %= 2.0 * np.pi

        # ---------------------------------------------------------
        # Return estimates
        # ---------------------------------------------------------

        return (
            estimated_frequency,
            estimated_phase,
        )
