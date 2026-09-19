import numpy as np

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
            2.0
            * np.pi
            * self.loop_bandwidth
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

    def _mix(self, sample, time):
        # Current local carrier phase
        phase = (
            2
            * np.pi
            * self.carrier_frequency
            * time
            + self.phase
        )

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
        return -np.sign(i) * q

    def _pi_controller(self, error):
        # Proportional term
        proportional = self.kp * error

        # Integral term
        self.integrator_state += (
            self.ki
            * error
            * self.dt
        )

        # Frequency correction
        self.frequency_correction = (
            proportional
            + self.integrator_state
        )

        return self.frequency_correction

    def _update_phase(self):
        # Update phase from estimated frequency
        self.phase += (
            self.frequency_correction
            * self.dt
        )

        # Keep phase within one cycle
        self.phase %= 2 * np.pi

        return self.phase

    def process(self, sample, time):

        # Mix with local oscillator
        i_mixed, q_mixed = self._mix(
            sample,
            time,
        )

        # Low-pass filter I and Q
        i, q = self._low_pass(
            i_mixed,
            q_mixed,
        )

        # Calculate phase error
        error = self._phase_detector(
            i,
            q,
        )

        # PI controller
        self._pi_controller(
            error,
        )

        # Update estimated phase
        return (
            i_mixed,
            self._update_phase()
        )
