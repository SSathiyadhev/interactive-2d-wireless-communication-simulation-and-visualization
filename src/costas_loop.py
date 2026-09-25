"""
src/costas_loop.py

Decision-directed Costas Loop for blind BPSK carrier phase and frequency tracking.
"""

import numpy as np
from src.filter import Filter


class CostasLoop:
    def __init__(
        self,
        dt,
        carrier_frequency,
        bit_rate,
        rrc_rolloff=0.35,
        loop_bandwidth_ratio=0.05,  # 1% of bit rate for stable lock
        damping_factor=0.707,
    ):
        self.dt = float(dt)
        self.carrier_frequency = float(carrier_frequency)
        self.bit_rate = float(bit_rate)
        self.rrc_rolloff = float(rrc_rolloff)
        self.damping_factor = float(damping_factor)

        # ---------------------------------------------------------
        # Arm Low-Pass Filters (Cut off 2fc carrier ripple)
        # ---------------------------------------------------------
        self.rrc_bandwidth = (self.bit_rate / 2.0) * (1.0 + self.rrc_rolloff)
        self.lpf_cutoff = 1.2 * self.rrc_bandwidth

        self.i_filter = Filter(
            "butterworth",
            self.dt,
            order=2,  # Order 2 avoids excessive loop phase lag
            filter_response="lowpass",
            cutoff_frequency=self.lpf_cutoff,
        )

        self.q_filter = Filter(
            "butterworth",
            self.dt,
            order=2,
            filter_response="lowpass",
            cutoff_frequency=self.lpf_cutoff,
        )

        # ---------------------------------------------------------
        # Discrete PI Loop Filter Gains (Gardner standard)
        # ---------------------------------------------------------
        self.loop_bw = loop_bandwidth_ratio * self.bit_rate
        wn = 2.0 * np.pi * self.loop_bw
        
        theta = (wn * self.dt) / (self.damping_factor + 0.25 / self.damping_factor)
        d = 1.0 + 2.0 * self.damping_factor * theta + theta * theta

        self.kp = (4.0 * self.damping_factor * theta) / d
        self.ki = (4.0 * theta * theta) / d

        # ---------------------------------------------------------
        # Oscillator & State Tracking
        # ---------------------------------------------------------
        self.phase = 0.0
        self.integrator_state = 0.0
        self.frequency_correction = 0.0

    def _mix(self, sample, time):
        total_phase = (
            2.0 * np.pi * self.carrier_frequency * time + self.phase
        )
        i_mixed = 2.0 * sample * np.cos(total_phase)
        q_mixed = -2.0 * sample * np.sin(total_phase)
        return i_mixed, q_mixed

    def _phase_detector(self, i, q):
        # Decision-directed BPSK phase error detector
        # e = -sgn(I) * Q drives phase to 0 or pi
        return -np.sign(i) * q

    def _loop_filter(self, error):
        error = np.clip(error, -2.0, 2.0)
        self.integrator_state += self.ki * error
        self.frequency_correction = self.kp * error + self.integrator_state
        return self.frequency_correction

    def process(self, sample, time):
        # 1. Downconversion mixing
        i_mixed, q_mixed = self._mix(sample, time)

        # 2. Arm low-pass filtering
        i = self.i_filter.filter(i_mixed)
        q = self.q_filter.filter(q_mixed)

        # 3. Phase error detection
        error = self._phase_detector(i, q)

        # 4. PI loop filter
        freq_corr = self._loop_filter(error)

        # 5. Advance NCO carrier phase offset
        self.phase -= freq_corr
        self.phase %= (2.0 * np.pi)

        # Return the in-phase baseband (or mixer output) and phase
        return i_mixed, self.phase
