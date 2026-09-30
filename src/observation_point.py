"""
src/observation_point.py

Defines the ObservationPoint class.

Records scalar electromagnetic field values at a specific grid coordinate
to perform real-time signal processing, including windowed Fast Fourier
Transform (FFT), spectral peak tracking, and cumulative/instantaneous signal energy.
"""

from collections import deque
import numpy as np


class ObservationPoint:
    """
    Monitors and analyzes field values at a fixed coordinate (x, y)
    in the simulation space.
    """

    def __init__(
        self,
        simulation_space,
        x,
        y,
        buffer_duration=20e-9,
        label="Observation Point",
    ):
        self.simulation_space = simulation_space
        self.label = label

        self.x = float(x)
        self.y = float(y)

        if not self.simulation_space.is_inside(self.x, self.y):
            raise ValueError(
                f"Observation point ({self.x} m, {self.y} m) "
                f"is outside the simulation space "
                f"(0..{self.simulation_space.width} m, "
                f"0..{self.simulation_space.height} m)."
            )

        self.buffer_duration = float(buffer_duration)
        if self.buffer_duration <= 0:
            raise ValueError("Buffer duration must be strictly positive.")

        # Rolling sample buffer capacity
        self.buffer_size = max(
            32,
            int(round(self.buffer_duration / self.simulation_space.dt)),
        )

        self.time_history = deque(maxlen=self.buffer_size)
        self.signal_history = deque(maxlen=self.buffer_size)

        # Running energy accumulation
        self._cumulative_energy = 0.0
        self.fft_window_type = "hann"
        self.fft_zero_padding_factor = 4

    # =============================================================
    # DATA ACQUISITION
    # =============================================================

    def sample(self):
        """
        Samples the field at (x, y) at the current simulation time step
        and updates running statistics.
        """
        t = self.simulation_space.time
        val = float(self.simulation_space.get_field(self.x, self.y))

        self.time_history.append(t)
        self.signal_history.append(val)

        # Integrate energy: E = sum(val^2 * dt)
        self._cumulative_energy += (val**2) * self.simulation_space.dt

        return val

    # =============================================================
    # ENERGY COMPUTATIONS
    # =============================================================

    def get_instantaneous_power(self):
        """Returns instantaneous power p(t) = s(t)^2."""
        if not self.signal_history:
            return 0.0
        return self.signal_history[-1] ** 2

    def get_windowed_energy(self):
        """
        Computes discrete signal energy within the current rolling window:
        E_window = integral(s(t)^2 dt) ~= sum(s[n]^2) * dt
        """
        if not self.signal_history:
            return 0.0
        samples = np.asarray(self.signal_history)
        return float(np.sum(samples**2) * self.simulation_space.dt)

    def get_cumulative_energy(self):
        """Returns total accumulated signal energy since the simulation started."""
        return self._cumulative_energy

    def get_rms_amplitude(self):
        """Calculates root-mean-square (RMS) amplitude over the rolling window."""
        if not self.signal_history:
            return 0.0
        samples = np.asarray(self.signal_history)
        return float(np.sqrt(np.mean(samples**2)))

    def set_fft_window(self, buffer_duration):
        buffer_duration = float(buffer_duration)

        if buffer_duration <= 0:
            raise ValueError(
                "FFT window duration must be strictly positive."
            )

        self.buffer_duration = buffer_duration

        self.buffer_size = max(
            32,
            int(
                round(
                    self.buffer_duration
                    / self.simulation_space.dt
                )
            ),
        )

        self.time_history = deque(maxlen=self.buffer_size)
        self.signal_history = deque(maxlen=self.buffer_size)

    def get_fft_window(self):
        return self.buffer_duration

    def set_fft_parameters(
        self,
        window_type=None,
        zero_padding_factor=None,
    ):
        if window_type is not None:
            window_type = str(window_type).lower()

            if window_type not in (
                "rect",
                "hann",
                "hamming",
                "blackman",
            ):
                raise ValueError(
                    f"Unsupported FFT window type: {window_type}"
                )

            self.fft_window_type = window_type

        if zero_padding_factor is not None:
            zero_padding_factor = int(zero_padding_factor)

            if zero_padding_factor < 1:
                raise ValueError(
                    "Zero-padding factor must be >= 1."
                )

            self.fft_zero_padding_factor = zero_padding_factor


    def get_fft_parameters(self):
        return (
            self.fft_window_type,
            self.fft_zero_padding_factor,
        )

    # =============================================================
    # FAST FOURIER TRANSFORM (FFT) & SPECTRUM
    # =============================================================

    def compute_fft(self):
        """
        Computes the single-sided amplitude spectrum of the stored signal.

        Returns:
            frequencies
            amplitudes
            frequency_bin_spacing
            true_frequency_resolution
        """

        n = len(self.signal_history)

        if n < 8:
            return (
                np.array([]),
                np.array([]),
                0.0,
                0.0,
            )

        signal = np.asarray(
            self.signal_history,
            dtype=np.float64,
        )

        # Window
        if self.fft_window_type == "hann":
            window = np.hanning(n)
        elif self.fft_window_type == "hamming":
            window = np.hamming(n)
        elif self.fft_window_type == "blackman":
            window = np.blackman(n)
        else:
            window = np.ones(n)

        windowed_signal = signal * window

        # Coherent gain
        window_sum = np.sum(window)
        coherent_gain = (
            window_sum / n
            if window_sum > 0
            else 1.0
        )

        # FFT
        n_fft = n * self.fft_zero_padding_factor
        dt = self.simulation_space.dt

        fft_values = np.fft.rfft(
            windowed_signal,
            n=n_fft,
        )

        frequencies = np.fft.rfftfreq(
            n_fft,
            d=dt,
        )

        amplitudes = (
            2.0 * np.abs(fft_values) / n
        ) / coherent_gain

        amplitudes[0] /= 2.0

        frequency_bin_spacing = (
            frequencies[1] - frequencies[0]
            if len(frequencies) > 1
            else 0.0
        )

        true_frequency_resolution = 1.0 / (n * dt)

        return (
            frequencies,
            amplitudes,
            frequency_bin_spacing,
            true_frequency_resolution,
        )

    def get_peak_frequency(self):
        """
        Finds the dominant frequency component excluding DC.
        """

        freqs, amps, _, _ = self.compute_fft()

        if len(freqs) <= 1:
            return 0.0, 0.0

        peak_idx = np.argmax(amps[1:]) + 1

        return (
            float(freqs[peak_idx]),
            float(amps[peak_idx]),
        )

    # =============================================================
    # STATE MANAGEMENT
    # =============================================================

    def reset(self):
        """Clears buffers and energy counters."""
        self.time_history.clear()
        self.signal_history.clear()
        self._cumulative_energy = 0.0

    def set_position(self, x, y):
        """Relocates the probe to a new grid point."""
        x = float(x)
        y = float(y)
        if not self.simulation_space.is_inside(x, y):
            raise ValueError(f"Position ({x}, {y}) is outside the simulation space.")
        self.x = x
        self.y = y
        self.reset()

    def get_time_series(self):
        """Returns time and field history as NumPy arrays."""
        return np.asarray(self.time_history), np.asarray(self.signal_history)