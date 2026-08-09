import numpy as np
import random

class Transmitter:
    """Generates BPSK modulated signals and injects them into SimulationSpace."""
    
    def __init__(self, simulation_space, x: float, y: float, 
                 carrier_frequency: float, carrier_amplitude: float, 
                 bit_rate: float, custom_bit_sequence: list = None):
        self.simulation_space = simulation_space
        self.x = float(x)
        self.y = float(y)
        self.carrier_frequency = float(carrier_frequency)
        self.carrier_amplitude = float(carrier_amplitude)
        self.bit_rate = float(bit_rate)
        
        # Stores user sequence or generated random bits
        self.custom_bit_sequence = custom_bit_sequence
        self.generated_bits = {}

    def get_bit_at_time(self, t: float) -> int:
        """Returns the bit (0 or 1) for the current time slot based on bit rate."""
        bit_index = int(t * self.bit_rate)
        
        # Use custom sequence if provided
        if self.custom_bit_sequence:
            return self.custom_bit_sequence[bit_index % len(self.custom_bit_sequence)]
            
        # Otherwise, generate and save a random bit for this new slot
        if bit_index not in self.generated_bits:
            self.generated_bits[bit_index] = random.choice([0, 1])
        return self.generated_bits[bit_index]

    def set_custom_bit_sequence(self, bit_sequence: list) -> None:
        """Sets a custom bit array to transmit."""
        self.custom_bit_sequence = bit_sequence
        self.generated_bits.clear()

    def clear_custom_bit_sequence(self) -> None:
        """Switches back to random bit generation."""
        self.custom_bit_sequence = None
        self.generated_bits.clear()

    def get_current_carrier_value(self) -> float:
        """Calculates carrier wave value at current time: Ac * cos(2 * pi * fc * t)"""
        t = self.simulation_space.time
        return self.carrier_amplitude * np.cos(2.0 * np.pi * self.carrier_frequency * t)

    def get_current_transmitted_value(self) -> float:
        """Applies BPSK modulation: Bit 0 -> +1, Bit 1 -> -1"""
        t = self.simulation_space.time
        current_bit = self.get_bit_at_time(t)
        
        polar_bit = 1.0 if current_bit == 0 else -1.0
        return polar_bit * self.get_current_carrier_value()

    def transmit(self) -> None:
        """Injects current BPSK signal value into SimulationSpace at (x, y)."""
        val = self.get_current_transmitted_value()
        # Cast x and y to int so numpy array indexing works in SimulationSpace
        self.simulation_space.set_field(int(self.x), int(self.y), val)

    # Position & Parameter Modifiers
    def set_position(self, x: float, y: float) -> None:
        self.x, self.y = float(x), float(y)

    def get_position(self) -> tuple:
        return (self.x, self.y)

    def set_carrier_frequency(self, frequency: float) -> None:
        self.carrier_frequency = float(frequency)

    def get_carrier_frequency(self) -> float:
        return self.carrier_frequency

    def set_carrier_amplitude(self, amplitude: float) -> None:
        self.carrier_amplitude = float(amplitude)

    def get_carrier_amplitude(self) -> float:
        return self.carrier_amplitude

    def set_bit_rate(self, bit_rate: float) -> None:
        self.bit_rate = float(bit_rate)

    def get_bit_rate(self) -> float:
        return self.bit_rate