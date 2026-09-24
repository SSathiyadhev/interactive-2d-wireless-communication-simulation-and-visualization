"""
src/link_evaluator.py

Independent link analyzer that compares transmitted symbols with
demodulated receiver symbols, accounting for physical propagation
and DSP group delays to compute BER.
"""

import math


class LinkEvaluator:
    def __init__(self, transmitter, receiver, speed_of_light=3.0e8, filter_group_delay_samples=8, warmup_bits=2):
        self.tx = transmitter
        self.rx = receiver
        self.c = speed_of_light
        self.filter_group_delay_samples = filter_group_delay_samples
        self.warmup_bits = warmup_bits

        self.total_bits_compared = 0
        self.bit_errors = 0
        self.last_evaluated_idx = -1

    def calculate_propagation_delay_seconds(self):
        """Calculates distance-based propagation delay tau = d / c."""
        dx = self.rx.x - self.tx.x
        dy = self.rx.y - self.tx.y
        dist = math.hypot(dx, dy)
        return dist / self.c

    def get_estimated_total_delay_seconds(self):
        """Total link latency = physical channel delay + filter pipeline delay."""
        t_prop = self.calculate_propagation_delay_seconds()
        t_filter = self.filter_group_delay_samples * (1.0 / self.tx.bit_rate)
        return t_prop + t_filter

    def sync_receiver_delay(self):
        """Passes computed propagation delay to receiver for phase alignment."""
        t_prop = self.calculate_propagation_delay_seconds()
        if hasattr(self.rx, "set_estimated_propagation_delay"):
            self.rx.set_estimated_propagation_delay(t_prop)

    def evaluate(self):
        """
        Extracts demodulated bits from the receiver, aligns them with the
        transmitted bits, and updates BER metrics while ignoring startup transients.
        """
        rx_bits = getattr(self.rx, "demodulated_bits", [])
        tx_bits = getattr(self.tx, "generated_bits", [])

        if not rx_bits or not tx_bits:
            return

        while self.last_evaluated_idx + 1 < len(rx_bits):
            rx_idx = self.last_evaluated_idx + 1
            tx_idx = rx_idx

            # Ignore startup filter transient bits
            if tx_idx < self.warmup_bits:
                self.last_evaluated_idx = rx_idx
                continue

            if tx_idx >= len(tx_bits):
                break

            actual_tx = tx_bits[tx_idx]
            detected_rx = rx_bits[rx_idx]

            if actual_tx != detected_rx:
                self.bit_errors += 1

            self.total_bits_compared += 1
            self.last_evaluated_idx = rx_idx

    def get_bit_errors(self):
        return self.bit_errors

    def get_total_bits_compared(self):
        return self.total_bits_compared

    def get_bit_error_rate(self):
        if self.total_bits_compared == 0:
            return 0.0
        return self.bit_errors / self.total_bits_compared

    def reset(self):
        self.total_bits_compared = 0
        self.bit_errors = 0
        self.last_evaluated_idx = -1
    