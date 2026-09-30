"""
src/link_evaluator.py

Independent link analyzer that compares transmitted symbols with
demodulated receiver symbols and calculates BER.

Synchronization is performed using a known bit sequence transmitted
at the beginning of the transmission.

The synchronization detector:
    - Uses a rolling window containing the last N received bits.
    - Calculates correlation with the known synchronization sequence.
    - Tracks the strongest correlation peak.
    - Uses upper/lower thresholds based on the synchronization
      sequence length.
    - Supports BPSK polarity inversion.
    - Locks to the stored maximum-correlation position once the
      correlation peak has passed.

Also provides physical distance and propagation delay between
the transmitter and receiver.
"""

import math


class LinkEvaluator:

    def __init__(
        self,
        transmitter,
        receiver,
        speed_of_light=3.0e8,
        upper_fraction=0.75,
        lower_fraction=0.625,
    ):
        self.tx = transmitter
        self.rx = receiver
        self.c = speed_of_light

        # ---------------------------------------------------------
        # BER state
        # ---------------------------------------------------------

        self.total_bits_compared = 0
        self.bit_errors = 0

        # ---------------------------------------------------------
        # Synchronization state
        # ---------------------------------------------------------

        self.sync_found = False
        self.sync_index = None
        self.inverted = False
        self.sync_decision_time = None

        # Peak detection state
        self.peak_active = False

        self.max_correlation = 0
        self.max_correlation_index = None
        self.max_correlation_inverted = False

        # Track how many RX bits have been processed by the
        # synchronization detector.
        self.last_rx_bit_count = 0

        # Track how many RX bits have already been printed.
        self.last_printed_rx_bit_count = 0

        # ---------------------------------------------------------
        # Synchronization threshold fractions
        # ---------------------------------------------------------

        self.upper_fraction = upper_fraction
        self.lower_fraction = lower_fraction

    # =============================================================
    # Physical link information
    # =============================================================

    def calculate_distance(self):
        """Calculates the physical distance between transmitter
        and receiver.
        """

        dx = self.rx.x - self.tx.x
        dy = self.rx.y - self.tx.y

        return math.hypot(dx, dy)

    def calculate_propagation_delay_seconds(self):
        """Calculates propagation delay tau = d / c."""

        distance = self.calculate_distance()

        return distance / self.c

    def get_estimated_total_delay_seconds(self):
        """Returns the physical propagation delay."""

        return self.calculate_propagation_delay_seconds()

    # =============================================================
    # Correlation
    # =============================================================

    def _calculate_correlation(self, rx_window, sync_bits):
        """
        Calculates BPSK correlation between an RX window and the
        known synchronization sequence.

        Bit mapping:

            1 -> +1
            0 -> -1

        A positive correlation indicates normal polarity.

        A negative correlation indicates inverted BPSK polarity.
        """

        correlation = 0

        for rx_bit, sync_bit in zip(
            rx_window,
            sync_bits,
        ):
            rx_value = 1 if rx_bit == 1 else -1
            sync_value = 1 if sync_bit == 1 else -1

            correlation += rx_value * sync_value

        return correlation

    # =============================================================
    # Synchronization
    # =============================================================

    def _update_synchronization(
        self,
        rx_bits,
        sync_bits,
    ):
        """
        Updates the rolling synchronization detector.

        Only the latest N RX bits are examined, where N is the
        synchronization sequence length.

        The detector:

            1. Waits until N bits are available.
            2. Calculates correlation.
            3. Starts peak tracking when the upper threshold
               is crossed.
            4. Stores the maximum correlation and its index.
            5. Continues tracking while the peak is active.
            6. When correlation falls below the lower threshold,
               the peak is considered complete.
            7. The stored maximum-correlation position becomes
               the synchronization position.

        Both normal and inverted BPSK polarity are supported.
        """

        sync_length = len(sync_bits)

        if sync_length == 0:
            return

        # ---------------------------------------------------------
        # Only process when a NEW RX bit has been decoded.
        # ---------------------------------------------------------

        if len(rx_bits) <= self.last_rx_bit_count:
            return

        self.last_rx_bit_count = len(rx_bits)

        if len(rx_bits) < sync_length:
            return

        # ---------------------------------------------------------
        # Latest N received bits
        # ---------------------------------------------------------

        window_start = (
            len(rx_bits) - sync_length
        )

        rx_window = rx_bits[
            window_start:
        ]

        # ---------------------------------------------------------
        # Calculate correlation
        # ---------------------------------------------------------

        correlation = self._calculate_correlation(
            rx_window,
            sync_bits,
        )

        absolute_correlation = abs(
            correlation
        )

        # ---------------------------------------------------------
        # Thresholds scale with sync sequence length
        # ---------------------------------------------------------

        upper_threshold = (
            self.upper_fraction
            * sync_length
        )

        lower_threshold = (
            self.lower_fraction
            * sync_length
        )

        # ---------------------------------------------------------
        # Start peak tracking
        # ---------------------------------------------------------

        if not self.peak_active:

            if absolute_correlation >= upper_threshold:

                self.peak_active = True

                self.max_correlation = (
                    absolute_correlation
                )

                self.max_correlation_index = (
                    window_start
                )

                self.max_correlation_inverted = (
                    correlation < 0
                )

            return

        # ---------------------------------------------------------
        # Peak is active
        # ---------------------------------------------------------

        if absolute_correlation > self.max_correlation:

            self.max_correlation = (
                absolute_correlation
            )

            self.max_correlation_index = (
                window_start
            )

            self.max_correlation_inverted = (
                correlation < 0
            )

            return

        # ---------------------------------------------------------
        # Peak has passed
        # ---------------------------------------------------------

        if absolute_correlation < lower_threshold:

            if self.max_correlation_index is not None:

                self.sync_found = True

                self.sync_index = (
                    self.max_correlation_index
                )

                self.inverted = (
                    self.max_correlation_inverted
                )

                # -----------------------------------------------------
                # Exact simulation time when synchronization was decided
                # -----------------------------------------------------

                decision_rx_index = len(rx_bits) - 1

                bit_times = getattr(
                    self.rx,
                    "demodulated_bit_times",
                    [],
                )

                if decision_rx_index < len(bit_times):
                    self.sync_decision_time = (
                        bit_times[decision_rx_index]
                    )

                print(
                    "\n========== SYNC DECISION =========="
                )
                print(
                    f"Decision time       : "
                    f"{self.sync_decision_time * 1e9:.6f} ns"
                )
                print(
                    f"Decision RX index   : "
                    f"{decision_rx_index}"
                )
                print(
                    f"Sync index          : "
                    f"{self.sync_index}"
                )
                print(
                    f"Decision correlation: "
                    f"{correlation}"
                )
                print(
                    f"Decision |corr|     : "
                    f"{absolute_correlation}"
                )
                print(
                    f"Peak correlation    : "
                    f"{self.max_correlation}"
                )
                print(
                    f"Lower threshold     : "
                    f"{lower_threshold}"
                )
                print(
                    f"Polarity            : "
                    f"{'INVERTED' if self.inverted else 'NORMAL'}"
                )
                print(
                    "===================================\n"
                )

            self.peak_active = False

    # =============================================================
    # Raw TX/RX bit display
    # =============================================================

    def _print_raw_bit_sequences(
        self,
        tx_bits,
        rx_bits,
    ):
        """
        Prints TX and RX bits with the simulation time at which
        each RX bit decision was made.

        RX decision time comes from:
            self.rx.demodulated_bit_times
        """

        if (
            len(rx_bits)
            <= self.last_printed_rx_bit_count
        ):
            return

        print("\nINDEX   TX   RX   RX DECISION TIME")

        max_length = max(
            len(tx_bits),
            len(rx_bits),
        )

        bit_times = getattr(
            self.rx,
            "demodulated_bit_times",
            [],
        )

        for i in range(max_length):

            tx_bit = (
                tx_bits[i]
                if i < len(tx_bits)
                else "-"
            )

            rx_bit = (
                rx_bits[i]
                if i < len(rx_bits)
                else "-"
            )

            if i < len(bit_times):
                decision_time = (
                    f"{bit_times[i] * 1e9:.6f} ns"
                )
            else:
                decision_time = "-"

            print(
                f"{i:5d}   "
                f"{tx_bit}    "
                f"{rx_bit}   "
                f"{decision_time}"
            )

        print()

        self.last_printed_rx_bit_count = (
            len(rx_bits)
        )

    # =============================================================
    # Main evaluator
    # =============================================================

    def evaluate(self):
        """
        Synchronizes using the known sequence and then compares
        newly received payload bits sequentially.
        """

        rx_bits = getattr(
            self.rx,
            "demodulated_bits",
            [],
        )

        tx_bits = getattr(
            self.tx,
            "generated_bits",
            [],
        )

        sync_bits = getattr(
            self.tx,
            "custom_bit_sequence",
            None,
        )

        if not rx_bits or not tx_bits or not sync_bits:
            return

        # =========================================================
        # Raw TX/RX bit display
        # =========================================================

        self._print_raw_bit_sequences(
            tx_bits,
            rx_bits,
        )

        # =========================================================
        # Synchronization
        # =========================================================

        if not self.sync_found:

            self._update_synchronization(
                rx_bits,
                sync_bits,
            )

            if not self.sync_found:
                return

        # =========================================================
        # Payload alignment
        # =========================================================

        sync_length = len(sync_bits)

        rx_payload_start = (
            self.sync_index
            + sync_length
        )

        tx_payload_start = sync_length

        # ---------------------------------------------------------
        # Available payload
        # ---------------------------------------------------------

        rx_payload = rx_bits[
            rx_payload_start:
        ]

        tx_payload = tx_bits[
            tx_payload_start:
        ]

        # ---------------------------------------------------------
        # Correct BPSK polarity if necessary
        # ---------------------------------------------------------

        if self.inverted:

            rx_payload = [
                1 - bit
                for bit in rx_payload
            ]

        # =========================================================
        # Compare only newly available bits
        # =========================================================

        available_bits = min(
            len(rx_payload),
            len(tx_payload),
        )

        if (
            available_bits
            <= self.total_bits_compared
        ):
            return

        for i in range(
            self.total_bits_compared,
            available_bits,
        ):

            if (
                rx_payload[i]
                != tx_payload[i]
            ):
                self.bit_errors += 1

            self.total_bits_compared += 1

    # =============================================================
    # BER results
    # =============================================================

    def get_bit_errors(self):
        return self.bit_errors

    def get_total_bits_compared(self):
        return self.total_bits_compared

    def get_bit_error_rate(self):

        if self.total_bits_compared == 0:
            return 0.0

        return (
            self.bit_errors
            / self.total_bits_compared
        )

    def get_sync_decision_time(self):
        return self.sync_decision_time

    # =============================================================
    # Reset
    # =============================================================

    def reset(self):

        self.total_bits_compared = 0
        self.bit_errors = 0

        # Synchronization
        self.sync_found = False
        self.sync_index = None
        self.inverted = False
        self.sync_decision_time = None

        # Peak detector
        self.peak_active = False

        self.max_correlation = 0
        self.max_correlation_index = None
        self.max_correlation_inverted = False

        self.last_rx_bit_count = 0
        self.last_printed_rx_bit_count = 0
