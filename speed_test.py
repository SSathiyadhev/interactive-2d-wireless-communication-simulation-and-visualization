import time

from src.simulation_space import SimulationSpace
from src.wave_solver import WaveSolver
from src.transmitter import Transmitter
from src.receiver import Receiver
from src.link_evaluator import LinkEvaluator
from src.observation_point import ObservationPoint


def main():

    resolution_x = 600
    resolution_y = 600

    simulation_space = SimulationSpace(
        width=10.0,
        height=10.0,
        resolution_x=resolution_x,
        resolution_y=resolution_y,
        dt_stability_multiplier=0.85,
    )

    bit_rate = 500.0e6

    transmitter = Transmitter(
        simulation_space=simulation_space,
        x=1.0,
        y=5.0,
        carrier_frequency=1.0e9,
        carrier_amplitude=2.0,
        bit_rate=bit_rate,
    )

    receiver = Receiver(
        simulation_space=simulation_space,
        x=1.35,
        y=5.0,
        tuned_frequency=1.0e9,
        bit_rate=bit_rate,
    )

    evaluator = LinkEvaluator(
        transmitter=transmitter,
        receiver=receiver,
        speed_of_light=3.0e8,
        filter_group_delay_samples=8,
        warmup_bits=2,
    )

    evaluator.sync_receiver_delay()

    observation_point = ObservationPoint(
        simulation_space=simulation_space,
        x=2.0,
        y=5.0,
        buffer_duration=15e-9,
        label="RX Point",
    )

    wave_solver = WaveSolver(
        simulation_space,
        noise_level=0,
    )

    # -------------------------------------------------------------
    # Simulation duration
    # -------------------------------------------------------------

    simulation_duration = 15e-9

    # -------------------------------------------------------------
    # Timing
    # -------------------------------------------------------------

    start_time = time.perf_counter()

    steps = 0

    while simulation_space.time < simulation_duration:

        transmitter.transmit()

        wave_solver.solve()

        receiver.receive()

        observation_point.sample()

        evaluator.evaluate()

        simulation_space.advance_time()

        steps += 1

    end_time = time.perf_counter()

    # -------------------------------------------------------------
    # Results
    # -------------------------------------------------------------

    elapsed_time = end_time - start_time

    print()
    print("=" * 50)
    print("SIMULATION COMPLETE")
    print("=" * 50)

    print(f"Simulation time : {simulation_space.time * 1e9:.3f} ns")
    print(f"Execution time  : {elapsed_time:.6f} s")
    print(f"Total steps     : {steps}")

    print("=" * 50)


if __name__ == "__main__":
    main()
