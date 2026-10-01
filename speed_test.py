import time

from src.simulation_space import SimulationSpace
from src.wave_solver import WaveSolver
from src.transmitter import Transmitter
from src.receiver import Receiver
from src.link_evaluator import LinkEvaluator
from src.observation_point import ObservationPoint


def main():

    # -------------------------------------------------------------
    # Simulation space
    # -------------------------------------------------------------

    resolution_x = 1000
    resolution_y = 1000

    simulation_space = SimulationSpace(
        width=10.0,
        height=10.0,
        resolution_x=resolution_x,
        resolution_y=resolution_y,
        dt_stability_multiplier=0.25,
    )

    # -------------------------------------------------------------
    # Common parameters
    # -------------------------------------------------------------

    bit_rate = 400.0e6

    # -------------------------------------------------------------
    # 2 TRANSMITTERS
    # -------------------------------------------------------------

    transmitters = [

        Transmitter(
            simulation_space=simulation_space,
            x=1.0,
            y=5.0,
            carrier_frequency=1.0e9,
            carrier_amplitude=2.0,
            bit_rate=300.0e6,
        ),

        Transmitter(
            simulation_space=simulation_space,
            x=1.0,
            y=3.0,
            carrier_frequency=1.5e9,
            carrier_amplitude=2.0,
            bit_rate=400.0e6,
        ),

    ]

    # -------------------------------------------------------------
    # 2 RECEIVERS
    # -------------------------------------------------------------

    receivers = [

        Receiver(
            simulation_space=simulation_space,
            x=1.35,
            y=5.0,
            tuned_frequency=1.0e9,
            bit_rate=300.0e6,
        ),

        Receiver(
            simulation_space=simulation_space,
            x=1.35,
            y=3.0,
            tuned_frequency=1.5e9,
            bit_rate=400.0e6,
        ),

    ]

    # -------------------------------------------------------------
    # 2 LINK EVALUATORS
    # -------------------------------------------------------------

    evaluators = [

        LinkEvaluator(
            transmitter=transmitters[0],
            receiver=receivers[0],
            speed_of_light=3.0e8,
        ),

        LinkEvaluator(
            transmitter=transmitters[1],
            receiver=receivers[1],
            speed_of_light=3.0e8,
        ),

    ]

    # -------------------------------------------------------------
    # 1 OBSERVATION POINT
    # -------------------------------------------------------------

    observation_point = ObservationPoint(
        simulation_space=simulation_space,
        x=2.0,
        y=5.0,
        buffer_duration=15e-9,
        label="RX Point",
    )

    # -------------------------------------------------------------
    # WAVE SOLVER
    # -------------------------------------------------------------

    wave_solver = WaveSolver(
        simulation_space,
        noise_level=0,
    )

    # -------------------------------------------------------------
    # Simulation duration
    # -------------------------------------------------------------

    simulation_duration = 15e-9

    # -------------------------------------------------------------
    # START TIMING
    # -------------------------------------------------------------

    start_time = time.perf_counter()

    steps = 0

    while simulation_space.time < simulation_duration:

        # ---------------------------------------------------------
        # All TXs transmit BEFORE the solver
        # ---------------------------------------------------------

        for transmitter in transmitters:
            transmitter.transmit()

        # ---------------------------------------------------------
        # ONE FDTD SOLVE
        # ---------------------------------------------------------

        wave_solver.solve()

        # ---------------------------------------------------------
        # All RXs receive AFTER the solver
        # ---------------------------------------------------------

        for receiver in receivers:
            receiver.receive()

        # ---------------------------------------------------------
        # One observation point
        # ---------------------------------------------------------

        observation_point.sample()

        # ---------------------------------------------------------
        # Both link evaluators
        # ---------------------------------------------------------

        for evaluator in evaluators:
            evaluator.evaluate()

        # ---------------------------------------------------------
        # Advance simulation
        # ---------------------------------------------------------

        simulation_space.advance_time()

        steps += 1

    # -------------------------------------------------------------
    # STOP TIMING
    # -------------------------------------------------------------

    end_time = time.perf_counter()

    elapsed_time = end_time - start_time

    # -------------------------------------------------------------
    # Results
    # -------------------------------------------------------------

    print()
    print("=" * 55)
    print("SIMULATION SPEED TEST")
    print("=" * 55)

    print(f"Resolution      : {resolution_x} × {resolution_y}")
    print(f"Simulation time : {simulation_space.time * 1e9:.3f} ns")
    print(f"Execution time  : {elapsed_time:.6f} s")
    print(f"Total steps     : {steps}")

    print()
    print("Objects:")
    print(f"  Transmitters      : {len(transmitters)}")
    print(f"  Receivers         : {len(receivers)}")
    print(f"  Observation points: 1")
    print(f"  Link evaluators   : {len(evaluators)}")

    print()
    print(f"Simulation rate : {steps / elapsed_time:,.0f} steps/s")

    print("=" * 55)


if __name__ == "__main__":
    main()
