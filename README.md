# Interactive 2D Wireless Communication Simulation and Visualization

A modular 2D wireless communication simulation platform for electromagnetic wave propagation, BPSK communication, digital signal processing, link analysis, and real-time visualization.

The project combines a two-dimensional heterogeneous electromagnetic FDTD simulation with an end-to-end BPSK communication system. It allows electromagnetic propagation, material effects, receiver DSP, synchronization, BER analysis, FFT analysis, and individual signal-processing stages to be observed through an interactive browser-based interface.

---

# Why this repository exists?

This repository contains the implementation of a Signals and Systems Project Course project.

The objective is to develop a modular two-dimensional wireless communication simulator that connects physical electromagnetic wave propagation with digital communication and signal processing.

The simulator is designed so that the propagation model, communication system, receiver DSP chain, link evaluation, and visualization interface are implemented as separate modules.

The current implementation includes:

* Two-dimensional heterogeneous electromagnetic wave propagation
* Finite Difference Time Domain (FDTD) simulation
* BPSK digital communication
* Root Raised Cosine (RRC) pulse shaping
* RRC matched filtering
* RF band-pass filtering
* Costas carrier recovery
* Gardner symbol timing recovery
* BPSK symbol decisions
* Known-sequence synchronization
* BPSK polarity inversion detection
* BER evaluation
* Propagation-delay estimation
* Multiple transmitters and receivers
* Material-dependent electromagnetic properties
* Optional Gaussian noise
* Observation-point analysis
* FFT visualization
* Interactive scenario presets
* Real-time browser visualization

---

# Objectives

The project aims to:

* Simulate BPSK-based wireless communication.

* Simulate electromagnetic wave propagation in a two-dimensional environment.

* Visualize electromagnetic field propagation and communication signals.

* Analyze transmitted and received signals in both time and frequency domains.

* Study the effect of distance, noise, multiple transmitters, and physical materials on wireless communication.

* Demonstrate the complete receiver signal-processing chain.

* Demonstrate the interaction between electromagnetic propagation and digital communication.

* Provide an interactive environment for observing intermediate signal-processing stages.

---

# Technology Stack

* Python
* NumPy
* Numba
* FastAPI
* Uvicorn
* HTML
* CSS
* JavaScript
* HTML5 Canvas

NumPy is used for numerical arrays and signal-processing data.

Numba is used to accelerate the computationally intensive FDTD update.

FastAPI provides the backend and WebSocket communication.

HTML, CSS, JavaScript, and HTML5 Canvas provide the interactive browser interface.

---

# Project Scope

## Phase 1

* Single transmitter and single receiver

* BPSK modulation and demodulation

* Additive White Gaussian Noise (AWGN)

* Free-space path loss

* FFT visualization at the transmitter and receiver

* Signal-to-Noise Ratio (SNR) computation

* Bit Error Rate (BER) computation

---

## Phase 2

* Multiple transmitters

* Signal superposition

* Basic physical obstacles

* Material-based signal attenuation

* Point-in-space electromagnetic field analysis

* FFT analysis at any selected point

* Interactive 2D simulation canvas

---

## Planned Future Work (Phase 3)

The following features are planned for future development if time permits.

* Reflection modelling

* Additional material types

* Propagation delay through obstacles

* Phase shift through obstacles

* Enhanced information panels

---

# Current Implementation

The project has progressed beyond the original basic Phase 1 architecture.

The current implementation contains a complete physical propagation model together with an end-to-end BPSK receiver DSP chain.

Implemented functionality includes:

* Heterogeneous electromagnetic FDTD propagation
* Spatially varying permittivity
* Spatially varying permeability
* Spatially varying conductivity
* Material regions
* Absorbing boundary layer
* Multiple transmitters
* Multiple receivers
* BPSK signal generation
* RRC pulse shaping
* RF carrier generation
* RF BPSK modulation
* RF band-pass filtering
* Costas carrier recovery
* RRC matched filtering
* Gardner symbol timing recovery
* Hard BPSK decisions
* Known-sequence synchronization
* BPSK polarity inversion handling
* BER calculation
* Propagation-delay estimation
* Observation-point waveform analysis
* FFT analysis
* Optional Gaussian noise
* Interactive scenario presets
* Interactive transmitter configuration
* Interactive receiver configuration
* Interactive material configuration
* Real-time WebSocket telemetry
* Interactive browser-based visualization

---

# Software Architecture

The project is divided into modules responsible for the simulation environment, electromagnetic propagation, transmission, reception, filtering, synchronization, link evaluation, and visualization.

---

## SimulationSpace

`SimulationSpace` represents the computational electromagnetic environment.

Responsibilities include:

* Maintaining the physical simulation dimensions.
* Maintaining the computational grid.
* Maintaining the simulation time.
* Maintaining the simulation timestep.
* Storing the current electromagnetic field.
* Storing the previous electromagnetic field.
* Storing material-property maps.
* Providing access to electromagnetic field values.
* Providing access to permittivity, permeability, and conductivity maps.
* Managing the absorbing computational region.
* Mapping physical coordinates to the simulation grid.

The current simulation uses a two-dimensional computational grid.

The user-defined physical domain is surrounded by an absorbing computational region.

---

## WaveSolver

`WaveSolver` is the electromagnetic propagation engine.

The current implementation solves the heterogeneous electromagnetic wave equation using the Finite Difference Time Domain (FDTD) method.

The governing equation is:

<div align="center">
ε(x,y) ∂<sup>2</sup>E/∂t<sup>2</sup> + σ(x,y) ∂E/∂t = ∇ · ((1/μ(x,y)) ∇E)
</div>

where:

* <i>E</i>(x,y,t) is the electromagnetic field.
* ε(x,y) is the spatially varying permittivity.
* μ(x,y) is the spatially varying permeability.
* σ(x,y) is the spatially varying conductivity.

The spatial operator is implemented using the flux form:

<div align="center">
∇ · ((1/μ) ∇E)
</div>

rather than simply multiplying a Laplacian by a spatially varying coefficient.

This allows the spatial variation of permeability to be handled at cell interfaces.

For fixed material properties, the equation is linear, so electromagnetic fields obey linear superposition.

The solver:

* Advances the electromagnetic field one timestep at a time.
* Uses spatially varying material coefficients.
* Uses precomputed FDTD coefficients.
* Handles material interfaces using interface coefficients.
* Applies the absorbing boundary region.
* Supports optional Gaussian noise.
* Validates the timestep using the stability condition.
* Uses Numba JIT compilation for the numerical update.
* Uses parallel numerical execution for the grid update.

---

## Materials

`materials.py` defines electromagnetic material properties.

A material is described using:

* Relative permittivity ε<sub>r</sub>
* Relative permeability μ<sub>r</sub>
* Conductivity σ

The absolute properties are calculated using:

<div align="center">
ε = ε<sub>0</sub>ε<sub>r</sub>
</div>

<div align="center">
μ = μ<sub>0</sub>μ<sub>r</sub>
</div>

Materials can be assigned to rectangular regions of the simulation space.

The material properties are written into the corresponding regions of the simulation grid and are used by the FDTD solver.

The current project includes predefined material definitions and supports custom material properties through the material interface.

---

## Transmitter

`transmitter.py` implements the wireless BPSK transmitter.

The transmitter:

* Generates the transmitted bit sequence.
* Converts bits into BPSK symbols.
* Generates the baseband waveform.
* Applies RRC pulse shaping.
* Generates the RF carrier.
* Performs BPSK RF modulation.
* Injects the RF signal into the electromagnetic simulation.

The transmitted signal is injected as a soft source into the electromagnetic field.

The transmitter maintains signal-history buffers for visualization.

Available transmitter telemetry includes:

* Generated bits
* BPSK symbols
* Shaped/baseband waveform
* Carrier waveform
* BPSK RF waveform
* FFT data

---

## Receiver

`receiver.py` implements the complete wireless receiver.

The receiver samples the electromagnetic field at its configured position and processes the received signal through multiple DSP stages.

The current processing chain is:

```text
Electromagnetic Field
        |
        v
Antenna Sampling
        |
        v
RF Band-Pass Filter
        |
        v
Costas Mixer
        |
        +----> Mixed
        |
        v
Costas I Low-Pass Filter
        |
        v
RRC Matched Filter
        |
        v
Gardner Symbol Timing Recovery
        |
        v
BPSK Decision
        |
        v
Recovered Bits
```

The receiver maintains separate histories for the major processing stages.

The browser currently exposes:

* Antenna
* BP Filter
* Mixed
* Filtered Mixed
* Matched
* Bits
* FFT Spectrum

---

# Receiver Processing

## RF Band-Pass Filter

The received electromagnetic field is first passed through the receiver RF band-pass filter.

This stage removes signal components outside the configured RF passband before carrier recovery.

---

## Costas Loop

The Costas Loop performs carrier recovery for the BPSK signal.

The current implementation uses:

* I/Q mixing
* I/Q low-pass filters
* Decision-directed phase detection
* PI loop filtering
* Internal phase correction

The mixer equations are:

<div align="center">
I<sub>mixed</sub> = 2r(t) cos(θ)
</div>

<div align="center">
Q<sub>mixed</sub> = −2r(t) sin(θ)
</div>

The BPSK phase detector is:

<div align="center">
e = −sign(I)Q
</div>

The PI loop processes the phase error and updates the internal phase correction.

The raw I mixer output is exposed as:

```text
Mixed
```

The low-pass-filtered I output is exposed as:

```text
Filtered Mixed
```

The Costas Loop keeps its carrier phase state internally.

---

## RRC Matched Filter

The Costas I output is passed through the receiver RRC matched filter.

The matched filter provides the receiver-side counterpart to the transmitter RRC pulse-shaping filter.

Its output is exposed to the UI as:

```text
Matched
```

---

## Gardner Symbol Timing Recovery

The matched-filter output is passed to the Gardner timing recovery loop.

The Gardner loop performs:

* Fractional interpolation
* Timing-error detection
* Timing-error normalization
* Second-order loop filtering
* Timing correction
* Symbol extraction

The recovered symbols are then passed to the BPSK decision stage.

---

## BPSK Decision

The recovered symbol is converted into a binary decision.

The resulting recovered bits are stored and exposed through the receiver telemetry.

---

# Filter

`filter.py` provides the reusable streaming filter implementation.

The current filtering system supports:

### Butterworth IIR

* Low-pass
* High-pass
* Band-pass

The Butterworth implementation uses second-order sections for numerical stability.

### Root Raised Cosine FIR

The RRC filter is used for:

* Transmitter pulse shaping
* Receiver matched filtering

The RRC implementation supports configurable parameters including:

* Roll-off factor
* Samples per symbol
* Filter span
* Normalization

The filters operate in streaming mode and maintain their internal state between samples.

---

# GardnerLoop

`gardner_loop.py` implements streaming Gardner symbol timing recovery.

The processing chain is:

```text
Matched Filter Output
        |
        v
Fractional Interpolation
        |
        v
Gardner Timing Error Detector
        |
        v
Timing Error
        |
        v
Second-Order Loop Filter
        |
        v
Timing Correction
        |
        v
Recovered Symbol
```

The timing detector uses the real-valued BPSK signal to estimate the symbol timing error.

The loop includes:

* Timing phase correction
* Timing-rate correction
* Error normalization
* Timing-rate limiting
* Anti-windup
* Fractional interpolation

---

# ObservationPoint

`observation_point.py` represents a field-probe location in the simulation.

An observation point:

* Samples the electromagnetic field at a specified physical position.
* Maintains a signal history.
* Provides waveform information.
* Provides FFT analysis.

Observation points allow the electromagnetic field to be examined at selected locations independently of transmitter and receiver nodes.

---

# LinkEvaluator

`link_evaluator.py` evaluates a transmitter-receiver communication link.

The evaluator provides:

* Transmitter-receiver distance
* Propagation-delay estimation
* Known-sequence synchronization
* Correlation-based synchronization
* BPSK polarity inversion detection
* Received-bit alignment
* Bit comparison
* Bit-error counting
* BER calculation

The physical propagation delay is estimated from:

<div align="center">
τ = d/c
</div>

where:

* <i>d</i> is the transmitter-receiver distance.
* <i>c</i> is the configured propagation speed.

The synchronization process uses a known sequence and correlation to locate the received data.

The evaluator also handles the BPSK polarity inversion case.

The backend exposes link information including:

* BER
* Estimated delay
* Number of compared bits
* Number of bit errors
* Synchronization state
* Polarity inversion state

---

# Simulation Flow

```text
Initialize SimulationSpace
        |
        v
Load Scenario / Configuration
        |
        v
Create Material Maps
        |
        v
Create Transmitters
        |
        v
Create Receivers
        |
        v
Create Observation Points
        |
        v
Create WaveSolver
        |
        v
Create Link Evaluators
        |
        v
Start Simulation
        |
        v
+--------------------------------+
| Simulation Loop                |
|                                |
| Advance simulation time        |
|                                |
| Transmitters generate and      |
| inject RF signals              |
|                                |
| WaveSolver advances the        |
| electromagnetic field         |
|                                |
| Receivers sample the field     |
|                                |
| Receiver DSP processes samples |
|                                |
| Gardner recovers symbols       |
|                                |
| LinkEvaluator evaluates links  |
|                                |
| Observation points update      |
|                                |
| Backend sends telemetry        |
|                                |
| Frontend updates visualization |
+--------------------------------+
```

---

# Interactive Visualization

The project includes a browser-based interactive simulation interface.

The interface contains:

* 2D electromagnetic field visualization
* Simulation controls
* Simulation configuration
* Transmitter controls
* Receiver controls
* Material controls
* Observation-point controls
* Scenario preset selection
* Floating information panels
* Signal scope windows
* FFT visualization
* Link-evaluation information

The main simulation display uses two canvas layers:

```text
Field Canvas
    |
    +-- Electromagnetic field

Overlay Canvas
    |
    +-- Transmitter markers
    +-- Receiver markers
    +-- Material boundaries
    +-- Other simulation graphics
```

The field canvas displays the numerical electromagnetic field while the overlay canvas displays simulation objects and interaction elements.

---

# Signal Visualization

## Transmitter Scope

The transmitter scope provides:

* Bits
* Baseband
* Carrier Wave
* BPSK RF
* FFT Spectrum

---

## Receiver Scope

The receiver scope provides:

* Antenna
* BP Filter
* Mixed
* Filtered Mixed
* Matched
* Bits
* FFT Spectrum

### Antenna

The raw electromagnetic field value sampled at the receiver location.

### BP Filter

The received signal after RF band-pass filtering.

### Mixed

The raw in-phase output of the Costas mixer:

<div align="center">
I<sub>mixed</sub> = 2r(t) cos(θ)
</div>

### Filtered Mixed

The low-pass-filtered Costas I-channel output.

### Matched

The output of the receiver RRC matched filter.

### Bits

The recovered BPSK bit decisions.

---

## Observation Point Scope

Observation points provide:

* Field waveform
* FFT spectrum

---

# FFT Analysis

FFT analysis is available for transmitter, receiver, and observation-point signals.

The frontend provides configurable signal observation windows.

The current interface supports an observation-window range from 2 ns to 500 ns.

FFT processing also supports configurable windowing and zero-padding parameters through the signal-processing implementation.

---

# Simulation Configuration

The backend maintains configurable simulation parameters including:

* Physical width
* Physical height
* Grid resolution
* Time-step stability multiplier
* Noise level
* Simulation speed

The default runtime configuration uses:

```text
Physical domain:
10 m × 10 m

Default grid:
1000 × 1000

Minimum configured resolution:
100 × 100
```

The simulation timestep is derived from the spatial resolution and the configured stability multiplier.

---

# Scenario Presets

The backend provides predefined scenarios for quickly configuring the simulation.

## Single Link (Line of Sight)

A single transmitter communicates with a single receiver without a material obstacle.

Default configuration:

```text
TX:
Position      = (3.0, 5.0)
Carrier       = 1.5 GHz
Bit rate      = 400 Mbps
Amplitude     = 2.0

RX:
Position      = (7.0, 5.0)
Carrier       = 1.5 GHz
Bit rate      = 400 Mbps

Noise:
0
```

An observation point is also included.

---

## Dual Link (Line of Sight)

Contains two simultaneous transmitter-receiver links.

Default configuration:

```text
Link 0:
Carrier       = 1.5 GHz
Bit rate      = 400 Mbps

Link 1:
Carrier       = 1.0 GHz
Bit rate      = 300 Mbps
```

This scenario allows multiple transmitted electromagnetic signals to coexist in the same simulation environment.

---

## Single Link With Noise

Uses the single-link configuration with non-zero simulation noise.

This scenario is used to observe the effect of noise on the propagated signal and receiver processing.

---

## Single Link With Concrete

Contains a concrete material region between the transmitter and receiver.

The configured region is:

```text
x = 4.7 m to 5.3 m
y = 2.5 m to 7.5 m
```

This scenario allows the effect of spatially varying electromagnetic material properties on propagation to be visualized.

---

## Empty Canvas

Creates an empty simulation environment without predefined transmitters, receivers, observation points, links, or materials.

It can be used as a clean starting point for manually configuring the simulation.

---

# Backend

The backend is implemented using FastAPI.

The backend is responsible for:

* Serving the frontend.
* Maintaining the simulation runtime.
* Processing WebSocket control messages.
* Advancing the simulation.
* Managing transmitters.
* Managing receivers.
* Managing materials.
* Managing observation points.
* Managing link evaluators.
* Loading scenario presets.
* Sending electromagnetic field data.
* Sending signal telemetry.
* Sending link-performance information.

The backend exposes the current simulation state to the frontend through WebSocket telemetry.

Telemetry includes:

* Simulation state
* Grid dimensions
* Physical dimensions
* Computational dimensions
* Absorbing-layer information
* Simulation timestep
* Simulation time
* Transmitter data
* Receiver data
* Observation-point data
* Link-evaluator data
* Material data
* Noise level
* Available presets
* Current preset

---

# Frontend

The frontend is implemented using:

* HTML
* CSS
* JavaScript
* HTML5 Canvas

The frontend communicates with the FastAPI backend through WebSocket messages.

The interface provides controls for:

* Starting and pausing the simulation
* Resetting the simulation
* Changing simulation resolution
* Changing simulation parameters
* Changing noise level
* Loading scenario presets
* Adding and removing transmitters
* Configuring transmitter parameters
* Adding and removing receivers
* Configuring receiver parameters
* Adding observation points
* Configuring materials
* Viewing signal scopes
* Viewing FFT data
* Viewing link evaluation information

The preset selector is dynamically populated from the backend telemetry.

---

# Numerical Stability

The FDTD simulation timestep is derived from the spatial resolution and the stability requirements of the numerical solver.

The simulation uses a configurable timestep stability multiplier.

The solver validates the stability condition before running the numerical update.

The computational domain also contains an absorbing boundary region.

The absorbing region uses a graded conductivity profile to reduce artificial reflections from the outer computational boundary.

---

# Noise Model

The wave solver supports optional Gaussian noise.

The noise is applied during the FDTD field update.

Therefore, the noise becomes part of the propagated electromagnetic field rather than being added only to a receiver display.

The resulting noisy field is subsequently sampled by receivers and observation points.

The noise level can be configured through the backend and the interactive frontend.

---

# Material Model

The simulation space maintains spatial maps for:

* Permittivity
* Permeability
* Conductivity

Material properties are applied to the corresponding regions of the simulation grid.

The heterogeneous wave equation therefore uses different electromagnetic properties at different locations in the simulation domain.

This allows the simulation to represent an environment containing different material regions rather than treating the entire simulation space as a single homogeneous medium.

---

# Superposition

The heterogeneous wave equation implemented by the solver is linear for fixed material properties.

Therefore, if two electromagnetic fields E<sub>1</sub> and E<sub>2</sub> satisfy the same fixed-material equation, their sum also satisfies the equation:

<div align="center">
E = E<sub>1</sub> + E<sub>2</sub>
</div>

This allows multiple transmitter fields to propagate simultaneously and naturally superpose within the same simulation field.

---

# Repository Structure

```text
interactive-2d-wireless-communication-simulation-and-visualization/
│
├── server.py
├── requirements.txt
│
├── src/
│   ├── simulation_space.py
│   ├── wave_solver.py
│   ├── materials.py
│   ├── transmitter.py
│   ├── receiver.py
│   ├── filter.py
│   ├── costas_loop.py
│   ├── gardner_loop.py
│   ├── link_evaluator.py
│   └── observation_point.py
│
└── static/
    ├── index.html
    └── app.js
```

---

# Module Summary

| Module | Responsibility |
|---|---|
| `simulation_space.py` | Computational grid, electromagnetic field, material maps, simulation time |
| `wave_solver.py` | Heterogeneous electromagnetic FDTD propagation |
| `materials.py` | Electromagnetic material definitions |
| `transmitter.py` | BPSK generation, pulse shaping and RF transmission |
| `receiver.py` | RF reception and receiver DSP chain |
| `filter.py` | Butterworth and RRC streaming filters |
| `costas_loop.py` | BPSK carrier recovery |
| `gardner_loop.py` | Symbol timing recovery |
| `link_evaluator.py` | Synchronization, delay estimation and BER |
| `observation_point.py` | Electromagnetic field sampling and FFT analysis |
| `server.py` | Simulation runtime, WebSocket backend and telemetry |
| `static/app.js` | Interactive frontend and visualization |
| `static/index.html` | Web interface and controls |

---

# Performance

The FDTD solver performs numerical calculations over the complete simulation grid at every timestep.

NumPy arrays are used to store the electromagnetic field and material-property maps.

The numerical FDTD update kernel is JIT compiled using Numba.

The spatial update is parallelized to improve computational performance.

Material-dependent coefficients are precomputed so that expensive material calculations do not need to be repeated during every FDTD timestep.

---

# Repository Status

The project is currently in an advanced implementation stage for the Signals and Systems Project Course.

The current implementation has moved from the original basic BPSK communication model to an integrated electromagnetic propagation and digital communication simulation.

The currently implemented system includes:

* Heterogeneous electromagnetic FDTD propagation.
* Spatially varying electromagnetic material properties.
* Absorbing computational boundary.
* Multiple transmitter support.
* Multiple receiver support.
* BPSK transmission.
* RRC pulse shaping.
* RF carrier generation.
* RF band-pass filtering.
* Costas carrier recovery.
* RRC matched filtering.
* Gardner symbol timing recovery.
* BPSK hard decisions.
* Known-sequence synchronization.
* BPSK polarity inversion handling.
* BER evaluation.
* Propagation-delay estimation.
* Observation-point analysis.
* FFT visualization.
* Optional Gaussian noise.
* Interactive scenario presets.
* Interactive transmitter and receiver configuration.
* Interactive material configuration.
* Real-time WebSocket telemetry.
* Browser-based interactive visualization.

Features listed under Phase 3 remain planned future work.

---

# Local Installation & Setup

## 1. Clone the Repository

```bash
git clone https://github.com/SSathiyadhev/interactive-2d-wireless-communication-simulation-and-visualization.git
```

## 2. Create a Virtual Environment

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Run the Development Server

```bash
uvicorn server:app --reload --host 127.0.0.1 --port 8000
```

## 5. Access the Workbench

Open a browser and navigate to:

```text
http://127.0.0.1:8000
```

---

# Basic Usage

1. Start the FastAPI server.
2. Open the simulation interface in a browser.
3. Select a scenario preset or configure the simulation manually.
4. Start the simulation.
5. Observe electromagnetic field propagation on the 2D canvas.
6. Select transmitter, receiver, or observation-point nodes.
7. Open the corresponding signal scope.
8. Inspect the individual receiver DSP stages.
9. Observe the FFT spectrum.
10. Adjust simulation parameters when required.
11. Inspect synchronization and BER information from the link evaluator.
12. Use different scenarios to study propagation, material effects, multiple signals, and noise.