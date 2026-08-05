# Interactive 2D Wireless Communication Simulation and Visualization

A modular 2D wireless communication simulation platform for electromagnetic wave propagation, BPSK communication, digital signal processing, and real-time visualization.


## Why this repository exists?

This repository contains the implementation of a custom project developed as part of a Signals And Systems Project Course.

The objective of this project is to develop a modular two-dimensional wireless communication simulator capable of modelling electromagnetic wave propagation, BPSK-based communication, and the influence of the physical environment on wireless signal propagation. The simulator follows a modular object-oriented architecture, allowing each major component to be developed and extended independently.

---

# Objectives

The project aims to:

* Simulate BPSK-based wireless communication.
* Simulate electromagnetic wave propagation in a two-dimensional environment.
* Visualize transmitted, propagated, and received signals.
* Analyze communication performance using FFT, Signal-to-Noise Ratio (SNR), and Bit Error Rate (BER).
* Study the effect of distance, noise, multiple transmitters, and physical obstacles on wireless communication.

---

# Technology Stack

* Python

Additional libraries will be introduced as the project develops.

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

# Software Architecture

For Phase 1, The simulator is organized into four primary components.

## SimulationSpace

Represents the simulation environment.

Responsibilities include:

* Maintaining the simulation space
* Maintaining simulation time and time step
* Storing the electromagnetic field at every simulation point
* Providing public APIs for reading and updating field values

---

## Transmitter

Represents a wireless signal source.

Responsibilities include:

* Generating the carrier signal
* Performing BPSK modulation
* Injecting the transmitted signal into the simulation space

---

## WaveSolver

Represents the propagation engine.

Responsibilities include:

* Computing electromagnetic wave propagation
* Applying free-space path loss
* Applying global AWGN noise
* Combining signals from multiple transmitters
* Updating the electromagnetic field throughout the simulation space

---

## Receiver

Represents a wireless receiver.

Responsibilities include:

* Reading the electromagnetic field
* Performing real-time BPSK demodulation
* Recovering transmitted bits
* Computing FFT
* Computing Signal-to-Noise Ratio (SNR)
* Computing Bit Error Rate (BER)

---

# Simulation Flow

```text
Initialize SimulationSpace

Create Transmitters

Create Receivers

Create WaveSolver

while simulation is running

    Advance simulation time

    All transmitters inject source values

    WaveSolver propagates the electromagnetic field

    All receivers process the received signal

    Update visualization
```

---

# Repository Status

The project is currently under active development.

The primary implementation target is the completion of the Phase 1 and Phase 2 features defined for the course project. Features listed under Phase 3 are planned as future enhancements.
