# TEAM 35 Solenoid Optimization with Model Order Reduction

This repository contains Python implementations for solving the TEAM 35 benchmark problem using projection-based model order reduction (MOR). The optimization targets the magnetic field in an air-cored solenoid by varying either turn currents or turn radii, or both. 

## Project Overview

The TEAM 35 benchmark problem defines a multi-turn coil with an air core. The goal is to optimize the geometry and/or excitation to achieve a homogeneous and robust magnetic field distribution in a specific control region. The problem is addressed using:

- **Full Order Finite Element Models (FOM)**
- **Projection-Based Reduced Order Models (ROM)** using a Cauer Ladder Network (CLN)-based method
- **Multi-objective optimization** using NSGA-II (including bi-objective and tri-objective formulations)

The MOR approach leads to speedups of more than an order of magnitude compared to FOM.

## Optimization Scripts

The following scripts can be executed independently:

| File name                       | Description |
|--------------------------------|-------------|
| `opti_radius.py`               | Two-objective **radius-only** optimization |
| `opti_current.py`              | Two-objective **current-only** optimization |
| `opti_current_triobjective.py`| Three-objective **current-only** optimization (adds Joule loss) |
| `opti_combined.py`            | Two-objective **combined current & radius** optimization |

Each script uses a ROM generated from the FEM model and runs a multi-objective evolutionary algorithm (NSGA-II) to generate Pareto-optimal solutions.

## Parameters

Below is the list of key parameters used in the TEAM 35 solenoid optimization problem: 

| Parameter              | Inital Value | Description                                  |
|------------------------|--------------|----------------------------------------------|
| `n_turns`              | `10`         | Number of coil turns                         |
| `insulation_thickness` | `0.00006` m  | Insulation thickness between turns (60 µm)   |
| `I`                    | `3.18` A     | Initial excitation current per turn          |
| `solw`                 | `0.005` m    | Total width of a turn                        |
| `solh`                 | `0.005` m    | Total height of a turn                       |
| `coordinates`          |              | Grid of control points (11×6)                |
| `maxh`                 | `0.0002` m   | Maximum mesh element size                    |
| `controlledmaxh`       | `0.0002` m   | Maximum element size in control region       |
| `B_target`             | `2e-3` T     | Target magnetic flux density                 |
| `w`                    | `0.001` m    | Width of conductor cross-section             |
| `wstep`                | `0.0005` m   | Width of conductor section cross-section     |
| `h`                    | `0.0015` m   | Height of conductor cross-section            |
| `sigma_coil`           | `4e7` S/m    | Electrical conductivity of the coil (copper) |
| `mincurrent`           | `0` A        | Minimum current per turn (lower bound)       |
| `maxcurrent`           | `15` A       | Maximum current per turn (upper bound)       |
| `radius`               | `0.01` m     | Default radius of the coil                   |
| `minradius`            | `0.006` m    | Minimum radius of the coil                   |
| `maxradius`            | `0.030` m    | Maximum radius of the coil                   |


## Dependencies

Make sure to install the required packages before running the code:

numpy==1.26.4

pandas==2.2.1

matplotlib==3.8.3

pymoo==0.6.1.3

scikit-learn==1.6.1

ngsolve==6.2.2404.post51

netgen-mesher==6.2.2404.post16

## Publication
The methods implemented in this repository are documented in the article:

Numerical modelling of a Solenoid Optimization Problem with a Superposition-based Model Order Reduction method

Kristóf Levente Kiss, Tamás Orosz

Submitted to Computers & Structures, Elsevier, 2024

