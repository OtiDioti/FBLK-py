# Four Bands Luttinger Kohn

## Overview

This repository contains an implementation of a numerical approach to fully solve the four bands Luttinger Kohn Hamiltonian using the finite element method. The solver is designed to accurately calculate the electronic structure of materials described by the four bands Luttinger Kohn Hamiltonian, providing insights into their physical properties.

## Background

The Luttinger-Kohn model is a widely used theoretical framework in condensed matter physics for describing the electronic structure of semiconductors and other materials. It provides a comprehensive understanding of the behavior of charge carriers in the presence of external potentials and perturbations.

The four bands Luttinger Kohn Hamiltonian allows for a detailed description of the electronic structure in materials with complex band structures, such as semiconductors with multiple valleys or anisotropic band dispersions.

The repository is divided into two separate methods: 

* **Bulk**: allows for the modelling of the full 3-dimensional system. Here it is possible to include the effect of external potential to study the effect of different confinement geometries.
* **Planar Confinement**: Allows for the modelling of a 2-dimensional model of the Luttinger-Kohn hamiltonian. Compared to **Bulk** (including a strong confinement along one of the directions), this requires less computational power to exectute. However, this method is limited to a spherical approximation of the Luttinger Kohn Hamiltonian.

## Features

* **Finite Element Method Solver**: The repository includes an implementation of the finite element method (FEM) to solve the four bands Luttinger Kohn Hamiltonian numerically. FEM is a powerful technique for solving partial differential equations, making it well-suited for accurately capturing the behavior of electrons/holes in materials described by the Hamiltonian.

* **Customizable Parameters**: Users can easily customize parameters such as Luttinger parametes, external potentials, boundary conditions and external fields to study different scenarios and explore the effects of various factors on the electronic structure. The coupling to external electrict and magnetic fields is handled via a semi-classical approach through a minimal coupling substitution method. 

* **Visualization Tools**: The repository includes tools for visualizing the results of the simulations, allowing users to gain insights into the electronic properties of the materials under investigation. Visualization aids in interpreting the simulation results and identifying important features of the ele ctronic structure.

## Usage

To use the solver, follow these steps:
