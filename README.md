# Four Bands Luttinger Kohn

## Overview

This repository contains an implementation of a numerical approach to fully solve the four bands Luttinger Kohn Hamiltonian using the finite element method. The solver is designed to accurately calculate the electronic structure of materials described by the four bands Luttinger Kohn Hamiltonian, providing insights into their physical properties.

## Background

The Luttinger-Kohn model is a widely used theoretical framework in condensed matter physics for describing the electronic structure of semiconductors and other materials. It provides a comprehensive understanding of the behavior of charge carriers in the presence of external potentials and perturbations.

The four bands Luttinger Kohn Hamiltonian allows for a detailed description of the electronic structure in materials with complex band structures, such as semiconductors with multiple valleys or anisotropic band dispersions.

## Features
![Alt text](/Images/ground_state_planar_geomety.png)

* **Finite Element Method Solver**: The repository includes an implementation of the finite element method (FEM) to solve the four bands Luttinger Kohn Hamiltonian numerically. FEM is a powerful technique for solving partial differential equations, making it well-suited for accurately capturing the behavior of electrons/holes in materials described by the Hamiltonian.

* **Customizable Parameters**: Users can easily customize parameters such as Luttinger parametes, external potentials, boundary conditions and external fields to study different scenarios and explore the effects of various factors on the electronic structure. The coupling to external electrict and magnetic fields is handled via a semi-classical approach through a minimal coupling substitution method. 

* **Visualization Tools**: The repository includes tools for visualizing the results of the simulations, allowing users to gain insights into the electronic properties of the materials under investigation. Visualization aids in interpreting the simulation results and identifying important features of the ele ctronic structure.
* **GUI**: The repository includes an intuitive GUI that will allow you to easily pick a confinement potential, as well as the strength of the external magnetic field, Luttinger parameters, and problem size. 
## Usage

To use the solver, open the terminal and follow these steps:

* **Move to desired directory** (in this case Desktop):
    ```
    cd Desktop/
    ```
* **Clone this repository**:
  
  ```
  git clone https://github.com/OtiDioti/FBLK-py.git
  ```
* **Move inside folder**:
    ```
    cd FBLK-py
    ```
* **Create new conda enviroment**:
    ```
    conda create -n FBLK python==3.9
    ```
* **Install requirements**:
    ```
    conda activate FBLK
    ```
    ```
    pip install -r requirements.txt
    ```
* **Run program**:
    ```
    streamlit run Home.py
    ```
  


