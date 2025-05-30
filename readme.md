<a id="readme-top"></a>

<h3 align="center">Cooperative Coding in Spiking Neural Networks</h3>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a>
    <li><a href="#Prerequisites and Installation">Prerequisites and Installation</a></li>
    <li><a href="#Content">Content</a></li>
    <li><a href="#Usage">Usage</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>


<!-- ABOUT THE PROJECT -->
## About The Project

This repo contains the code used to simulate the spiking networks presented in the manuscript "Cooperative coding of continuous variables in networks with sparsity constraint" by Paul Züge, Natalie Schieferstein, and Raoul-Martin Memmesheimer.


<!-- GETTING STARTED -->

## Prerequisites and Installation

1. Install uv [(https://docs.astral.sh/uv/getting-started/installation/)](https://docs.astral.sh/uv/getting-started/installation/)
2. Clone the repo
   ```sh
   git clone https://github.com/NatalieSchieferstein/CoopCodeSpiking.git 
   ```
3. Go to the repo folder.
4. Install all required packages by running
   ```sh
   uv sync
   ```
Alternatively, check pyproject.toml for a list of dependencies.

<!-- USAGE EXAMPLES -->
## Content 

`main.ipynb`: resimulation of networks shown in main text (Figs 8 & 10); demonstration of analytical tuning <br>
`supplement_poisson.ipynb`: resimulation of Fig H in S1 Appendix (Poisson input in small ER-network) <br>
`supplement.ipynb`: resimulation of Fig I in S1 Appendix ("homogeneous network" with tuned input to interneurons)  <br>
`methods.py`: contains all functions to create, simulate, and analyze the spiking networks  <br>
`tools.py`: auxiliary functions

## Usage

To get started, simply open `main.ipynb` where you will be guided step-by-step on how to
* inspect parameter sets 
* resimulate networks optimized for a certain receptive field size 
* tune networks analytically for a certain receptive field size 

A similar guide through the simulations of Appendix S1 of the paper can be found in `supplement_poisson.ipynb` and `supplement.ipynb`.

<!-- CONTACT -->
## Contact

Natalie Schieferstein - natalie.schieferstein@freenet.de

Project Link: [https://github.com/NatalieSchieferstein/CoopCodeSpiking](https://github.com/NatalieSchieferstein/CoopCodeSpiking)

## Acknowledgements 

Thanks to https://github.com/othneildrew/Best-README-Template for the nice README template!
<p align="right">(<a href="#readme-top">back to top</a>)</p>
