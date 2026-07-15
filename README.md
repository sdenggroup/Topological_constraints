# TopoConstraints_SHG

This repository contains the numerical codes used in the paper:

**Topological constraints on second-harmonic generation in essentially noncentric insulators: A two-band model study**

This project studies the relationship between band topology, the shift vector, and second-harmonic generation (SHG) in essentially noncentric insulators. In the paper, the SHG response under linearly polarized light is derived for a two-dimensional two-band model based on the Floquet formalism and the Keldysh Green's function method, and the results are compared with those obtained from Sipe's formalism.

The programs in this repository are mainly used to:

- calculate the SHG response in a two-dimensional two-band model;
- compare the SHG results obtained from Sipe's formalism and the method developed in this work;
- analyze the dependence of the SHG response on the model parameters alpha and beta;
- analyze the T1a term related to the shift vector;
- generate one-dimensional line plots, two-dimensional parameter maps, histograms, and kernel density estimation curves.

## Repository Structure

### `Sipe1D`

This folder contains the programs for calculating the SHG response using Sipe's formalism.

The programs calculate the SHG response as a function of the parameter alpha with a fixed value of beta.

### `Sipe2D`

This folder contains the programs for calculating the SHG response using Sipe's formalism.

The programs calculate the SHG response as a function of both alpha and beta.

### `Thiswork1D`

This folder contains the programs for calculating the SHG response using the method developed in this work.

The programs calculate the SHG response as a function of the parameter alpha with a fixed value of beta.

### `Thiswork2D`

This folder contains the programs for calculating the SHG response using the method developed in this work.

The programs calculate the SHG response as a function of both alpha and beta.

### `T1a`

This folder contains the programs and results for analyzing the T1a term.

The results include histograms and kernel density estimation curves for studying the distribution features of the T1a term.

### `T1a1T1a2`

This folder contains the programs and results for separately analyzing the T1a1 and T1a2 terms.

The results include histograms and kernel density estimation curves for the T1a1 and T1a2 terms.

## Requirements

The programs are written in Python and mainly depend on the following third-party packages:

- NumPy
- Matplotlib
- mpi4py

Some programs use the following packages when saving data files:

- pandas
- openpyxl

## Citation

If you use the codes in this repository, please cite the related paper:

Topological constraints on second-harmonic generation in essentially noncentric insulators: A two-band model study, 
Yaomin Ren, Xiyue Cheng,* Hanxiang Mi, and Shuiquan Deng, Phys. Rev. B, DOI:https://doi.org/10.1103/ptdl-6q13
