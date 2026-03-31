<p align="center">
  <img src="./nara_logo.png" width="50%" title="NARA" alt="NARA"/>
</p>

[![DOI](https://img.shields.io/badge/DOI-10.20517%2Faiagent.2025.13-blue)](https://dx.doi.org/10.20517/aiagent.2025.13)


# NARA: Nature-inspired Algorithm for Robust Atomic structure search

***NARA*** is an atomic structure exploration framework based on the Firefly algorithm. The name ***NARA*** originates from the Korean expression nal-a (날-아), which represents the flying motion of a firefly. It has been developed as a companion package to ***LLUMYS***, a machine learning interatomic potential that provides the "light" of the firefly.

## Prerequisites

- llumys
- Python ≥ 3.10
- ASE ≥ 3.26.0

> Earlier versions are compatible, but 3.26.0 or later is recommended because optimizer.irun now explicitly includes gradient evaluation/log
> 
- scikit-learn
- scipy
- spglib
- e3nn
- pytorch ≥ 2.4

> Please follow the installation guide on the official [PyTorch website](https://pytorch.org/get-started/locally/)
> 
- (optional) openequivariance, GCC≥9

> For additional GPU acceleration. Requires installation of the matching CUDA toolkit for your PyTorch and CUDA driver versions.
> 
