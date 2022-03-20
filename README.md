# **Modeling the mind of a predator: Interactive cognitive maps support flexible avoidance of dynamic threats**
### Toby Wise, Caroline J Charpentier, Peter Dayan & Dean Mobbs

----

This repository contains all code necessary to reproduce the analyses from our work on flexible avoidance of dynamic threats.

## Notebooks

The `/notebooks` directory contains Jupyter notebooks that run through all the main analyses, including code to produce the figures in the paper. Some model fitting is run in separate scripts as it makes it easier to run multiple fits in parallel. This is described in more detail in the relevant notebooks.

## Code

The code for much of the model fitting is provided in the `/code` directory. This provides high-level scripts to run the fitting procedures, the use of which is described in the relevant notebooks.

## Dependencies

Aside from common Python dependencies for data manipulation and processing (e.g., Numpy, Pandas), this code relies upon the Multi-Agent MDP package (https://github.com/tobywise/multi_agent_mdp).

In addition, the hypothesis testing invers reinforcement learning model depends on [Jax](https://github.com/google/jax) and [Numpyro](https://github.com/pyro-ppl/numpyro), which do not work on Windows.