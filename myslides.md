---
author: Josh Mitchell
title: Enhanced Sampling
subtitle: In Molecular Dynamics

# slideNumber: "c"
margin: 0.1

revealjs-url: reveal.js
---
# Minute of diversity

----

![](figs/pride_flags/lgbtqi_progress.jpg){width="40%"}

![](figs/pride_flags/bi.png){width="40%"}
![](figs/pride_flags/pan.png){width="40%"}

# Basic MD is slow

## Goals of an MD simulation

- Get an equilibrium ensemble of states of a system
- ... so that we can calculate some equilibrium property
  - Rate
  - Free energy change
  - Average macroscopic observable
  - Lowest energy structure
- Non-equilibrium properties much harder

## Biomolecules conspire against us

- Many interesting states with rare transitions
  - solvation changes
  - side chain rotation

# The Free Energy Landscape

## Free Energy is probability

$$
    \epsilon_{\mathrm{rel}}(x) = - K_B T \log(P(x))
$$

- Energy is just maths applied to the probability
- Energy barriers are just transition regions of low probability
- If we know the FES along an interesting variable, we can calculate the average value by integrating over it

----

![](figs/cln025_prob_fes.svg){width=1500px}

----

![So if we have an ensemble...](figs/cln025_fes2d.svg)

# Get that ensemble with slow sampling

