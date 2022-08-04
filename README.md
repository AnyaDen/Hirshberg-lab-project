# Data-Driven Collective Variables in Meta Dynamics

MD simulations are very powerful but suffer from time scale problem. Metadynamics and other approaches can overcome it by biasing towards enhanced sampling of rare events.
These approaches rely on identifying CVs (also named reaction coordinates), a small subset of the degrees of freedom that describes the slowest modes of the process.
Finding CVs is still a major challenge in complex chemical systems. In this work, we tested several dimensionality reduction algorithms to identify optimal CVs for a toy model, alanine dipeptide.

This repository includes some scripts and lammps/gromacs setup files, used to run MetaD simulations, as well as extracting PCA, LDA, and HLDA CVs.
