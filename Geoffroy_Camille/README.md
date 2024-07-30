This folder contains code for the automatic sleep scoring mcRBM algorithm.

Using 2 raw lfp signals from the hippocampus and prefrontal cortex (resp. HPC and PFC), this pipeline will train a model and run it to obtain automatic sleep scoring of a sleep period. Sleep will be calssified in 3 states, Wake, NREM and REM.

In order, the repositories used should be:

1- hdf5_files

2- mcRBM

3- StateInferrence

4- FurtherAnalysis

A trained model on untreated rats from 2 datasets (OS basic and RGS14) is available to use in the mcRBM section if the user only wishes to do automatic sleep scoring.
This repository can also be used to train a new model and look for new latent states.
