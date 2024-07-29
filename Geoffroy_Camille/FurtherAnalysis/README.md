# Get plots on LFP
To be used to analyse individual or groups of latent states to find possible substates.

This notebook utilizes:
- 3 files from the training (obsKeys.npz, inferredStates.npz, uniqueStates.npz)
- 3 files from running the model on a posttrial (obsKeyspt.npz, inferredStatespt.npz, uniqueStatespt.npz)
- 2 raw lfp signals of the posttrial (HPC and PFC)

When running the notebook, the user will be able to select a couple of latent states to view them on the raw signals.
An interactive plot has been implemented for better visualisation.
The latent states should be selected by looking at the boxplots created when running the model.
