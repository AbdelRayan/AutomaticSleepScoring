# Inferring states

Training the model will return 3 files, obsKeys.npz, uniqueStates.npz and latentStates.npz.
Runnning the model will return the same 3 files, which are known as obsKeyspt.npz, uniqueStatespt.npz and latentStatespt.npz in the code.

Uploading these 6 files and running the notebook will use the proportions of different sleep states in all latent states of the training to assign to each latent state found in the training an inferred sleep state (1,3 or 5).
The notebook will then match the latent states from the training to those found in the test run to assign the corresponding sleep state to each epoch from the test run.

Running this notebook on the data obtained from a run on a posttrial will create a confusion matrix between inferred sleep states and manual scoing and also print automatic and manual hypnograms.
