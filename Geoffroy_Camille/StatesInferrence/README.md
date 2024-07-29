# Data
These notebooks utilize 6 files.

- Training the model will return 3 files, obsKeys.npz, uniqueStates.npz and latentStates.npz.
- Runnning the model will return the same 3 files, which are known as obsKeyspt.npz, uniqueStatespt.npz and latentStatespt.npz in the code.

# Automatic scoring
'AutomaticScoring.ipynb'
Run the model on a posttrial and upload the 6 files mentionned above (If the manual scores are unknown,use an array of zeros in when running the model). The notebook will use the proportions of different sleep states in all latent states of the training to assign to each of these latent states an inferred sleep state (1,3 or 5). The notebook will then match the latent states from the training to those found in the test run to assign the corresponding sleep state to each epoch from the test run. 
This notebook will return a matrix containing the inferred sleep scoring.

# Inferred states analysis
'Stateinferrence+ConfMat+hypnos.ipynb'
Upload the same 6 files. Manual scoring of the test data must exist and used when running the model.
Running this notebook on the data obtained from a run on a posttrial will create a confusion matrix between inferred sleep states and manual scoing and also print automatic and manual hypnograms.
