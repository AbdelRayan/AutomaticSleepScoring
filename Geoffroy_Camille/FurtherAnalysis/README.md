# Get plots on LFP
To be used to analyse individual or groups of latent states to find possible substates.

This notebook utilizes:
- 3 files from the training (obsKeys.npz, inferredStates.npz, uniqueStates.npz)
- 3 files from running the model on a posttrial (obsKeyspt.npz, inferredStatespt.npz, uniqueStatespt.npz)
- 2 raw lfp signals of the posttrial (HPC and PFC)

When running the notebook, the user will be able to select a couple of latent states to view them on the raw signals.
An interactive plot has been implemented for better visualisation.
The latent states should be selected by looking at the boxplots created when running the model.


# Mitigated Latent State Plots
To be used to analyse a mitigated latent state.

This notebook utilizes:
- 3 files from running the model on a posttrial (obsKeys.npz, inferredStates.npz, uniqueStates.npz)
- 2 raw lfp signals of the posttrial (HPC and PFC)
- the hdf5 file from which the data were obtained.

When running the notebook, the user will be able to select a latent states to view it on the raw signals - with different colors depending on the manual score of the epoch.
An interactive plot has been implemented for better visualisation.
The latent state should be selected by looking at the boxplots created when running the model.


# Hypnodensity graphs
To be used to obtain hypnodensity plots along with the manual hypnogram.

This notebook utilizes:
- 3 files from the training (obsKeys.npz, inferredStates.npz, uniqueStates.npz)
- 3 files from running the model on a posttrial (obsKeyspt.npz, inferredStatespt.npz, uniqueStatespt.npz)

Using this notebook will first plot a hypnodensity graph for the full post trial, colouring in different colours the proportions of true Wake, NREM or REM in the current latent state. Zoomed in plots on target regions can then be performed.


# Latent states reduction
In addition to 100 latent states being too much, many latent states contain very few epochs. This notebook explores a way of reducing the number of latent states. 

This notebook utilizes:
- 3 files from the training (obsKeys.npz, inferredStates.npz, uniqueStates.npz) (MAIN)
- 3 files from running the model on a posttrial (obsKeyspt.npz, inferredStatespt.npz, uniqueStatespt.npz) (OPTIONAL)
- 2 raw lfp signals of the posttrial (HPC and PFC) (OPTIONAL)

In this notebook, you will be able to set proportion-based threshold on the minimum size of a latent state per state for it to be considered a large latent state. Small latent states will then be mapped to the larger latent states that share their states, based on euclidean distance minimising (Small REM mapped to large REM, Small NREM mapped to large NREM....etc).
You can then generate boxplot comparing the values of each features between the groups of latent states. (multiple plots possible)

To go further:
You can upload the optional data mentionned above to plot these groups of latent states on a post trial's raw data with zooms. (Color coded: Greys=Wake, Blues=NREM, Reds=REM).

# Latent states clustering
This notebook explores another way of reducing the number of latent states. 

This notebook utilizes:
- 3 files from running the model on a temporally ordered dataset (obsKeys.npz, inferredStates.npz, uniqueStates.npz),
- the temporally ordered dataset fed to the model,
- an array with the length of each posttrial.

In this notebook, you will be able to cluster the latent states based on the epochs they are most often close to (time-wise).
You can vizualise (before and after clusterization) the state transition graphs, the histogram of the number of epoch per latent state, confidence matrices,  as well as hypnograms for a specific posttrial.

# State transition
This notebook displays the latent state transition graph. Bear in mind that state transitions only make sense if the epochs are ordered with regards to time.

Thus notebook utilizes:
- 3 files from running the model on a temporally ordered dataset (obsKeys.npz, inferredStates.npz, uniqueStates.npz),
- an array with the length of each posttrial.
