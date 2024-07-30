# Sleep scoring model

## Prerequisite
A CUDA-enabled GPU machine. This means having NVCC toolbox installed on an NVIDIA GPU.

## Setup

1. Run `setup.sh` to install neccessary python packages.
(It is possible that you get an error, but the next steps still work as long as you manage to run `cudamat_example.ipynb`).
2. Run `cudamat_example.ipynb` to verify cudamat installation, cudamat provides an interface to perform matrix calculations on CUDA-enabled GPUs.

## Preprocess data
Add feature file to `sample_data/input/` dir. The feature file should be of the following format.

### Bands
NxM data array, where N is the number of epochs, and columns refer to Delta PFC, Theta HPC etc.

### EpochsLinked
Nx4 data array, where N is the number of epochs, and columns are described as follows:
- column 1: epoch ID
- column 2: epoch index (currently not used)
- column 3: ground truth sleep stage ID, where
  - 0 is associated with artefacts,
  - 1 is associated with wakefulness,
  - 3 is associated with NREM sleep,
  - 4 is associated with TS (intermediate) sleep,
  - 5 is associated with REM sleep
- column 4: the subject ID (used in multi-subject analysis only)

### EpochTime
Nx3 data array, where N is the number of epochs, and columns are described as follows:
- column 1: epoch ID
- column 2: recording mode (i.e. baseline or recovery), where
  - 1 is associated with baseline,
  - 2 is associated with recovery (after sleep deprivation)
- column 3: the epoch date-time

## Converting your file in the right format
If your feature file is not in the above format, use `mcRBM_input_features.ipynb` in `sample_data/` dir to generate the feature file (`.npz`) needed as input for the model.

To do that, you can modify 5 variables :
- **data_path** : the file path which will usually be : 
`/teamspace/studios/this_studio/mouse-sleep-analysis/sample_data/input`
- **feature_npy_file_name** : the name of your npy file containing the features for each epochs
- **feature_npz_file_name** : the name of the npz file you want to obtain (usually same name as the npy file)
- **states_file_name** : the name of your mat file containing the manual scoring for each epoch
  

## Configuration
For the next steps, we differenciate two cases : 
1. Training Dataset : if you want to create a model trained on your dataset
2. Test Dataset : if you want to run an existing model on your dataset

### Training Dataset
Once you have the feature file (`.npz`) ready, update the experiment details in `configuration_files/exp_details` file. 

1. Set `dsetDir` to the absolute path of `input` dir 
(usually : `/teamspace/studios/this_studio/mouse-sleep-analysis/sample_data/input/`).
2. Set `dSetName` to the name of the feature file.
3. Set `statesFile` to the name of the manual scoring file.
4. Set `expsDir` to the absolute path of output or analysis dir where model weights, inference analysis and plots will be stored.
(usually : `/teamspace/studios/this_studio/mouse-sleep-analysis/sample_data/experiments/`)
5. Set `expID` to the unique name of your experiment

There are some parameters and flags that you can set that are useful for training the model and they are described in the `configuration_files/exp_details` file.

You can also tune parameters described in the `configuration_files/input_configuration` file that are also useful for training the model.
### Test Dataset
For the test dataset, you can do the same steps described in the training dataset configuration part, but you have to do these updates in the `exp_details_test` file instead of the `exp_details` file. 
In addition, you have to do 2 more updates : 

6. Set `modelDirName` to the name of the directory containing your model.
7. Set `modelName` to the name of your model.


# Training, running and inference

Now, the model can be trained following which we can do inference analysis to get latent states and classification of the states as well as running the model on other datasets. You may need to modify the file paths if you are using the code locally on your machine.

## Training
### Step 1: Train the model
Make sure that `expFile`is equal to `exp_details` in `train_model.ipynb`.
Run `train_model.ipynb` on GPU.
You can now switch back to CPU.
### Step 2: Latent state inference
Make sure that `expFile`is equal to `exp_details` in `infer_states.ipynb`.
Run `infer_states.ipynb`
### Step 3: Latent state analysis
If you only have epochs of Wake, NREM and REM in your dataset use in the first block of code the following line : 
`from latent_analysis import StatesAnalysis`.
If you have in addition TS (intermediate) epochs, use the line : 
`from latent_analysis_with_TS import StatesAnalysis`
If you also have artefact epochs in addition of the previous ones, use the line : 
`from latent_analysis_with_TS_art import StatesAnalysis`

Make sure that `expFile`is equal to `exp_details` in `latent_states_analysis.ipynb`.
Run `latent_states_analysis.ipynb`

## Running
### Step 1: Latent state inference
Make sure that `expFile`is equal to `exp_details_test` in `infer_states_test.ipynb`.
Run `infer_states_test.ipynb`
### Step 2: Latent state analysis
If you only have epochs of Wake, NREM and REM in your dataset use in the first block of code the following line : 
`from latent_analysis import StatesAnalysis`.
If you have in addition TS (intermediate) epochs, use the line : 
`from latent_analysis_with_TS import StatesAnalysis`
If you also have artefact epochs in addition of the previous ones, use the line : 
`from latent_analysis_with_TS_art import StatesAnalysis`

Make sure that `expFile`is equal to `exp_details_test` in `latent_states_analysis.ipynb`.
Run `latent_states_analysis.ipynb`

# Results and following steps
You can acces the results in your experiment folder (sample_data/experiments/...).

You can modelize the transition state network by running `state_transition.ipynb` in colab (as it needs to install an old Python version).
For this, you need to download `latentStates.npz`, `obsKeys.npz` and `uniqueStates.npz`. You will find them in the analysis/epoch9999 folder of your experiment folder.

`latentStates.npz` consists of:
- `probabilities` : numpy.darray of the hidden-activation probabilities of each epochs;
- `binary` : numpy.darray of the binary hidden units. For each epoch i, we have binary[i] = probabilities[i] > 0.5;
- `inferredStates` : numpy.darray containing the id of the corresponding latent state and the manual score for each epoch;
- `uniqueStates` : numpy.darray containing for each latent state:
    - its id (/!\ from 1 to len(uniqueStates));
    - the number of epochs;
    - the associated binary code.
      
`uniqueStates.npz` consists of:
- `probabilities` : numpy.ndarray - for each latent state : the hidden-activation probabilities associated with the first epoch linked to this latent state;
- `uniqueStates` : numpy.darray containing for each latent state:
    - its id  (/!\ from 0 to len(uniqueStates) - 1);
    - the number of epochs;
    - the associated binary code.

`obsKeys.npz` consists of:
- `obsKeys` - numpy.ndarray - for each epoch :
    - its id;
    - its corresponding latent state;
    - its id;
    - its manual scoring;
    - 1\.


# Tips 

1. You might be confused because there are two different types of epochs in this code. The first one (that we used in this `README.md` file) is the one that is used to describe a period of recording. The second one will appear in the output of train_model, it corresponds to the number of iteration of the mcRBM algorithm.




