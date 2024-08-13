# Introduction
This folder allows us to collect features and manual scoring to feed the mcRBM algorithm. 

It is divided in 2 steps :
- step 1 : Creation of the hdf5 file;
- step 2 : Extraction of a dataset or a training dataset from the hdf5 file.

# File structure
## Pre Processing
The raw data (HPC and PFC) first go through `Artefacts_Detection.py`. This function detects the artefacts and sets them to zero.
Epochs where artefacts are detected will have their manual score set to 0.

Then we use `EMG_buzsakiMethod.py` - this function allows to get an EMG-like array from the HPC and PFC signal.

## Features
To obtain the desired features, we first compute the mean spectral power of the following frequency ranges on 2-second epochs:
- $`EEG_{lo}`$ : [0; 0.5] Hz;
- $`\delta`$ : [0.5; 5] Hz;
- $`\theta`$ : [6; 10] Hz;
- $`\sigma`$ : [11; 17] Hz;
- $`\beta`$ : [22; 30] Hz;
- $`\gamma`$ : [35; 45] Hz;
  
as well as:
- $`EMG`$ : mean value of the EMG.

From these values, we compute the following features for each 2-second epoch:
 - Index_W: $\frac{E M G^2 \cdot \gamma}{\theta}$
 - Index_N : $\frac{\sigma \cdot \delta}{\gamma^2}$
 - Index_R : $`\frac{\theta^2 \cdot \gamma}{\delta^2 \cdot E M G^2}`$
 - Index_1 : $\frac{E M G \cdot \gamma}{\delta}$
 - Index_2 : $\frac{\sigma \cdot \delta}{\theta}$
 - Index_3 : $\frac{\theta \cdot \gamma}{\delta}$
 - Index_4 : $\frac{\delta}{\theta}$
 - Theta
 - Delta
 - $`EEG_{lo}`$ (Noise)

## hdf5 file structure
The existing hdf5 file (`OS_RGS_CBD.h5`) has a tree-like structure. The first groups consist of the different datasets. The subgroups consist of the different rats. 
Finally, the subsubgroups consist of the different posttrials. In each posttrial group we have two datasets :
- the `Features` dataset which contains arrays of the 10 features for each 2-second epoch of the recording;
- the `Mapped_scores` dataset which consists of the corresponding manual score for each 2-second epoch.

## Dataset names
For each posttrial, we used the following naming convention :
- OS_Basic : `Rat<i>_SD<j>_<condition>_posttrial<k>` with:
    - i : the rat number;
    - j : the study day number;
    - condition : the condition of the posttrial;
    - k : the posttrial number.
      
 - RGS14 : `Rat<i>_SD<j>_<condition>_<treatment>_posttrial<k>` with:
     - i, j, k - as described above;
     - the treatment of the rat : 
       - 2 : Control;
       - 3 : RGS14 treated.
     
 - CBD : `Rat<i>_SD<j>_<condition>_<treatment>_posttrial<k>` with:
     - i, j, k - as described above;
     - the treatment of the day : 
       - 0 : not treated;
       - 1 : treated.
      

# hdf5 file creation
To create an hdf5 file, we recommend that you first create the hdf5 file for each rat in the dataset, and then merge the different hdf5 files you get.
To create an hdf5, you need to add the path (`path_to_pt5`) to the folder containing the posttrial files (HPC, PFC, states).

The various files must be named as follows:
- `<name of the posttrial>_HPC`;
- `<name of the posttrial>_PFC`;
- `<name of the posttrial>_states`.

You also need to add the path (`hdf5_path`) to the new hdf5 file you want to create. Be careful, as this will overwrite on a potential already-existing file.
Finally, don't forget to change the sampling frequency (`fs`) and the epoch length (`epoch_length`) if needed.

This function uses parallel execution with **joblib** to be faster. 
Therefore, this function must be run directly on your computer (if you have enough logical processors).

# Extracting a dataset
To extract a dataset, you can use one of the functions in the `dataset_extraction.py` file. 
The first function : `training_dataset_OS_RGS_CBD` allows you to create a training dataset consisting of posttrials from the rats of the different datasets (OS_Basic, RGS and CBD).
You can choose which rats from which datasets to include, as well as the number of epochs from each manually scored state (Artefact, Wake, NREM, TS and REM) and the number of repetitions of each epoch for each state.
The epochs are randomly selected from the different posttrials included in the dataset.

The second function : `dataset_OS_RGS_CBD` allows you to create a dataset on which to run a model.
You can choose which rats to include in the dataset and which manually scored states (Artefact, Wake, NREM, TS and REM).
The resulting dataset is ordered by state and then by posttrial.

The third function `dataset_in_order_OS_RGS_CBD` also allows you to create a dataset on which to run a model. 
The only difference is that the resulting dataset is ordered in time by posttrial.
Therefore, if you don't want all states in the dataset, it will take longer to compute.

The last function `posttrial_OS_RGS_CBD` allows you to create a dataset consisting of only one posttrial of a rat from OS_Basic, RGS14 or CBD.
You can select the posttrial as well as the states.
The result will be ordered in time.

# NewFeatures
This is an analysis notebook that is not required for running the pipeline. Running it will show how the spectral powers are computed per frequency band, how they correlate with different sleep states in the manual scoring and how one can infer formulas as they did in the Wei 2019 biomed eng paper to create features that discriminate best between the 3 main sleep states. The 'new' features are the ones we created, inspired by Wei. 

This notebook will compare the new features to Wei's and show plots of their performance.
ps. computing the powers is not instantaneous. Expect about 2 min per frequency band for a 3h signal.
