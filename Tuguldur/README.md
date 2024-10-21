# Phasic/Tonic analysis pipeline
This repository contains the scripts and methods used for phasic/tonic analysis project.

## Overview
- **Dataset Loader**: Loading and iterating through multiple datasets (CBD, RGS14, and OS basic).
- **Phasic/Tonic Detection**: Identify REM sleep substages (phasic and tonic).
- **Statistical Analysis**: Scripts used for analysing sleep composition and phasic/tonic states.

## Guide for the phasic/tonic analysis pipeline
### Setting up the datasets
Using `DatasetLoader` will require few modifications to the original datasets. 
1. **For working with CBD dataset**: Replace `overview.csv` with the [modified version](https://github.com/AbdelRayan/AutomaticSleepScoring/blob/main/Tuguldur/data/overview.csv) in the CBD folder.
2. **For working with OS basic dataset** Use [this](https://github.com/AbdelRayan/AutomaticSleepScoring/blob/main/Tuguldur/data/nameOSbasic.ipynb) script for renaming
the folders into proper form.
3. **For using the DatasetLoader** Make sure to give the path to [this config file](https://github.com/AbdelRayan/AutomaticSleepScoring/blob/main/Tuguldur/data/dataset_loading.yaml) when initializing DatasetLoader.

### Loading Datasets

Initializing `DatasetLoader` requires a path to a configuration file. The configuration file includes paths to datasets and the naming patterns for: posttrial folder, LFP of HPC and PFC, hypnogram files, etc.).

#### Configuration File

```yaml
CBD:
  path: "/path/to/CBD/"
  posttrial: "[\w-]+posttrial[\w-]+"
  hpc: "*HPC*continuous*"
  pfc: "*PFC_100*"
  states: "*-states*"
RGS:
  path: "/path/to/RGS14/"
  posttrial: "[\w-]+post[\w-]+trial[\w-]+"
  hpc: "*HPC*continuous*"
  pfc: "*PFC_100*"
  states: "*-states*"
OS:
  path: "/path/to/OSbasic/"
  posttrial: ".*post_trial.*"
  hpc: "*HPC*"
  pfc: "*PFC*"
  states: "*states*"
```

Change the `path` fields to point to their local dataset directories.
If you're working on a single dataset then you can remove the other fields.

### Using DatasetLoader
Check out this [tutorial](https://github.com/AbdelRayan/AutomaticSleepScoring/blob/main/Tuguldur/notebooks/tutorial_dataset_loader.ipynb) on how to use DatasetLoader to iterate through multiple datasets.

### Naming
The naming scheme used here is: Rat ID, Study Day, Behaviour/Condition, Treatment, Trial number separated by underscores.
For treatment values:
- 0 refers to Negative treatment in CBD
- 1 refers to Positive treatment in CBD
- 2 refers to Negative treatment in RGS
- 3 refers to Positive treatment in RGS
- 4 refers to OS basic
  
For example, "Rat5_SD8_HC_0_posttrial3" would mean recording of a Rat number 5, Study Day 8, Home Cage, Negative treatment and post-trial number 5.

This [helper module](https://github.com/AbdelRayan/AutomaticSleepScoring/blob/main/Tuguldur/pipeline/helper.py) contains `get_metadata` that takes a string like "Rat5_SD8_HC_0_posttrial3" and returns dictionary of metadatas.

### Partitioning
In the datasets, post-trial 1 to 4 recordings are 45 minutes long, whereas post-trial 5 recording is 180 minutes. 
These post-trial 5 recordings can be further partioned into 4 segments, each 45 minutes long. 
In that case, we refer its trial number as 5.1, or 5.4 depending on which segment.

I've used [this script](https://github.com/AbdelRayan/AutomaticSleepScoring/blob/main/Tuguldur/notebooks/old/build_dataset.ipynb)
for partitioning and extracting only REM epochs from the datasets.

`partition_to_4` function in `pipeline.helper` module also segments LFP and Hypnogram arrays into 4 parts. 

**Formatting tables**

We can use this naming scheme, to make tables.

For example:
| rat_id | study_day | condition | treatment | trial_num | state  | duration |
|--------|-----------|-----------|-----------|-----------|--------|----------|
| 2      | 2         | OR        | 0         | 1         | phasic | 0.992    |
| 2      | 4         | OR        | 0         | 2         | phasic | 1.998    |
| 1      | 4         | HC        | 1         | 3         | phasic | 2.048    |
| 2      | 2         | CON       | 4         | 4         | phasic | 1.586    |
| 4      | 2         | HC        | 4         | 5         | tonic  | 36.996   |

### Detecting Phasic/Tonic states
Refer to this [repo](https://github.com/8Nero/phasic_tonic) for documentations on detecting phasic and tonic states.

This [tutorial](https://phasic-tonic.readthedocs.io/en/latest/generated/gallery/tutorial_detect_phasic/) covers how to use `detect_phasic` function to get phasic REM indices.

### Sleep composition analysis notebooks
`sleep_analysis` folder in `notebooks` contain the sleep composition analysis scripts. 

Refer to this [documentation](https://github.com/AbdelRayan/AutomaticSleepScoring/blob/main/Tuguldur/notebooks/sleep_analysis/pipeline_documentation.docx.pdf) for the outputs of the scripts.

