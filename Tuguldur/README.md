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

### Detecting Phasic/Tonic states
Refer to this [repo](https://github.com/8Nero/phasic_tonic) for documentations on detecting phasic and tonic states.

This [tutorial](https://phasic-tonic.readthedocs.io/en/latest/generated/gallery/tutorial_detect_phasic/) covers how to use `detect_phasic` function to get phasic REM indices.
