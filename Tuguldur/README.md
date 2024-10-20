# Phasic/Tonic analysis pipeline
This repository contains the scripts and methods used for phasic/tonic analysis project.

## Overview
- **Dataset Loader**: Loading and iterating through multiple datasets (CBD, RGS14, and OS basic).
- **Phasic/Tonic Detection**: Identify REM sleep substages (phasic and tonic).
- **Statistical Analysis**: Scripts used for analysing sleep composition and phasic/tonic states.

# Instructions
## Setting up the datasets
Using `DatasetLoader` will require few modifications to the original datasets. 
1. **For working with CBD dataset**: Replace `overview.csv` with the [modified version](https://github.com/AbdelRayan/AutomaticSleepScoring/blob/main/Tuguldur/data/overview.csv)
2. **For working with OS basic dataset** Use [this](https://github.com/AbdelRayan/AutomaticSleepScoring/blob/main/Tuguldur/data/nameOSbasic.ipynb) script for renaming
the folders into proper form.

### Loading Datasets

The `DatasetLoader` class is tailored for datasets from our lab. It requires a path to a configuration file, which defines the structure of the datasets. The configuration file includes paths to different recordings and the naming patterns for various signals (e.g., LFP of HPC and PFC, hypnogram files, etc.).

#### Example Configuration File

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

The user must modify the `path` fields to point to their local dataset directories.

#### Usage of `DatasetLoader`

Once the configuration file is set up, the `DatasetLoader` can be used to load the datasets and iterate through the recordings. It functions similarly to PyTorch’s `DataFolder`:

```python
from phasic_tonic import DatasetLoader

# Initialize the loader with the path to your config file
loader = DatasetLoader(config_path='path/to/your/config.yaml')

# Loop through the dataset recordings
for recording in loader:
    # Access data (LFP, hypnogram, etc.)
    lfp_hpc = recording['LFP_HPC']
    lfp_pfc = recording['LFP_PFC']
    hypnogram = recording['hypnogram']
    
    # Apply your analysis or further processing here
```
