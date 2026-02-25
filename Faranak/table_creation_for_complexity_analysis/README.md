# Rodent Sleep Feature Table Builder for Complexity Analysis

This repository provides a pipeline to:

1) **Index and load datasets** (per posttrial / rat / condition) from `.mat` files  
2) **Parse metadata** from the dataset 
3) **Build a long-format table** that contains:
   - Metadata columns
   - Sleep state labels
   - Complexity features (aperiodic exponent, DFA, MSE)

## Repository Structure: 
.
├── dataset_loader/     \
│ ├── init.py       \
│ ├── loader.py # Scans dataset directory and builds a dataset map     \
│ ├── helper.py # Extracts metadata from dataset name / filename convention   \
│ └── utils.py # Utility functions (windowing, alignment, etc.)   \
│     \
├── configs/    \
│ ├── config_RGS.yaml      \
│ ├── config_OS.yaml      \
│ └── config_CBD.yaml     \
│     \
├── main_loader_RGS.ipynb    \
├── main_loader_OS.ipynb     \
├── main_loader_CBD.ipynb     \
│      \
└── README.md       \

> Notes  
> - The **`dataset_loader/` folder and the notebook(s)** should remain in the **same repository** (same project root).  
> - Each dataset has its own **config YAML** and **main notebook**, but the usage pattern is the same.

---

## Quick Start
### 1. Set the dataset path in the config file
Each dataset has a config file (YAML). Open the one you want (e.g. configs/config_RGS.yaml) and set the dataset root path:
```
Path: "D:/Path/To/Your/Dataset"
```
