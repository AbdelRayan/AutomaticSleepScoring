import os
import re
import logging
import pandas as pd
from pathlib import Path
from .helper import load_config

logger = logging.getLogger("runtime")

class DatasetLoader:
    def __init__(self, CONFIG_DIR):
        """
        Initialize the DatasetLoader with dataset arguments and configuration directory.

        Args:
            CONFIG_DIR: Path to the YAML configuration file.
            {'dataset_name' : {'path' : '/path/to/dataset', ...}
        """
        self.config = load_config(CONFIG_DIR)
        self.combined_mapped = {}

        # name_func(HPC_filename) ->
        cbd_wrapper = decorate_cbd(cbd_name_func=create_name_cbd, CBD_DIR=self.config['CBD']['path'])
        self.naming_functions = {"CBD": cbd_wrapper, "RGS": create_name_rgs, "OS": create_name_os}

    def load_datasets(self):
        """
        Load datasets.
        
        Args:
            dataset_args: Dictionary containing dataset arguments.
        Returns: 
            Combined mapping of dataset files.
            {name : (sleep_states_fname, hpc_fname, pfc_fname), 
            name : (sleep_states_fname, hpc_fname, pfc_fname), 
            ...}
        """
        for name, info in self.config.items():
            logger.debug(f"STARTED: Loading the dataset {name}.")
            dataset_dir = info['path']
            name_func = self.naming_functions[name]

            for root, dirs, _ in os.walk(dataset_dir):
                mapped = process_directory(root, dirs, info, name_func)
                self.combined_mapped.update(mapped)

            logger.debug(f"FINISHED: Loading the dataset {name}.")
            logger.debug(f"Number of files {len(self.combined_mapped)}.")
        
        return self.combined_mapped

    def __getitem__(self, key):
        return self.combined_mapped[key]

    def __iter__(self):
        return iter(self.combined_mapped)

    def __len__(self):
        return len(self.combined_mapped)

    def __str__(self):
        return f"DatasetLoader contains: {len(self.patterns)} datasets. Total loaded recordings: {len(self.combined_mapped)}"


def process_directory(root, dirs, patterns, name_func):
    """
    Process a directory to map sleep states and HPC files using specified patterns and naming function.
    
    Args:
        root: Root directory path.
        dirs: List of directories.
        patterns: Dictionary containing regex patterns for matching files.
        name_func: Function to generate a name based on the HPC filename.

    Returns: 
        mapped: Dictionary mapping generated names to sleep states and HPC files.
    """
    mapped = {}
    posttrial_pattern = patterns["posttrial"]
    hpc_pattern = patterns["hpc"]
    pfc_pattern = patterns["pfc"]
    states_pattern = patterns["states"]

    for dir in dirs:
        if dir.startswith('.'):
            continue
        #logger.debug(f"Dir: {dir}")
        if re.match(posttrial_pattern, dir, flags=re.IGNORECASE):
            dir_path = Path(root) / dir
        #    logger.debug(f"MATCH: {dir_path}")
            try:
                hpc_file = str(next(dir_path.glob(hpc_pattern)))
                pfc_file = str(next(dir_path.glob(pfc_pattern)))
                states_file = str(next(dir_path.glob(states_pattern)))
                name = name_func(hpc_file)
        #        logger.debug(f"{name}:({states_file}, {hpc_file})")
                mapped[name] = (states_file, hpc_file, pfc_file)
            except StopIteration:
                logger.warning(f"Expected files not found in directory: {dir_path}")
            except Exception as e:
                logger.error(f"Error processing directory {dir_path}: {e}")
        #else:
        #    logger.debug(f"MISMATCH: {Path(root) / dir}")
    return mapped

def decorate_cbd(cbd_name_func, CBD_DIR):
    """
    Decorator function to load the CBD overview file and wrap the CBD naming function.
    """
    try:
        path_to_overview = Path(CBD_DIR) / "overview.csv"
        overview_df = pd.read_csv(path_to_overview)
    except Exception as e:
        raise ValueError(f"Failed to load CBD overview file. {e}")

    def wrapper(file):
        return cbd_name_func(file, overview_df=overview_df)

    return wrapper

def create_name_cbd(file, overview_df):
    """
    Create a name for the CBD dataset based on the HPC filename and overview DataFrame.

    Args:
        file: HPC filename.
        overview_df: Overview DataFrame containing metadata.
    Returns: 
        Generated name.
    """
    pattern = r'Rat(\d+)_.*_SD(\d+)_([A-Z]+).*posttrial(\d+)'
    match = re.search(pattern, file)

    if not match:
        raise ValueError(f"Filename {file} does not match the expected pattern.")

    rat_num = int(match.group(1))
    sd_num = int(match.group(2))
    condition = str(match.group(3))
    posttrial_num = int(match.group(4))

    mask = (overview_df['Rat no.'] == rat_num) & (overview_df['Study Day'] == sd_num) & (overview_df['Condition'] == condition)

    if not any(mask):
        raise ValueError(f"No matching record found for Rat {rat_num}, SD {sd_num}, Condition {condition}.")

    treatment_value = overview_df.loc[mask, 'Treatment'].values[0]

    treatment = '1' if treatment_value != 0 else '0'

    return f'Rat{rat_num}_SD{sd_num}_{condition}_{treatment}_posttrial{posttrial_num}'

def create_name_rgs(fname):
    """
    Create a name for the RGS dataset based on the HPC filename.
    """
    pattern = r'Rat(\d+)_.*_SD(\d+)_([A-Z]+).*post[\w-]+trial(\d+)'
    match = re.search(pattern, fname, flags=re.IGNORECASE)

    if not match:
        raise ValueError(f"Filename {fname} does not match the expected pattern.")

    rat_num = int(match.group(1))
    sd_num = int(match.group(2))
    condition = str(match.group(3))
    posttrial_num = int(match.group(4))

    treatment = '2' if rat_num in [1, 2, 6, 9] else '3'

    return f'Rat{rat_num}_SD{sd_num}_{condition}_{treatment}_posttrial{posttrial_num}'

def create_name_os(hpc_fname):
    """
    Create a name for the OS dataset based on the HPC filename.
    """

    metadata = str(Path(hpc_fname).parent.parent.name).split("_")

    if len(metadata) < 4:
        raise ValueError(f"Filename {hpc_fname} does not contain enough metadata.")

    title = f"{metadata[1]}_{metadata[2]}_{metadata[3]}"

    pattern = r"post_trial(\d+)"
    match = re.search(pattern, hpc_fname, re.IGNORECASE)

    if not match:
        raise ValueError(f"Filename {hpc_fname} does not match the expected pattern.")

    title += f"_4_posttrial{match.group(1)}"

    return title
