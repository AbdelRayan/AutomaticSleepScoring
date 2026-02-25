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
        """
        self.config = load_config(CONFIG_DIR)
        self.combined_mapped = {}

        self.naming_functions = {}

        if 'CBD' in self.config:
            try:
                cbd_wrapper = decorate_cbd(cbd_name_func=create_name_cbd, CBD_DIR=self.config['CBD']['path'])
                self.naming_functions["CBD"] = cbd_wrapper
                logger.info("CBD naming function loaded.")
            except Exception as e:
                logger.warning(f"Could not initialize CBD naming function: {e}")

        if 'RGS' in self.config:
            self.naming_functions["RGS"] = create_name_rgs
            logger.info("RGS naming function loaded.")

        if 'OS' in self.config:
            self.naming_functions["OS"] = create_name_os
            logger.info("OS naming function loaded.")

    def load_datasets(self):
        """
        Load datasets from the configured sources.

        Returns:
            dict: Combined mapping of dataset names to their respective file tuples.
        """
        for name, info in self.config.items():
            if name not in self.naming_functions:
                logger.warning(f"No naming function defined for dataset: {name}. Skipping...")
                continue

            logger.info(f"STARTED: Loading the dataset {name}")
            dataset_dir = info['path']
            name_func = self.naming_functions[name]

            for root, dirs, _ in os.walk(dataset_dir):
                mapped = process_directory(root, dirs, info, name_func)
                self.combined_mapped.update(mapped)

            logger.info(f"FINISHED: Loading the dataset {name}")
            logger.info(f"Number of files loaded: {len(self.combined_mapped)}")

        return self.combined_mapped

    def __getitem__(self, key):
        return self.combined_mapped[key]

    def __iter__(self):
        return iter(self.combined_mapped)

    def __len__(self):
        return len(self.combined_mapped)

    def __str__(self):
        return f"Total loaded recordings: {len(self.combined_mapped)}"


def process_directory(root, dirs, patterns, name_func):
    """
    Process a directory to map sleep states and HPC files using specified patterns and naming function.

    Args:
        root: Root directory path.
        dirs: List of subdirectories.
        patterns: Dictionary containing regex patterns for matching files.
        name_func: Function to generate a name based on the HPC filename.

    Returns:
        dict: Mapping from generated names to file tuples.
    """
    mapped = {}
    posttrial_pattern = patterns["posttrial"]
    hpc_pattern = patterns["hpc"]
    pfc_pattern = patterns["pfc"]
    states_pattern = patterns["states"]

    for dir in dirs:
        if dir.startswith('.'):
            continue
        if re.match(posttrial_pattern, dir, flags=re.IGNORECASE):
            dir_path = Path(root) / dir
            try:
                hpc_file = str(next(dir_path.glob(hpc_pattern)))
                pfc_file = str(next(dir_path.glob(pfc_pattern)))
                states_file = str(next(dir_path.glob(states_pattern)))
                name = name_func(hpc_file)
                mapped[name] = (states_file, hpc_file, pfc_file)
            except StopIteration:
                logger.warning(f"Expected files not found in directory: {dir_path}")
            except Exception as e:
                logger.error(f"Error processing directory {dir_path}: {e}")
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
    pattern = r'Rat(\d+)_.*_SD(\d+)_([A-Z]+).*post[\w-]*trial(\d+)'
    match = re.search(pattern, fname, flags=re.IGNORECASE)

    if not match:
        raise ValueError(f"Filename {fname} does not match the expected pattern.")

    rat_num = int(match.group(1))
    sd_num = int(match.group(2))
    condition = str(match.group(3))
    posttrial_num = int(match.group(4))

    # Rat IDs 1, 2, 6, 9 → Negative treatment (2), others → Positive (3)
    treatment = '2' if rat_num in [1, 2, 6, 9] else '3'

    return f'Rat{rat_num}_SD{sd_num}_{condition}_{treatment}_posttrial{posttrial_num}'


def create_name_os(hpc_fname):
    """
    Create a clean name for OS dataset in the format:
    Rat{rat_id}_SD{sd_num}_{condition}_4_posttrial{posttrial}
    """

    try:
        path = Path(hpc_fname)

        # --- Get posttrial number
        trial_folder = path.parent.name
        match = re.search(r'post[\-_]*trial(\d+)', trial_folder, re.IGNORECASE)
        if not match:
            raise ValueError(f"Could not extract posttrial from: {trial_folder}")
        posttrial = match.group(1)

        # --- Get rat ID from 'Rat_6' or 'Rat6'
        rat_folder = path.parents[2].name
        rat_match = re.search(r'Rat[_\-]?(\d+)', rat_folder, re.IGNORECASE)
        if not rat_match:
            raise ValueError(f"Could not extract rat ID from: {rat_folder}")
        rat_id = rat_match.group(1)

        # --- Search for SD inside higher folders
        full_parts = path.parts
        sd_match = None
        for part in full_parts:
            match = re.search(r'(?:SD|study[_\s]*day)[_\s]*(\d+)', part, re.IGNORECASE)
            if match:
                sd_match = match
                break

        sd_num = sd_match.group(1) if sd_match else '1'  # fallback to SD1 if not found

        condition_folder = path.parents[1].name
        valid_conditions = ['OR-N', 'OD-N', 'OR', 'OD', 'HC', 'CN']  # long first
        condition = None
        condition_folder_lower = condition_folder.lower()
        for cond in valid_conditions:
            if cond.lower() in condition_folder_lower:
                condition = cond
                break

        if not condition:
            raise ValueError(f"Could not extract condition from: {condition_folder}")


        return f'Rat{rat_id}_SD{sd_num}_{condition}_4_posttrial{posttrial}'

    except Exception as e:
        raise ValueError(f"Error creating OS name from {hpc_fname}: {e}")