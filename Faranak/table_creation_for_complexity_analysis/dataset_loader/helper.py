import yaml
import re
from pathlib import Path

def load_config(path):
    with open(path, 'r') as file:
        return yaml.safe_load(file)

def get_metadata(metaname):
    """
    Example metaname: Rat2_SD4_HC_2_posttrial1
    """
    metadata = {}

    metaname  = metaname.split('_')
    metadata["rat_id"]    = int(metaname[0][3:])     # from 'Rat2' → 2
    metadata["study_day"] = int(metaname[1][2:])     # from 'SD4'  → 4
    metadata["condition"] = metaname[2]              # e.g. 'HC'
    metadata["treatment"] = int(metaname[3])         # from '2'
    metadata["trial_num"] = metaname[4][9:]          # from 'posttrial1' → 1

    return metadata

def create_title(metadata):
    treatment = {
        0: "Negative CBD",
        1: "Positive CBD",
        2: "Negative RGS14",
        3: "Positive RGS14",
        4: "OS basic"
    }

    title = f"Rat {metadata['rat_id']} | Study Day: {metadata['study_day']} | "
    title += f"Treatment: {treatment[metadata['treatment']]} | Post-trial: {metadata['trial_num']}"
    return title

def str_to_tuple(string):
    string = string.strip("()")
    parts = string.split(",")
    return tuple(map(int, parts))

def load_data(fname):
    import numpy as np
    loaded_data = np.load(fname)
    loaded_dict = {str_to_tuple(key): loaded_data[key] for key in loaded_data.files}
    return loaded_dict

def extract_sleep_states(mat_path):
    import scipy.io
    import numpy as np

    state_map = {1: 'Wake', 3: 'NREM', 4: 'TS', 5: 'REM'}

    try:
        mat = scipy.io.loadmat(mat_path)
        # Try to locate the correct key
        possible_keys = [k for k in mat.keys() if not k.startswith('__')]
        for key in possible_keys:
            arr = mat[key]
            if isinstance(arr, (np.ndarray, list)) and arr.size > 0:
                states_array = arr.squeeze()
                if all(state in state_map for state in np.unique(states_array)):
                    return [state_map.get(int(s), "Unknown") for s in states_array]
    except Exception as e:
        print(f"❌ Failed to load or parse {mat_path}: {e}")
    return []