from src.DatasetLoader import DatasetLoader
from src.runtime_logger import logger_setup
from pathlib import Path

import numpy as np
import pandas as pd

def analysis():
    pass

if __name__ == "__main__":
    logger = logger_setup()

    CONF = "/home/nero/phasic_tonic/configs/dataset_loading.yaml"
    CBD_DIR = "/home/nero/datasets/CBD/"
    RGS_DIR = "/home/nero/datasets/RGS14/"
    OS_DIR = "/home/nero/datasets/OSbasic/"

    datasets = {
    # 'dataset_name' : {'dir' : '/path/to/dataset', 'pattern_set': 'pattern_set_in_config'}
        "CBD": {"dir": CBD_DIR, "pattern_set": "CBD"},
        "RGS": {"dir": RGS_DIR, "pattern_set": "RGS"},
        "OS": {"dir": OS_DIR, "pattern_set": "OS"}
    }

    Datasets = DatasetLoader(datasets, CONF)
    mapped_datasets = Datasets.load_datasets()

    print(type(mapped_datasets))
