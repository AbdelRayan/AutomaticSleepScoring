import os
import re
import numpy as np
import pandas as pd
import yasa

from pathlib import Path
from scipy.io import loadmat
from mne.filter import resample
from tqdm import tqdm

from phasic_tonic.utils import *
from phasic_tonic.helper import *
from phasic_tonic.runtime_logger import logger_setup

CBD_DIR = "/home/nero/datasets/CBD/"
RGS_DIR = "/home/nero/datasets/RGS14/"
OS_DIR = "/home/nero/datasets/OSbasic/"

OUTPUT_DIR  = "/home/nero/datasets/preprocessed/"
CBD_OVERVIEW_PATH = "/home/nero/phasic_tonic/notebooks/new_method/overview.csv"

fs_cbd = 2500
fs_os = 2500
fs_rgs = 1000

targetFs = 500
n_down_cbd = fs_cbd/targetFs
n_down_rgs = fs_rgs/targetFs
n_down_os = fs_os/targetFs

nrem_rem_percentage = {
    'rat_id': [],
    'study_day': [],
    'condition': [],
    'treatment': [],
    'trial_num': [],
    'nrem': [],
    'rem': []
    }

phasic_tonic_percentage = {
    'rat_id': [],
    'study_day': [],
    'condition': [],
    'treatment': [],
    'trial_num': [],
    'state': [],
    'epoch_id': [],
    'duration': [],
    'count': [],
    'rem_epoch_dur': []
    }

def analysis(data, metadata, targetFs=targetFs):
    if len(data) == 0:
        _save_to_dict(durations=[], metadata=metadata, state="phasic", epoch_id=0, rem_epoch_dur=0)
        _save_to_dict(durations=[], metadata=metadata, state="tonic", epoch_id=0, rem_epoch_dur=0)
        return None

    # Detect phasic epochs
    phasicREM = phasic_detect(rem=data, fs=targetFs, thr_dur=900, nfilt=11)
    logger.debug("Detected phasic: {0}.".format(phasicREM))

    for i, rem_idx in enumerate(phasicREM):
        rem_start, rem_end = rem_idx
        epoch_dur = rem_end - rem_start
        phasic = phasicREM[rem_idx]

        _save_to_dict(durations=phasic, metadata=metadata, state="phasic", epoch_id=i, rem_epoch_dur=epoch_dur)

        # Tonic epochs are determined as everywhere that is not phasic in the REM epoch
        tonic = get_tonic(rem_start*targetFs, rem_end*targetFs, phasicREM[rem_idx])
        logger.debug("REM epoch: ({0}, {1}) ".format(rem_start*targetFs, rem_end*targetFs))

        _save_to_dict(durations=tonic, metadata=metadata, state="tonic", epoch_id=i, rem_epoch_dur=epoch_dur)

def _save_to_dict(durations, metadata, state, epoch_id, rem_epoch_dur):
    # Add metadata
    for condition in metadata.keys():
        phasic_tonic_percentage[condition].append(metadata[condition])
    
    phasic_tonic_percentage['state'].append(state)
    phasic_tonic_percentage["epoch_id"].append(epoch_id)
    phasic_tonic_percentage['duration'].append(np.sum(np.diff(durations))/targetFs)
    phasic_tonic_percentage['count'].append(len(durations))
    phasic_tonic_percentage['rem_epoch_dur'].append(rem_epoch_dur)

def run(mapped, name_func, n_down, targetFs=targetFs):
  with tqdm(mapped.keys()) as t:
      for state in t:
        hpc = mapped[state]
  
        title = name_func(hpc)
        t.set_postfix_str(title) # Set the title for the progress bar
        metadata = get_metadata(title)

        logger.debug("Loading: {0}".format(title))
        logger.debug("fname: {0}".format(state))

        # Load the LFP data
        lfpHPC = loadmat(hpc)['HPC']
        lfpHPC = lfpHPC.flatten()

        # Load the states
        hypno = loadmat(state)['states']
        hypno = hypno.flatten()

        # Skip if no REM epoch is detected
        if(not (np.any(hypno == 5))):
            logger.debug("No REM detected. Skipping.")
            continue

        logger.debug("STARTED: Resampling to 500 Hz.")
        # Downsample to 500 Hz
        data_resample = resample(lfpHPC, down=n_down, method='fft', npad='auto')
        logger.debug("FINISHED: Resampling to 500 Hz.")
        logger.debug("Resampled: {0} -> {1}.".format(str(lfpHPC.shape), str(data_resample.shape)))
        del lfpHPC

        logger.debug("STARTED: Remove artifacts.")
        # Remove artifacts
        art_std, _ = yasa.art_detect(data_resample, targetFs , window=1, method='std', threshold=4, verbose='info')
        art_up = yasa.hypno_upsample_to_data(art_std, 1, data_resample, targetFs)
        data_resample[art_up] = 0
        logger.debug("FINISHED: Remove artifacts.")
        del art_std, art_up

        data_resample -= data_resample.mean()

        logger.debug("STARTED: Extract REM epochs.")
        rem_seq = get_sequences(np.where(hypno == 5)[0])

        # Ensure each REM epochs are greater than the minimum duration
        try:
            rem_idx = ensure_duration(rem_seq, min_dur=3)
        except:
            logger.debug("Skipping.")
            continue
        
        # Another representation: matrix of 2 columns and n rows (n number of rem epochs), first row is the start and second is for end idx.          
        logger.debug("REM indices: {0}.".format(rem_seq))

        # get REM segments
        rem_idx = [(start * targetFs, (end+1) * targetFs) for start, end in rem_seq]
        rem_epochs = get_segments(rem_idx, data_resample)
        logger.debug("FINISHED: Extract REM epochs.")

        # Combine the REM indices with the corresponding downsampled segments
        rem = {seq:seg for seq, seg in zip(rem_seq, rem_epochs)}
        del rem_epochs, rem_seq

        if metadata["trial_num"] == '5':
            for i, partition in enumerate(partition_to_4(rem)):
                metadata["trial_num"] = '5-' + str(i+1)

                for condition in metadata.keys():
                    nrem_rem_percentage[condition].append(metadata[condition])
                        
                nrem_dur = np.sum(np.diff(get_sequences(np.where(hypno[i*2700:(i+1)*2700] == 3)[0])))
                logger.debug("NREM Duration: {0}".format(nrem_dur))
                nrem_rem_percentage["nrem"].append(nrem_dur)
                
                rem_dur = np.sum(np.diff(get_sequences(np.where(hypno[i*2700:(i+1)*2700] == 5)[0])))
                logger.debug("REM Duration: {0}".format(rem_dur))
                nrem_rem_percentage["rem"].append(rem_dur)

                logger.debug("Partition: {0}".format(str(partition)))
                # Detect phasic & save phasic/tonic percentage, rem_epoch durations
                analysis(partition, metadata)
        else:
            for condition in metadata.keys():
                nrem_rem_percentage[condition].append(metadata[condition])
                    
            nrem_dur = np.sum(np.diff(get_sequences(np.where(hypno==3)[0])))
            logger.debug("NREM Duration: {0}".format(nrem_dur))
            nrem_rem_percentage["nrem"].append(nrem_dur)
            
            rem_dur = np.sum(np.diff(get_sequences(np.where(hypno==5)[0])))
            logger.debug("REM Duration: {0}".format(rem_dur))
            nrem_rem_percentage["rem"].append(rem_dur)

            # Detect phasic & save phasic/tonic percentage, rem_epoch durations
            analysis(rem, metadata)

cbd_patterns = {
    "posttrial":r"[\w-]+posttrial[\w-]+",
    "hpc":"*HPC*continuous*",
    "states":"*-states*"
}

rgs_patterns = {
    "posttrial":r"[\w-]+post[\w-]+trial[\w-]+",
    "hpc":"*HPC*continuous*",
    "states":"*-states*"
}

os_patterns = {
    "posttrial":r".*post_trial.*",
    "hpc":"*HPC*",
    "states":"*states*"
}

def load_dataset(DATASET_DIR, pattern_args):
    mapped = {}

    posttrial_pattern = pattern_args["posttrial"]
    hpc_pattern = pattern_args["hpc"]
    states_pattern = pattern_args["states"]

    for root, dirs, _ in os.walk(DATASET_DIR):
        for dir in dirs:
            # Check if the directory is a post trial directory
            if re.match(posttrial_pattern, dir, flags=re.IGNORECASE):
                dir = Path(os.path.join(root, dir))
                HPC_file = next(dir.glob(hpc_pattern))
                states = next(dir.glob(states_pattern))
                mapped[str(states)] = str(HPC_file)
    return mapped

if __name__ == "__main__":
    logger = logger_setup()
    logger.debug("Saving to: {0}".format(OUTPUT_DIR))

    mapped1 = load_dataset(CBD_DIR, pattern_args=cbd_patterns)
    mapped2 = load_dataset(RGS_DIR, pattern_args=rgs_patterns)
    mapped3 = load_dataset(OS_DIR, pattern_args=os_patterns)
    
    # Wrapper for create name function for CBD dataset
    def wrapper(hpc):
        overview_df = pd.read_csv(CBD_OVERVIEW_PATH)
        return create_name_cbd(hpc, overview_df=overview_df)
    
    # CBD preprocessing
    logger.info("Number of CBD recordings: {0}.".format(len(mapped1)))
    run(mapped=mapped1, name_func=wrapper, n_down=n_down_cbd)

    # RGS14 preprocessing
    logger.info("Number of RGS14 recordings: {0}.".format(len(mapped2)))
    run(mapped=mapped2, name_func=create_name_rgs, n_down=n_down_rgs)

    # OS basic preprocessing
    logger.info("Number of OS_Basic recordings: {0}.".format(len(mapped3)))
    run(mapped=mapped3, name_func=create_name_os, n_down=n_down_os)

    nrem_rem_df = pd.DataFrame({key:pd.Series(value) for key, value in nrem_rem_percentage.items()})
    nrem_rem_df.to_csv("nrem_rem_percentage.csv", index=False)

    ph_df = pd.DataFrame({key:pd.Series(value) for key, value in phasic_tonic_percentage.items()})
    ph_df.to_csv("phasic_tonic_percentage.csv", index=False)

    