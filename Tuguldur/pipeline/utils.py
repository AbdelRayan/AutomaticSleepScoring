from typing import Dict, List, Tuple
import numpy as np
from mne.filter import resample
import yasa

def preprocess(signal: np.ndarray, n_down: int, target_fs=500) -> np.ndarray:
    """Downsample and remove artifacts."""
    # Downsample to 500 Hz
    data = resample(signal, down=n_down, method='fft', npad='auto')
    # Remove artifacts
    art_std, _ = yasa.art_detect(data, target_fs , window=1, method='std', threshold=4)
    art_up = yasa.hypno_upsample_to_data(art_std, 1, data, target_fs)
    data[art_up] = 0
    data -= data.mean()
    return data

def get_sequences(a: np.ndarray, ibreak: int = 1) -> List[Tuple[int, int]]:
    """
    Identify contiguous sequences.
    """
    if len(a) == 0:
        return []

    diff = np.diff(a)
    breaks = np.where(diff > ibreak)[0]
    breaks = np.append(breaks, len(a) - 1)
    
    sequences = []
    start_idx = 0
    
    for break_idx in breaks:
        end_idx = break_idx
        sequences.append((a[start_idx], a[end_idx]))
        start_idx = end_idx + 1
    
    return sequences

def get_segments(idx: List[Tuple[int, int]], signal: np.ndarray) -> List[np.ndarray]:
    """
    Extract segments of the signal between specified start and end time indices.
    """
    segments = []
    for (start_time, end_time) in idx:
        if end_time > len(signal):
            end_time = len(signal)
        segment = signal[start_time:end_time]
        segments.append(segment)
    
    return segments

def get_rem_epochs(
    eeg: np.ndarray, 
    hypno: np.ndarray, 
    fs: int, 
    min_dur: float = 3) -> Dict[Tuple[int, int], np.ndarray]:
    """
    Extract REM epochs from EEG data based on hypnogram.
    """
    rem_seq = get_sequences(np.where(hypno == 5)[0]) # Assuming 5 represents REM sleep
    rem_idx = [(start * fs, (end + 1) * fs) for start, end in rem_seq if (end - start) > min_dur]
   
    if not rem_idx:
        raise ValueError(f"No REM epochs greater than {min_dur} seconds.")
   
    rem_epochs = get_segments(rem_idx, eeg)
    return {seq: seg for seq, seg in zip(rem_seq, rem_epochs)}

def get_start_end(sleep_states: np.ndarray, sleep_state_id: int) -> Tuple[List[int], List[int]]:
    """Get start and end indices for a specific sleep state."""
    seq = get_sequences(np.where(sleep_states == sleep_state_id)[0])
    start, end = [], []
    for s, e in seq:
        start.append(s)
        end.append(e)
    return (start, end)

def detect_troughs(signal: np.ndarray, threshold: float = -3.0) -> np.ndarray:
    """Detect troughs in a signal below a certain threshold."""
    lidx = np.where(signal[0:-2] > signal[1:-1])[0]
    ridx = np.where(signal[1:-1] <= signal[2:])[0]
    thidx = np.where(signal[1:-1] < threshold)[0]
    return np.intersect1d(lidx, np.intersect1d(ridx, thidx)) + 1

def smooth_signal(signal: np.ndarray, window_size: int = 11) -> np.ndarray:
    """Apply moving average smoothing to a signal."""
    filt = np.ones(window_size) / window_size
    return np.convolve(signal, filt, 'same')