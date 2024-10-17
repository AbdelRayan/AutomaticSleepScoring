"""
Threshold based algorithm for detecting phasic REM states.
"""
from typing import Dict, List, Tuple

import numpy as np
from scipy.signal import hilbert
from neurodsp.filt import filter_signal

from .utils import get_sequences, get_rem_epochs

def preprocess_rem_epoch(epoch: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """Preprocess a REM epoch by applying bandpass filter and Hilbert transform."""
    epoch = filter_signal(epoch, fs, 'bandpass', (5, 12), remove_edges=False)
    analytic_signal = hilbert(epoch)
    return np.angle(analytic_signal), np.abs(analytic_signal)

def get_phasic_candidates(
    smoothed_trough_differences: np.ndarray, 
    trough_indices: np.ndarray, 
    threshold_percentile_10: float, 
    thr_dur: float, 
    fs: float
    ) -> List[Tuple[int, int]]:
    """
    Get candidate phasic REM periods based on smoothed trough differences.
    """
    cand_idx = np.where(smoothed_trough_differences <= threshold_percentile_10)[0]
    cand = get_sequences(cand_idx)
    return [(start, end) for start, end in cand if ((trough_indices[end] - trough_indices[start] + 1) / fs) * 1000 >= thr_dur]

def is_valid_phasic(
    smoothed_diffs_slice: np.ndarray, 
    inst_amp_slice: np.ndarray, 
    threshold_percentile_5: float, 
    mean_amplitude_threshold: float
    ) -> bool:
    """
    Check if a candidate phasic REM period is valid.
    """
    min_smoothed_diff = np.min(smoothed_diffs_slice)
    mean_amp = np.mean(inst_amp_slice)
    return (min_smoothed_diff <= threshold_percentile_5) and (mean_amp >= mean_amplitude_threshold)

def compute_thresholds(rem_epochs: Dict[Tuple[int, int], np.ndarray], fs: float):
    """
    Compute thresholds for detecting phasic REM periods.

    Parameters
    ----------
    rem_epochs : Dict[Tuple[int, int], np.ndarray]
        Dictionary where keys are tuples indicating the start and end times of REM epochs (in seconds),
        and values are the corresponding EEG signal segments.
    fs : float
        Sampling frequency.

    Returns
    -------
    thresholds : Tuple[float, float, float]
        A tuple containing:
        - 10th percentile of smoothed trough differences across all epochs.
        - 5th percentile of smoothed trough differences across all epochs.
        - Mean instantaneous amplitude across all REM epochs.
    trough_idx_seq : Dict[Tuple[int, int], np.ndarray]
        Dictionary mapping REM epoch indices to their trough indices.
    smooth_differences : Dict[Tuple[int, int], np.ndarray]
        Dictionary mapping REM epoch indices to their smoothed trough differences.
    eeg_seq : Dict[Tuple[int, int], np.ndarray]
        Dictionary mapping REM epoch indices to their instantaneous amplitudes.
    """
    if not rem_epochs:
        raise ValueError("The rem_epochs is empty.")
    
    all_trough_diffs = []
    all_inst_amplitudes = []
    eeg_seq = {}
    smooth_differences = {}
    trough_idx_seq = {}
   
    for rem_idx, epoch in rem_epochs.items():
        # Instantaneous phase and amplitude
        inst_phase, inst_amp = preprocess_rem_epoch(epoch, fs)
        
        # Detect trough indices
        trough_idx = detect_troughs(inst_phase)

        # Compute differences between consecutive trough indices
        trough_diffs = np.diff(trough_idx)
       
        # Smooth the trough differences
        smoothed_diffs = smooth_signal(trough_diffs)
        
        # Store per-epoch data
        smooth_differences[rem_idx] = smoothed_diffs
        trough_idx_seq[rem_idx] = trough_idx
        eeg_seq[rem_idx] = inst_amp
       
        # Accumulate trough differences and instantaneous amplitudes
        all_trough_diffs.extend(trough_diffs)
        all_inst_amplitudes.extend(inst_amp)

    # Convert accumulated lists to NumPy arrays
    all_trough_diffs = np.array(all_trough_diffs)
    all_inst_amplitudes = np.array(all_inst_amplitudes)

    # Smooth the concatenated trough differences across all epochs
    all_trough_diffs_smoothed = smooth_signal(all_trough_diffs)
    
    # Compute thresholds based on percentiles and mean amplitude
    threshold_10th_percentile = np.percentile(all_trough_diffs_smoothed, 10)
    threshold_5th_percentile = np.percentile(all_trough_diffs_smoothed, 5)
    mean_inst_amplitude = np.mean(all_inst_amplitudes)

    thresholds = (threshold_10th_percentile, threshold_5th_percentile, mean_inst_amplitude)
    
    return thresholds, trough_idx_seq, smooth_differences, eeg_seq

def detect_phasic(
    eeg: np.ndarray, 
    hypno: np.ndarray, 
    fs: float, 
    thr_dur: float = 900
    ) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
    """
    Detect phasic REM periods in EEG data based on the method described by Mizuseki et al. (2011).
    
    Parameters
    ----------
    eeg : np.ndarray
        EEG signal.
    hypno : np.ndarray
        Hypnogram array.
    fs : float
        Sampling frequency.
    thr_dur : float, optional
        Minimum duration threshold for phasic REM in milliseconds, by default 900.

    Returns
    -------
    Dict[Tuple[int, int], List[Tuple[int, int]]]
        Dictionary of detected phasic REM periods for each REM epoch.
    """
    rem_epochs = get_rem_epochs(eeg, hypno, fs)
    thresholds, trough_idx_seq, smooth_difference_seq, eeg_seq = compute_thresholds(rem_epochs, fs)
    threshold_percentile_10, threshold_percentile_5, mean_amplitude_threshold = thresholds

    phasicREM = {}

    for rem_idx, trough_indices in trough_idx_seq.items():
        rem_start, rem_end = rem_idx
        offset = rem_start * fs
        smooth_difference = smooth_difference_seq[rem_idx]
        inst_amp = eeg_seq[rem_idx]

        # Get candidate periods
        candidates = get_phasic_candidates(smooth_difference, trough_indices, threshold_percentile_10, thr_dur, fs)

        # Validate candidates
        valid_periods = []
        for start, end in candidates:
            smoothed_diffs_slice = smooth_difference[start:end]
            inst_amp_slice = inst_amp[trough_indices[start]:trough_indices[end] + 1]

            if is_valid_phasic(smoothed_diffs_slice, inst_amp_slice, threshold_percentile_5, mean_amplitude_threshold):
                start_time = trough_indices[start] + offset
                end_time = min(trough_indices[end] + offset, rem_end * fs)
                valid_periods.append((int(start_time), int(end_time) + 1))

        if valid_periods:
            phasicREM[rem_idx] = valid_periods

    return phasicREM