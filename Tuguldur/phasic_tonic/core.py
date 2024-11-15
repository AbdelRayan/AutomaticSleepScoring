"""
Threshold based algorithm for detecting phasic REM states.
"""
from typing import Dict, List, Tuple

import numpy as np
from scipy.signal import hilbert
from neurodsp.filt import filter_signal


def compute_thresholds(rem_epochs: Dict[Tuple[int, int], np.ndarray], fs: float):
    """
    Compute thresholds for detecting phasic REM epochs.

    Parameters
    ----------
    rem_epochs : Dict[Tuple[int, int], np.ndarray]
        Dictionary where keys are tuples indicating the start and end times of REM epochs (in seconds),
        and values are the corresponding EEG signal segments.
    fs : float
        Sampling rate, in Hz.

    Returns
    -------
    thresholds : Tuple[float, float, float]
        A tuple containing:

            - 10th percentile of smoothed trough differences across all epochs.
            - 5th percentile of smoothed trough differences across all epochs.
            - Mean instantaneous amplitude across all REM epochs.
    epoch_trough_idx : Dict[Tuple[int, int], np.ndarray]
        Dictionary mapping REM epoch indices to their trough indices.
    epoch_smooth_diffs : Dict[Tuple[int, int], np.ndarray]
        Dictionary mapping REM epoch indices to their smoothed trough differences.
    epoch_trough_idx : Dict[Tuple[int, int], np.ndarray]
        Dictionary mapping REM epoch indices to their instantaneous amplitudes.
    """
    if not rem_epochs:
        raise ValueError("The rem_epochs is empty.")

    all_trough_diffs = []
    all_inst_amplitudes = []
    epoch_amplitudes = {}
    epoch_smooth_diffs = {}
    epoch_trough_idx = {}
   
    for rem_idx, epoch in rem_epochs.items():
        # Instantaneous phase and amplitude
        inst_phase, inst_amp = preprocess_rem_epoch(epoch, fs)
        
        # Detect trough indices
        trough_idx = detect_troughs(inst_phase)

        # Compute differences between consecutive trough indices
        trough_diffs = np.diff(trough_idx)
       
        # Smooth the trough differences
        smooth_diffs = smooth_signal(trough_diffs)
        
        # Store per-epoch data
        epoch_smooth_diffs[rem_idx] = smooth_diffs
        epoch_trough_idx[rem_idx] = trough_idx
        epoch_amplitudes[rem_idx] = inst_amp
       
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
    
    return thresholds, epoch_trough_idx, epoch_smooth_diffs, epoch_amplitudes


def is_valid_phasic(
    smoothed_diffs_slice: np.ndarray, 
    inst_amp_slice: np.ndarray, 
    threshold_percentile_5: float, 
    mean_amplitude_threshold: float
) -> bool:
    """
    Determine if a candidate phasic REM period passes the thresholds.

    Parameters
    ----------
    smoothed_diffs_slice : np.ndarray
        Array of smoothed trough differences for the candidate period.
    inst_amp_slice : np.ndarray
        Array of instantaneous amplitudes for the candidate period.
    threshold_percentile_5 : float
        5th percentile of smoothed trough differences across all epochs.
    mean_amplitude_threshold : float
        Mean instantaneous amplitude across all REM epochs.

    Returns
    -------
    bool
        `True` if the candidate period meets the phasic criteria, `False` otherwise.
    """
    min_smoothed_diff = np.min(smoothed_diffs_slice)
    mean_amp = np.mean(inst_amp_slice)
    return (min_smoothed_diff <= threshold_percentile_5) and (mean_amp >= mean_amplitude_threshold)


def get_phasic_candidates(
    smoothed_trough_differences: np.ndarray, 
    trough_indices: np.ndarray, 
    threshold_percentile_10: float, 
    thr_dur: float, 
    fs: float
) -> List[Tuple[int, int]]:
    """
    Identify candidate phasic REM periods based on smoothed trough differences.

    Parameters
    ----------
    smoothed_trough_differences : np.ndarray
        Array of smoothed trough differences.
    trough_indices : np.ndarray
        Array of trough indices in the EEG signal.
    threshold_percentile_10 : float
        10th percentile of smoothed trough differences across all epochs.
    thr_dur : float
        Minimum duration for a phasic epoch in milliseconds.
    fs : float
        Sampling rate, in Hz.

    Returns
    -------
    List[Tuple[int, int]]
        List of tuples indicating the start and end trough indices of candidate phasic periods.
    """
    cand_idx = np.where(smoothed_trough_differences <= threshold_percentile_10)[0]
    cand = get_sequences(cand_idx)

    candidates = []
    for start, end in cand:
        if end < len(trough_indices):
            end += 1  # Add 1 to `end` to align with `trough_indices`

        # Compute duration in milliseconds
        duration_ms = ((trough_indices[end] - trough_indices[start]) / fs) * 1000
        if duration_ms >= thr_dur:  
            candidates.append((start, end))
    return candidates


def preprocess_rem_epoch(epoch: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess a REM epoch by filtering in theta band and compute the Hilbert transform.

    Parameters
    ----------
    epoch : np.ndarray
        EEG signal segment for a REM epoch.
    fs : float
        Sampling rate, in Hz.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing:
        
            - Instantaneous phase.
            - Instantaneous amplitude.
    """
    epoch = filter_signal(epoch, fs, 'bandpass', (5, 12), remove_edges=False)
    analytic_signal = hilbert(epoch)
    return np.angle(analytic_signal), np.abs(analytic_signal)


def detect_troughs(signal: np.ndarray, threshold: float = -3.0) -> np.ndarray:
    """
    Detect troughs in a signal that fall below a specified threshold.

    Parameters
    ----------
    signal : np.ndarray
        Signal in which to detect troughs (e.g., instantaneous phase).
    threshold : float, optional
        Threshold value for trough detection. Defaults to -3.0.

    Returns
    -------
    np.ndarray
        Array of indices where troughs occur.
    """
    lidx = np.where(signal[0:-2] > signal[1:-1])[0]
    ridx = np.where(signal[1:-1] <= signal[2:])[0]
    thidx = np.where(signal[1:-1] < threshold)[0]
    return np.intersect1d(lidx, np.intersect1d(ridx, thidx)) + 1


def smooth_signal(signal: np.ndarray, window_size: int = 11) -> np.ndarray:
    """
    Apply a moving average filter to smooth a signal.

    Parameters
    ----------
    signal : np.ndarray
        Input signal to be smoothed.
    window_size : int, optional
        Size of the moving window. Defaults to 11.

    Returns
    -------
    np.ndarray
        Smoothed signal.
    """
    filt = np.ones(window_size) / window_size
    return np.convolve(signal, filt, 'same')


def get_sequences(a: np.ndarray, ibreak: int = 1) -> List[Tuple[int, int]]:
    """
    Identify contiguous sequences in an array.

    Parameters
    ----------
    a : np.ndarray
        1D array of indices.
    ibreak : int, optional
        Maximum allowed gap between consecutive indices to be considered contiguous. Defaults to 1.

    Returns
    -------
    List[Tuple[int, int]]
        List of tuples where each tuple represents the start and end indices of a contiguous sequence.
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
    Extract segments of a signal between specified start and end sample indices.

    Parameters
    ----------
    idx : List[Tuple[int, int]]
        List of tuples indicating start and end sample indices for each segment.
    signal : np.ndarray
        The EEG signal from which to extract segments.

    Returns
    -------
    List[np.ndarray]
        List of signal segments corresponding to the specified indices.
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
    min_dur: float = 3
) -> Dict[Tuple[int, int], np.ndarray]:
    """
    Extract REM sleep epochs from EEG data based on a hypnogram.

    Parameters
    ----------
    eeg : np.ndarray
        EEG signal array.
    hypno : np.ndarray
        Hypnogram array. Expects an array of 1-second epochs where REM stage corresponds to value '5'.
    fs : int
        Sampling rate, in Hz.
    min_dur : float, optional
        Minimum duration for a REM epoch in seconds. Defaults to 3.

    Returns
    -------
    Dict[Tuple[int, int], np.ndarray]
        Dictionary where keys are tuples of (start_sample, end_sample) for each REM epoch,
        and values are the corresponding EEG signal segments.

    Raises
    ------
    ValueError
        If no REM epochs longer than `min_dur` seconds are found.
    """
    rem_seq = get_sequences(np.where(hypno == 5)[0])  # Assuming 5 represents REM sleep
    # Select only REM epochs above min_dur
    rem_seq = [(start, end) for start, end in rem_seq if (end - start) > min_dur] 
    rem_idx = [(start * fs, (end + 1) * fs) for start, end in rem_seq]
   
    if not rem_idx:
        raise ValueError(f"No REM epochs greater than {min_dur} seconds.")
   
    rem_epochs = get_segments(rem_idx, eeg)
    return {seq: seg for seq, seg in zip(rem_seq, rem_epochs)}


def get_start_end(sleep_states: np.ndarray, sleep_state_id: int, min_dur: float = 3) -> Tuple[List[int], List[int]]:
    """Get start and end indices for a specific sleep state."""
    seq = get_sequences(np.where(sleep_states == sleep_state_id)[0])
    start, end = [], []
    for s, e in seq:
        if (e-s) > min_dur:
            start.append(s)
            end.append(e)
    return (start, end)
