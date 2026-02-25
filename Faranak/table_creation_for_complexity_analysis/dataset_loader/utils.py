from pathlib import Path
import scipy.io
import numpy as np
import pandas as pd

def extract_windowed_sleep_states(state_file, signal_file, signal_key='PFC', fs=1000, window_sec=4):
    """
    Extracts windowed sleep states for a given signal and state file.

    Parameters:
    - state_file: path to the .mat file containing 'states'
    - signal_file: path to the .mat file containing the LFP signal
    - signal_key: name of the variable inside the .mat file for LFP
    - fs: sampling frequency in Hz
    - window_sec: window size in seconds

    Returns:
    - A list of (window_id, state_label)
    """
    try:
        state_map = {1: 'Wake', 3: 'NREM', 4: 'TS', 5: 'REM'}

        # Load signal and states
        signal_data = scipy.io.loadmat(signal_file)
        state_data = scipy.io.loadmat(state_file)

        signal = np.squeeze(signal_data[signal_key])
        states = np.squeeze(state_data['states'])

        # Calculate epoch-level labels (1 label per 4s epoch)
        epoch_duration = 4  # seconds
        epoch_samples = fs * epoch_duration
        total_epochs = len(states)

        # Now window the LFP signal
        window_samples = window_sec * fs
        num_windows = len(signal) // window_samples

        window_states = []
        for w in range(num_windows):
            start_sample = w * window_samples
            end_sample = start_sample + window_samples
            epoch_start = start_sample // epoch_samples
            epoch_end = min(end_sample // epoch_samples, total_epochs)

            if epoch_start < total_epochs and epoch_end > epoch_start:
                window_epoch_states = states[epoch_start:epoch_end]
                # Choose the most frequent state in the window
                if len(window_epoch_states) > 0:
                    unique, counts = np.unique(window_epoch_states, return_counts=True)
                    dominant_state_id = unique[np.argmax(counts)]
                    dominant_state = state_map.get(int(dominant_state_id), 'Unknown')
                    window_states.append((w, dominant_state))

        return window_states
    except Exception as e:
        print(f"Error processing windowed states: {e}")
        return []