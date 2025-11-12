from scipy.stats import binned_statistic
import scipy
import numpy as np
import matplotlib.pyplot as plt

import emd as emd
import emd.sift as sift
import emd.spectra as spectra
import scipy.stats

from scipy.io import loadmat
import numpy as np
from neurodsp.filt import filter_signal
import copy
import emd
from scipy.spatial import cKDTree
from tqdm import tqdm
from scipy.io import loadmat
from scipy.stats import entropy
import os
import re


def extract_frequency_sampling(lfp, hypno):
    fs = len(lfp)/len(hypno)

    return int(fs)


def get_data(lfp_path, state_path, bregion):

    data = scipy.io.loadmat(lfp_path)
    states = scipy.io.loadmat(state_path)

    if bregion == 'PFC':
        lfp = np.squeeze(data['PFC'])
    else:
        lfp = np.squeeze(data['HPC'])
        
    hypno = np.squeeze(states['states'])

    fs = extract_frequency_sampling(lfp, hypno)

    unique = np.unique(hypno)
    if unique[0] == 0:
        print('There was 0 in the dataset')
        lfp = lfp[7*fs:-11*fs]
        hypno = hypno[7:-11]
    else:
        None

    return lfp, hypno, fs


def plot_hypnogram(hypno):
    labels = {1: "Wake", 3: "NREM", 4: "Intermediate", 5: "REM"}
    plt.figure(figsize=(12, 6))
    time = np.arange(len(hypno)) / 60
    plt.step(time, hypno)
    plt.xlabel('Time (m)')
    plt.yticks(list(labels.keys()), list(labels.values()))
    plt.ylabel('States')
    plt.title('Hypnogram of sleep')
    plt.show()


def imf_freq(imf, sample_rate, mode='nht'):
    _, IF, _ = emd.spectra.frequency_transform(imf, sample_rate, 'nht')
    freq_vec = np.mean(IF, axis=0)
    return freq_vec


def extract_imfs_by_pt_intervals(lfp, fs, interval, config, return_imfs_freqs=False):

    all_imfs = []
    all_imf_freqs = []
    rem_lfp = []
    all_masked_freqs = []
    for ii in range(len(interval)):
        start_idx = int(interval.loc[ii, 'start'] * fs)
        end_idx = int(interval.loc[ii, 'end'] * fs)
        sig_part = lfp[start_idx:end_idx]
        sig = np.array(sig_part)

        rem_lfp.append(sig)

        try:
            imf, mask_freq = sift.mask_sift(sig, **config)
        except Exception as e:
            print(f"EMD Sift failed: {e}. Skipping this interval.")
            continue
        all_imfs.append(imf)
        all_masked_freqs.append(mask_freq)

        imf_frequencies = imf_freq(imf, fs)
        all_imf_freqs.append(imf_frequencies)

    if return_imfs_freqs:
        return all_imfs, all_imf_freqs, rem_lfp
    else:
        return all_imfs


def tg_split(mask_freq, theta_range=(5, 12)):
    """
        Split a frequency vector into sub-theta, theta, and supra-theta components.

        Parameters:
        mask_freq (numpy.ndarray): A frequency vector or array of frequency values.
        theta_range (tuple, optional): A tuple defining the theta frequency range (lower, upper).
            Default is (5, 12).

        Returns:
        tuple: A tuple containing boolean masks for sub-theta, theta, and supra-theta frequency components.

        Notes: - This function splits a frequency mask into three components based on a specified theta frequency
        range. - The theta frequency range is defined by the 'theta_range' parameter. - The resulting masks 'sub',
        'theta', and 'supra' represent sub-theta, theta, and supra-theta frequency components.
    """
    lower = np.min(theta_range)
    upper = np.max(theta_range)
    mask_index = np.logical_and(mask_freq >= lower, mask_freq < upper)
    sub_mask_index = mask_freq < lower
    supra_mask_index = mask_freq > upper
    sub = sub_mask_index
    theta = mask_index
    supra = supra_mask_index

    return sub, theta, supra


def compute_range(x):
    return x.max() - x.min()


def asc2desc(x):
    pt = emd.cycles.cf_peak_sample(x, interp=True)
    tt = emd.cycles.cf_trough_sample(x, interp=True)
    if (pt is None) or (tt is None):
        return np.nan
    asc = pt + (len(x) - tt)
    desc = tt - pt
    return asc / len(x)


def peak2trough(x):
    des = emd.cycles.cf_descending_zero_sample(x, interp=True)
    if des is None:
        return np.nan
    return des / len(x)


def extract_subsets(arr, max_size):
    subsets = []
    current_subset = [arr[0]]

    for i in range(1, len(arr)):
        if arr[i] == arr[i - 1] + 1 and len(current_subset) < max_size:
            current_subset.append(arr[i])
        else:
            subsets.append(current_subset)
            current_subset = [arr[i]]

    # Add the last subset
    if current_subset:
        subsets.append(current_subset)

    return subsets


def bin_tf_to_fpp(x, power, bin_count):
    """
       Bin time-frequency power data into Frequency Phase Power (FPP) plots using specified time intervals of cycles.

       Parameters:
       x (numpy.ndarray): A 1D or 2D array specifying time intervals of cycles for binning.
           - If 1D, it represents a single time interval [start, end].
           - If 2D, it represents multiple time intervals, where each row is [start, end].
       power (numpy.ndarray): The time-frequency power spectrum data to be binned.
       bin_count (int): The number of bins to divide the time intervals into.

       Returns:
       fpp(numpy.ndarray): Returns FPP plots

       Notes:
       - This function takes time-frequency power data and divides it into FPP plots based on specified
         time intervals.
       - The 'x' parameter defines the time intervals, which can be a single interval or multiple intervals.
       - The 'power' parameter is the time-frequency power data to be binned.
       - The 'bin_count' parameter determines the number of bins within each time interval.
       """

    if x.ndim == 1:  # Handle the case when x is of size (2)
        bin_ranges = np.arange(x[0], x[1], 1)
        fpp = binned_statistic(
            bin_ranges, power[:, x[0]:x[1]], 'mean', bins=bin_count)[0]
        # Add an extra dimension to match the desired output shape
        fpp = np.expand_dims(fpp, axis=0)
    elif x.ndim == 2:  # Handle the case when x is of size (n, 2)
        fpp = []
        for i in range(x.shape[0]):
            bin_ranges = np.arange(x[i, 0], x[i, 1], 1)
            fpp_row = binned_statistic(
                bin_ranges, power[:, x[i, 0]:x[i, 1]], 'mean', bins=bin_count)[0]
            fpp.append(fpp_row)
        fpp = np.array(fpp)
    else:
        raise ValueError("Invalid size for x")

    return fpp


def plot_cycles(imf, sig, ctrl, inds):
    xinds = np.arange(len(inds))
    plt.figure(figsize=(8, 6))
    plt.plot(xinds, sig[inds], color=[0.8, 0.8, 0.8], label="Raw LFP")
    theta_part = imf[inds, 5]
    plt.plot(xinds, theta_part, label="IMF-6")

    plt.scatter(ctrl, theta_part[ctrl], color='red',
                marker='o', label='Control Points')
    plt.ylim([-800, 800])
    plt.legend()
    plt.show()


def load_mat_data(path_to_data, file_name, states_file):
    data = loadmat(path_to_data + file_name)
    data = data['PFClfpCleaned'].flatten()

    states = loadmat(path_to_data + states_file)
    states = states['states'].flatten()
    return data, states


def get_first_NREM_epoch(arr, start):
    start_index = None
    for i in range(start, len(arr)):
        if arr[i] == 3:
            if start_index is None:
                start_index = i
        elif arr[i] != 3 and start_index is not None:
            return (start_index, i - 1, i)

    return (start_index, len(arr) - 1, len(arr)) if start_index is not None else None


def get_all_NREM_epochs(arr):
    nrem_epochs = []
    next_start = 0
    while next_start < len(arr)-1:
        indices = get_first_NREM_epoch(arr, next_start)
        if indices == None:
            break
        start, end, next_start = indices
        if end-start <= 30:
            continue
        nrem_epochs.append([start, end])
    return nrem_epochs


def get_filtered_epoch_data(data, epochs, band=(0.1, 4), fs=2500):
    epoch_data = []
    for start, end in epochs:
        data_part = data[start*fs:end*fs]
        epoch_data.extend(data_part)
    epoch_data = np.array(epoch_data)
    filtered_epoch_data = filter_signal(
        epoch_data, fs, 'bandpass', band, n_cycles=3, filter_type='iir', butterworth_order=6, remove_edges=False)
    return filtered_epoch_data, epoch_data


def get_cycles_with_conditions(cycles, conditions):
    C = copy.deepcopy(cycles)
    try:
        C.pick_cycle_subset(conditions)
    except ValueError as e:
        print(f"No cycles satisfy the conditions: {e}")
        return None
    return C


def peak_before_trough(arr):
    trough_val = np.min(arr)
    trough_pos = np.argmin(arr)
    for i in range(trough_pos - 1, 0, -1):
        if arr[i] > arr[i - 1] and arr[i] > arr[i + 1] and arr[i] >= 0:
            return arr[i]
    return -1


def peak_before_trough_pos(arr):
    trough_val = np.min(arr)
    trough_pos = np.argmin(arr)
    for i in range(trough_pos - 1, 0, -1):
        if arr[i] > arr[i - 1] and arr[i] > arr[i + 1] and arr[i] >= 0:
            return i
    return -1


def peak_to_trough_duration(arr):
    trough_val = np.min(arr)
    trough_pos = np.argmin(arr)
    for i in range(trough_pos - 20, 0, -1):
        if arr[i] > arr[i - 1] and arr[i] > arr[i + 1] and arr[i] >= 0:
            return trough_pos-i
    return -1


def num_inflection_points(arr):
    sign_changes = np.diff(np.sign(np.diff(arr, 2)))
    num_inflection_points = np.sum(sign_changes != 0)
    return num_inflection_points


def get_cycles_with_metrics(cycles, data, IA, IF, conditions=None):
    C = copy.deepcopy(cycles)

    C.compute_cycle_metric('duration_samples', data,
                           func=len, mode='augmented')
    C.compute_cycle_metric('peak2trough', data,
                           func=peak2trough, mode='augmented')
    C.compute_cycle_metric('asc2desc', data, func=asc2desc, mode='augmented')
    C.compute_cycle_metric('max_amp', IA, func=np.max, mode='augmented')
    C.compute_cycle_metric('trough_values', data,
                           func=np.min, mode='augmented')
    C.compute_cycle_metric('peak_values', data, func=np.max, mode='augmented')
    C.compute_cycle_metric('mean_if', IF, func=np.mean, mode='augmented')
    C.compute_cycle_metric('max_if', IF, func=np.max, mode='augmented')
    C.compute_cycle_metric(
        'range_if', IF, func=compute_range, mode='augmented')
    C.compute_cycle_metric('trough_position', data,
                           func=np.argmin, mode='augmented')
    C.compute_cycle_metric('peak_position', data,
                           func=np.argmax, mode='augmented')

    return C


def get_cycle_inds(cycles, subset_indices):

    all_cycles_inds = []
    for idx in subset_indices:
        if idx != -1:
            inds = cycles.get_inds_of_cycle(idx, mode='augmented')
            all_cycles_inds.append(inds)
    return all_cycles_inds


def get_cycle_ctrl(ctrl, subset_indices):
    all_cycles_ctrl = []
    for idx in subset_indices:
        if idx != -1:
            ctrl_inds = np.array(ctrl[idx], dtype=int)
            all_cycles_ctrl.append(ctrl_inds)
    return all_cycles_ctrl


def arrange_cycle_inds(all_cycles_inds):
    cycles_inds = []
    for ii in range(len(all_cycles_inds)):
        cycle = all_cycles_inds[ii]
        start = cycle[0]
        end = cycle[-1]
        cycles_inds.append([start, end])

    cycles_inds = np.array(cycles_inds)

    return cycles_inds


def compute_mode_frequency_and_entropy(FPP, frequencies, angles):
    mode_frequencies = []
    enropy_values = []

    for fpp in FPP:

        fpp = np.abs(fpp)

        fpp2_sum = np.sum(fpp)
        normalized_fpp2 = fpp / fpp2_sum

        max_index = np.unravel_index(
            np.argmax(normalized_fpp2, axis=None), fpp.shape)
        mode_frequency = frequencies[max_index[0]]

        avg_fpp2 = np.sum(normalized_fpp2, axis=1)
        window_size = 5
        smoothed_avg_fpp = np.convolve(avg_fpp2, np.ones(
            window_size)/window_size, mode='same')

        smoothed_avg_fpp_norm = (smoothed_avg_fpp - np.min(smoothed_avg_fpp)) / \
            (np.max(smoothed_avg_fpp) - np.min(smoothed_avg_fpp))
        dist_smoothed_avg_fpp_norm = smoothed_avg_fpp_norm / \
            np.sum(smoothed_avg_fpp_norm)

        shannon_entropy = entropy(dist_smoothed_avg_fpp_norm, base=2)
        enropy_values.append(shannon_entropy)

        mode_frequencies.append(mode_frequency)

    return np.array(mode_frequencies), np.array(enropy_values)


def abids(X, k):
    search_struct = cKDTree(X)
    return np.array([abid(X, k, x, search_struct) for x in X])


def abid(X, k, x, search_struct, offset=1):
    neighbor_norms, neighbors = search_struct.query(x, k+offset)
    neighbors = X[neighbors[offset:]] - x
    normed_neighbors = neighbors / neighbor_norms[offset:, None]
    # Original publication version that computes all cosines
    # coss = normed_neighbors.dot(normed_neighbors.T)
    # return np.mean(np.square(coss))**-1
    # Using another product to get the same values with less effort
    para_coss = normed_neighbors.T.dot(normed_neighbors)
    return k**2 / np.sum(np.square(para_coss))

def extract_experiment_info(path_to_hpc):

    path_parts = os.path.normpath(path_to_hpc).split(os.sep)

    try:
        idx_for_abdel = path_parts.index('for Abdel')
    except ValueError:
        raise ValueError("The path does not contain 'for Abdel' directory.")

    dataset_type = path_parts[idx_for_abdel + 1]

    rat_number = path_parts[idx_for_abdel + 2]

    treatment_part = path_parts[idx_for_abdel + 3]
    if '_' not in treatment_part and ' ' not in treatment_part:

        treatment = treatment_part
    else:

        tokens = re.split(r'[_\-]', treatment_part)

        tokens = [t for t in tokens if not re.match(r'Rat\d*|SD\d*|Rat|Ephys|OS', t, re.IGNORECASE)]

        non_numeric_tokens = [t for t in tokens if not t.isdigit()]
        if non_numeric_tokens:

            treatment = non_numeric_tokens[-1]
        else:
            treatment = 'Unknown'

    post_trial_folder = path_parts[-2]
    post_trial_match = re.search(r'post_trial(\d+)', post_trial_folder, re.IGNORECASE)
    if post_trial_match:
        post_trial = post_trial_match.group(1)
    else:
        post_trial = 'Unknown'

    return {
        'dataset_type': dataset_type,
        'rat_number': rat_number,
        'treatment': treatment,
        'post_trial': post_trial
    }

