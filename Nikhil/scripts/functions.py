from scipy.io import loadmat
import numpy as np
from neurodsp.filt import filter_signal
from scipy.spatial import cKDTree
import copy
import emd


# Function to load LFP data from .mat files
# The input includes both the LFP datafile and the sleep-scoring states file
# The LFP data should be cleaned using the matlab artifact cleaning
def load_mat_data(path_to_data, file_name, states_file):
    data = loadmat(path_to_data + file_name)
    data = data['PFClfpCleaned'].flatten()

    states = loadmat(path_to_data + states_file)
    states = states['states'].flatten()
    return data, states


# extract first nrem epochs from the sleep-scoring states file
def get_first_NREM_epoch(arr, start):
  start_index = None
  for i in range(start, len(arr)):
    if arr[i] == 3:
      if start_index is None:
        start_index = i
    elif arr[i] != 3 and start_index is not None:
      return (start_index, i - 1, i)

  return (start_index, len(arr) - 1, len(arr)) if start_index is not None else None


# extract all nrem epochs from the sleep-scoring states file
# Uses get_first_NREM_epoch() function
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


# Concatenating all the NREM epochs and filtering on the Delta band (0.1-4 Hz)
def get_nrem_filtered(pfc_data, nrem_epochs, fs):
    nrem_data = []
    for start, end in nrem_epochs:
        pfc_data_part = pfc_data[start*fs:end*fs]
        nrem_data.extend(pfc_data_part)
    nrem_data = np.array(nrem_data)
    nrem_filtered_data = filter_signal(nrem_data, fs, 'bandpass', (0.1, 4), n_cycles=3, filter_type='iir', butterworth_order=6, remove_edges=False)
    return nrem_filtered_data, nrem_data
  
  
def get_filtered_epoch_data(data, epochs, band=(0.1, 4), fs=2500):
    epoch_data = []
    for start, end in epochs:
        data_part = data[start*fs:end*fs]
        epoch_data.extend(data_part)
    epoch_data = np.array(epoch_data)
    filtered_epoch_data = filter_signal(epoch_data, fs, 'bandpass', band, n_cycles=3, filter_type='iir', butterworth_order=6, remove_edges=False)
    return filtered_epoch_data, epoch_data



# Get cycle filtered on conditions. Make a copy so the original is not affected.
def get_cycles_with_conditions(cycles, conditions):
    C = copy.deepcopy(cycles)
    C.pick_cycle_subset(conditions)
    return C


# Functions to calculate metrics
def peak_before_trough(arr):
  trough_val = np.min(arr)
  trough_pos = np.argmin(arr)
  for i in range(trough_pos - 1, 0, -1):
    if arr[i] > arr[i - 1] and arr[i] > arr[i + 1] and arr[i]>=0:
      return arr[i]
  return -1

def peak_before_trough_pos(arr):
  trough_val = np.min(arr)
  trough_pos = np.argmin(arr)
  for i in range(trough_pos - 1, 0, -1):
    if arr[i] > arr[i - 1] and arr[i] > arr[i + 1] and arr[i]>=0:
      return i
  return -1

def peak_to_trough_duration(arr):
  trough_val = np.min(arr)
  trough_pos = np.argmin(arr)
  for i in range(trough_pos - 20, 0, -1):
    if arr[i] > arr[i - 1] and arr[i] > arr[i + 1] and arr[i]>=0:
      return trough_pos-i
  return -1

def num_inflection_points(arr):
  sign_changes = np.diff(np.sign(np.diff(arr, 2)))
  num_inflection_points = np.sum(sign_changes != 0)
  return num_inflection_points

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

# Compute metrics for each cycle -
#     Maximum Amplitude
#     Cycle Duration
#     Trough Position and values
#     Peak (just before the trough) Position and Values
#     Peak Position and Values
#     Peak-to-Trough Duration
#     Peak to trough ratio ( P / P+T )
#     Ascending to Descending ratio ( A / A+D )
def get_cycles_with_metrics(cycles, data, IA, IF, conditions=None):
  C = copy.deepcopy(cycles)

  C.compute_cycle_metric('duration_samples', data, func=len)
  C.compute_cycle_metric('peak2trough', data, func=peak2trough)
  C.compute_cycle_metric('asc2desc', data, func=asc2desc)
  C.compute_cycle_metric('max_amp', IA, func=np.max)
  C.compute_cycle_metric('trough_values', data, func=np.min)
  C.compute_cycle_metric('peak_values', data, func=np.max)
  C.compute_cycle_metric('mean_if', IF, func=np.mean)
  C.compute_cycle_metric('max_if', IF, func=np.max)
  C.compute_cycle_metric('range_if', IF, func=compute_range)

  C.compute_cycle_metric('trough_position', data, func=np.argmin)
  C.compute_cycle_metric('peak_position', data, func=np.argmax)
  return C
  
  
def get_delta_cycles(pfc_data, sleep_scoring, fs=1000):
    # Get filtered NREM LFP data
    nrem_epochs = np.array(get_all_NREM_epochs(sleep_scoring))
    if len(nrem_epochs) == 0:
        return None
    nrem_filtered_data, nrem_data = get_nrem_filtered(pfc_data, nrem_epochs, fs=fs)

    # Get cycles using IP
    IP, IF, IA = emd.spectra.frequency_transform(nrem_filtered_data, fs, 'hilbert')
    C = emd.cycles.Cycles(IP)
    cycles = get_cycles_with_metrics(C, nrem_filtered_data, IA, IF)
    
    delta_cycle_data = {"fs": fs, 'nrem_epochs': nrem_epochs, 'nrem_filtered_data': nrem_filtered_data,
                       "IP": IP, "IF": IF, "IP": IP, "cycles": cycles}
    return delta_cycle_data


#  Gives cycle vector using a mask, it creates a mask with subset of cycles
def get_masked_cycles(IP, cycles):
  mask = np.full(cycles.nsamples, False)
  subset_cycles = cycles.get_metric_dataframe(subset=True)['index']

  for i in subset_cycles:
    inds = cycles.get_inds_of_cycle(i)
    mask[inds] = True

  masked_so_cycles = emd.cycles.get_cycle_vector(IP, mask=mask)
  return masked_so_cycles


# 
def rate_cycle(cycles_vector, duration=1, fs=1000):
  samples_per_segment = duration * fs
  segments = np.array_split(cycles_vector, np.arange(samples_per_segment, len(cycles_vector), samples_per_segment))
  segments = np.array(segments[:-1])

  rate = []
  for segment in segments:
    if -1 in segment:
      rate.append(len(np.unique(segment))-1)
    else:
      rate.append(len(np.unique(segment)))
  rate = np.array(rate)
  rate = rate/duration
  return rate, segments


def check_at_ends(cycle_vector, start_point, at_start=True):
    cycle_num = cycle_vector[start_point]
    
    end = start_point
    backward_len = 1
    while cycle_vector[end] != -1 and cycle_vector[end] == cycle_num:
        backward_len += 1
        end -= 1

    start = start_point
    forward_len = 0
    while cycle_vector[start] != -1 and cycle_vector[start] == cycle_num:
        forward_len += 1
        start += 1
    if at_start:
        result = forward_len/(backward_len + forward_len), start
    else:
        result = backward_len/(backward_len + forward_len), end
    return result


# Get rate of cycles in a cycle vector
def rate_cycle_partial_ends(cycle_vector, duration=10, fs=1000):
    num_of_segments = len(cycle_vector)//(fs*duration)
    rates = []
    segment = 0

    for i in range(num_of_segments):
        rate = 0
        start = i*fs*duration
        end = (i+1)*fs*duration
        if cycle_vector[start] != -1:
            cycle_start_part, start = check_at_ends(cycle_vector, start, at_start=True)
            rate += cycle_start_part
        if cycle_vector[end] != -1:
            cycle_end_part, end = check_at_ends(cycle_vector, end, at_start=False)
            rate += cycle_end_part
        segment = cycle_vector[start:end]
        if -1 in segment:
            rate += len(np.unique(segment))-1
        else:
            rate += len(np.unique(segment))
        rate = round(rate/duration, 2)
        rates.append(rate)
    return rates

def abids(X,k):
    search_struct = cKDTree(X)
    return np.array([abid(X,k,x,search_struct) for x in X])


def abid(X,k,x,search_struct,offset=1):
    neighbor_norms, neighbors = search_struct.query(x,k+offset)
    neighbors = X[neighbors[offset:]] - x
    normed_neighbors = neighbors / neighbor_norms[offset:,None]
    # Original publication version that computes all cosines
    # coss = normed_neighbors.dot(normed_neighbors.T)
    # return np.mean(np.square(coss))**-1
    # Using another product to get the same values with less effort
    para_coss = normed_neighbors.T.dot(normed_neighbors)
    return k**2 / np.sum(np.square(para_coss))