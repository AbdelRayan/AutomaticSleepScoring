import numpy as np
import h5py
from scipy.stats import mode
from scipy.io import loadmat
import os


from EMG_buzsakiMethod import compute_emg_buzsakiMethod
from computing_features import psd_multitaper, wei_normalizing, index_W, index_N, index_R, Index_1, Index_2, Index_3, Index_4
from Artefacts_Detection import removeArtefacts, artefact_epochs

def getNewFeatures(raw_hpc, raw_pfc, states, fs, epoch_length):
  """
  Computes new features from raw data.

  Parameters:
      raw_hpc (numpy.ndarray): The data from the hippocampus.
      raw_pfc (numpy.ndarray): The data from the prefrontal cortex.
      states (numpy.ndarray): The sleep states.
      fs (float) : the sampling frequency.
      epoch_length (int) : the length of an epoch in seconds.

  Returns:
      new_features (numpy.ndarray): The computed features.
      mapped_scores (numpy.ndarray): The mapped sleep scores.

  Notes:
      This function first computes the EMG from the raw data. 
      It then computes the power spectral density (PSD) in different frequency bands for the raw data. 
      The PSDs are normalized and smoothed to be used as features. 
      The function also computes several indices from the PSDs, normalizes and smooths them, and uses them as features. 
      The function returns a matrix of these features along with the mapped sleep scores.
  """

  # Getting EMG
  emg_sampling = 5
  smoothWindow = fs
  EMGFromLFP = compute_emg_buzsakiMethod(emg_sampling,fs,raw_pfc,raw_hpc,smoothWindow)
  EMG = EMGFromLFP['data']
  emg_mat_norm = EMGFromLFP['Norm']
  emg_mat_smoothed = EMGFromLFP['smoothed']
  if len(EMG) % 2 == 1:
    EMG = np.append(EMG, EMG[-1])
  EMG =EMG[:len(EMG) // (epoch_length*emg_sampling) * (epoch_length*emg_sampling)]
  EMG = EMG.reshape(-1, (epoch_length*emg_sampling))
  EMG = EMG.sum(axis=1)

  # Get mapped scores
  sleep_scoring = np.ravel(states)
  reshaped_scores = sleep_scoring[:len(sleep_scoring) // epoch_length * epoch_length].reshape(-1, epoch_length)
  majority_scores = mode(reshaped_scores, axis=1).mode.flatten()
  mapped_scores = np.array(majority_scores)

  #Frequency ranges
  noise_band = [0,0.5]
  delta_band = [0.5,5]
  theta_band = [6,10]
  sigma_band = [11,17]
  beta_band = [22,30]
  gamma_band = [35,45]
  total_band = [0,30]

  window_length = fs*epoch_length
  # Get powers
  noise = psd_multitaper(np.ravel(raw_hpc), fs, noise_band, window_length)
  delta = psd_multitaper(np.ravel(raw_pfc), fs, delta_band, window_length)
  theta = psd_multitaper(np.ravel(raw_hpc), fs, theta_band, window_length)
  sigma = psd_multitaper(np.ravel(raw_pfc), fs, sigma_band, window_length)
  gamma = psd_multitaper(np.ravel(raw_pfc), fs, gamma_band, window_length)

  # Normalize all these powers
  noise_norm = wei_normalizing(noise)
  delta_norm = wei_normalizing(delta)
  theta_norm = wei_normalizing(theta)
  sigma_norm = wei_normalizing(sigma)
  gamma_norm = wei_normalizing(gamma)
  EMG_norm = wei_normalizing(EMG)

  # Get smoothed powers as feature
  noise_smoothed = np.convolve(np.convolve(np.convolve(noise_norm, np.ones(5)/5, mode='same'), np.ones(5)/5, mode='same'), np.ones(5)/5, mode='same')
  theta_smoothed = np.convolve(np.convolve(np.convolve(theta_norm, np.ones(5)/5, mode='same'), np.ones(5)/5, mode='same'), np.ones(5)/5, mode='same')
  delta_smoothed = np.convolve(np.convolve(np.convolve(delta_norm, np.ones(5)/5, mode='same'), np.ones(5)/5, mode='same'), np.ones(5)/5, mode='same')

  # Compute new indices
  index_w = index_W(theta_norm, gamma_norm, EMG_norm)
  index_n = index_N(delta_norm, sigma_norm, gamma_norm)
  index_r = index_R(delta_norm, theta_norm, sigma_norm, EMG_norm, gamma_norm)
  index_1 = Index_1(delta_norm, gamma_norm, EMG_norm)
  index_2 = Index_2(delta_norm, theta_norm, sigma_norm)
  index_3 = Index_3(delta_norm, theta_norm, gamma_norm)
  index_4 = Index_4(delta_norm, theta_norm)

  # Take log of indices
  index_w_log = np.log(index_w)
  index_n_log = np.log(index_n)
  index_r_log = np.log(index_r)
  index_1_log = np.log(index_1)
  index_2_log = np.log(index_2)
  index_3_log = np.log(index_3)
  index_4_log = np.log(index_4)

  # Normalise indices
  index_w_norm = wei_normalizing(index_w_log)
  index_n_norm = wei_normalizing(index_n_log)
  index_r_norm = wei_normalizing(index_r_log)
  index_1_norm = wei_normalizing(index_1_log)
  index_2_norm = wei_normalizing(index_2_log)
  index_3_norm = wei_normalizing(index_3_log)
  index_4_norm = wei_normalizing(index_4_log)

  # Smooth indices
  index_w_smoothed = np.convolve(np.convolve(np.convolve(index_w_norm, np.ones(5)/5, mode='same'), np.ones(5)/5, mode='same'), np.ones(5)/5, mode='same')
  index_n_smoothed = np.convolve(np.convolve(np.convolve(index_n_norm, np.ones(5)/5, mode='same'), np.ones(5)/5, mode='same'), np.ones(5)/5, mode='same')
  index_r_smoothed = np.convolve(np.convolve(np.convolve(index_r_norm, np.ones(5)/5, mode='same'), np.ones(5)/5, mode='same'), np.ones(5)/5, mode='same')
  index_1_smoothed = np.convolve(np.convolve(np.convolve(index_1_norm, np.ones(5)/5, mode='same'), np.ones(5)/5, mode='same'), np.ones(5)/5, mode='same')
  index_2_smoothed = np.convolve(np.convolve(np.convolve(index_2_norm, np.ones(5)/5, mode='same'), np.ones(5)/5, mode='same'), np.ones(5)/5, mode='same')
  index_3_smoothed = np.convolve(np.convolve(np.convolve(index_3_norm, np.ones(5)/5, mode='same'), np.ones(5)/5, mode='same'), np.ones(5)/5, mode='same')
  index_4_smoothed = np.convolve(np.convolve(np.convolve(index_4_norm, np.ones(5)/5, mode='same'), np.ones(5)/5, mode='same'), np.ones(5)/5, mode='same')


  # Create matrix
  new_features = np.column_stack((index_w_smoothed, index_r_smoothed, index_n_smoothed, index_1_smoothed, index_2_smoothed, index_3_smoothed, index_4_smoothed, noise_smoothed, theta_smoothed, delta_smoothed))
  return new_features, mapped_scores

def prepare_for_hdf5(recording, fs, files_path, epoch_length):
  """
  Prepares data for HDF5 format.
  
  Parameters:
      recording (list): The list containing names of the HPC, PFC, and states data files.
      fs (float) : The sampling frequency.
      files_path (str) : The path that leads to the files.
      epoch_length (int) : the length of an epoch in seconds.
  
  Returns:
      Features (numpy.ndarray): The computed features from HPC and PFC data. It consists of a list of 10 indices for each epoch.
      Mapped_scores (numpy.ndarray): The mapped sleep scores after adding artefact epochs.
      recording_name (str): The name of the recording group.
  """
  # Get the right data
  for i in recording:
    if 'HPC' in i:
      hpc = i
    if 'PFC' in i:
      pfc = i
    if 'states' in i:
      states = i
  recording_name = hpc[:-4]     # Group name 
  path_to_hpc = files_path + "/" + hpc
  # Prepare data (Load + artefact removal)
  hpc_data = loadmat(path_to_hpc)
  hpc_data = hpc_data['HPC']
  hpc_data = hpc_data[8*fs:]
  sighpc = removeArtefacts(hpc_data, fs, [9,8], [0.2,0.1])
  hpc_filt = np.ravel(sighpc[0])
  hpc_artefact_indexes = np.ravel(sighpc[1])                 # Get the indexes of epochs containing artefacts
  pfc_data = loadmat(os.path.join(files_path, pfc))
  pfc_data = pfc_data['PFC']
  pfc_data = pfc_data[8*fs:]
  sigpfc = removeArtefacts(pfc_data, fs, [9,8], [0.2,0.1])
  pfc_filt = np.ravel(sigpfc[0])
  pfc_artefact_indexes = np.ravel(sigpfc[1])

  sleep_scoring = loadmat(os.path.join(files_path, states))
  states = sleep_scoring['states'][0][7:]
  print(f"len(hpc_data) : {len(hpc_data)}")
  print(f"len(states) : {len(states)}")


  # Create matrix for specific set of recordings
  a = getNewFeatures(hpc_filt, pfc_filt, states, fs, epoch_length)
  Features = a[0]
  Mapped_scores = a[1]

  # Add the artefact epochs to mapped scores
  window_length = fs * epoch_length
  hpc_arte_epochs = artefact_epochs(hpc_artefact_indexes, window_length)
  pfc_arte_epochs = artefact_epochs(pfc_artefact_indexes, window_length)
  artefact_indices = np.unique(np.concatenate((hpc_arte_epochs, pfc_arte_epochs)))
  artefact_indices = artefact_indices.astype(int)
  Mapped_scores[artefact_indices] = 0 
  print("##################")
  print(len(Mapped_scores))
  print(len(hpc_filt))
  print("##################")
  return(Features, Mapped_scores, recording_name)

def update_hdf5(a, path_to_hdf5):
  """
  Updates an HDF5 file with the data of a recording (features and mapped scores)/
  
  Parameters:  
      a (tuple): A tuple containing the features, mapped scores, and the recording name.
      path_to_hdf5 (str): The path to the HDF5 file to be updated.
  
  Notes:
      This function opens the HDF5 file at the given path in append mode.
      It creates a new group in the file with the name of the recording.
      The group is given two attributes: 'Description features' and 'Description Mapped_scores', 
      which describe the features and mapped scores respectively.
      Two datasets, 'Features' and 'Mapped_scores', are created in the group using the data from the input tuple.
  """
  # Add the data to the hdf5 file
  with h5py.File(path_to_hdf5, 'a')  as database:
  # Create group and 2 datasets
    print(path_to_hdf5)
    print(group_name)
    group = database.create_group(a[2])
    group.attrs['Description features'] = '[index_w_smoothed, index_r_smoothed, index_n_smoothed, index_1_smoothed, index_2_smoothed, index_3_smoothed, index_4_smoothed, noise_smoothed, theta_smoothed, delta_smoothed]'
    group.attrs['Description Mapped_scores'] = '[0: Artefact, 1: Wake, 3: NREM, 4: Intermediate, 5: REM]'
    group.create_dataset('Features', data = a[0])
    group.create_dataset('Mapped_scores', data = a[1])
