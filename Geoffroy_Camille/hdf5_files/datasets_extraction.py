import h5py
import numpy as np
import random

def training_dataset_OS_RGS_CBD(file_path, n_states, n_repeat_states, rat_indices_OS, rat_indices_RGS, rat_indices_CBD):
    """
    Extract a training dataset from the OS_RGS_CBD.h5 file.
    
    Parameters : 
    file_path (str): the file_path to the OS_RGS_CBD.h5 file
    n_states (int array): an array of 5 integers : [n_artefact, n_wake, n_nrem, n_TS, n_rem]. Each integer corresponds to the number of wanted epochs for the given state.
    n_repeat_states (int array): an array of 5 integers : [n_repeat_artefact, n_repeat_wake, n_repeat_nrem, n_repeat_TS, n_repeat_rem]. 
                      Each integer corresponds to the number of times the epochs of the given state will be repeated in the training dataset.
    rat_indices_OS (int array): the indices of the rats from the OS_Basic dataset that would be included in the training dataset. #Included in [1, 3, 4, 6, 9, 11, 13]
    rat_indices_RGS (int array): the indices of the rats from the RGS14 dataset that would be included in the training dataset. #Included in [1, 2, 3, 4, 6, 7, 8, 9] (RGS control : [1, 2, 6, 9] and RGS positive : [3, 4, 7, 8])
    rat_indices_CBD (int array): the indices of the rats from the CBD dataset that would be included in the training dataset. #Included in [3, 4, 5, 6]

    Returns :
    float numpy.array : an array containing the features for each epoch of the training dataset.
    int numpy.array : an array containing the manual sleep state scoring of each epoch of the training dataset.
    """
    n_artefact, n_wake, n_nrem, n_TS, n_rem = n_states
    n_repeat_artefact, n_repeat_wake, n_repeat_nrem, n_repeat_TS, n_repeat_rem = n_repeat_states
    training_dataset = np.empty((0, 10))
    manual_scoring = np.array([])

    array_scores_total = np.array([])
    array_features_total = np.empty((0, 10))
    indices_artefact_total = np.array([])
    indices_wake_total = np.array([])
    indices_nrem_total = np.array([])
    indices_TS_total = np.array([])
    indices_rem_total = np.array([])
    with h5py.File(file_path, 'r') as hdf:
      for i in rat_indices_OS:
        array_scores = np.array([])
        array_features = np.empty((0, 10))
        for group_name in hdf["OS_basic"][f"Rat{i}"]:
            group = hdf["OS_basic"][f"Rat{i}"][group_name]
            array_scores = np.concatenate((array_scores, group['Mapped_scores'][:]), axis = 0)
            array_features = np.concatenate((array_features, group['Features'][:]), axis = 0)
        indices_artefact = np.where(array_scores == 0)[0] # Artefact
        indices_wake = np.where(array_scores == 1)[0] # Wake
        indices_nrem = np.where(array_scores == 3)[0] # NREM
        indices_TS = np.where(array_scores  == 4)[0] #TS
        indices_rem = np.where(array_scores == 5)[0] # REM
        len_scores = len(array_scores_total)

        array_scores_total = np.concatenate((array_scores_total, array_scores), axis = 0)
        array_features_total = np.concatenate((array_features_total, array_features), axis = 0)
        indices_artefact = np.array([el + len_scores for el in indices_artefact])
        indices_wake = np.array([el + len_scores for el in indices_wake])
        indices_nrem = np.array([el + len_scores for el in indices_nrem])
        indices_TS = np.array([el + len_scores for el in indices_TS])
        indices_rem = np.array([el + len_scores for el in indices_rem])
        indices_artefact_total = np.concatenate((indices_artefact_total, indices_artefact), axis = 0)
        indices_wake_total = np.concatenate((indices_wake_total, indices_wake), axis = 0)
        indices_nrem_total = np.concatenate((indices_nrem_total, indices_nrem), axis = 0)
        indices_TS_total = np.concatenate((indices_TS_total, indices_TS), axis = 0)
        indices_rem_total = np.concatenate((indices_rem_total, indices_rem), axis = 0)

      for i in rat_indices_RGS:
        array_scores = np.array([])
        array_features = np.empty((0, 10))
        for group_name in hdf["RGS14"][f"Rat{i}"]:
            group = hdf["RGS14"][f"Rat{i}"][group_name]
            array_scores = np.concatenate((array_scores, group['Mapped_scores'][:]), axis = 0)
            array_features = np.concatenate((array_features, group['Features'][:]), axis = 0)
        indices_artefact = np.where(array_scores == 0)[0] # Artefact
        indices_wake = np.where(array_scores == 1)[0] # Wake
        indices_nrem = np.where(array_scores == 3)[0] # NREM
        indices_TS = np.where(array_scores  == 4)[0] #TS
        indices_rem = np.where(array_scores == 5)[0] # REM
        len_scores = len(array_scores_total)

        array_scores_total = np.concatenate((array_scores_total, array_scores), axis = 0)
        array_features_total = np.concatenate((array_features_total, array_features), axis = 0)
        indices_artefact = np.array([el + len_scores for el in indices_artefact])
        indices_wake = np.array([el + len_scores for el in indices_wake])
        indices_nrem = np.array([el + len_scores for el in indices_nrem])
        indices_TS = np.array([el + len_scores for el in indices_TS])
        indices_rem = np.array([el + len_scores for el in indices_rem])
        indices_artefact_total = np.concatenate((indices_artefact_total, indices_artefact), axis = 0)
        indices_wake_total = np.concatenate((indices_wake_total, indices_wake), axis = 0)
        indices_nrem_total = np.concatenate((indices_nrem_total, indices_nrem), axis = 0)
        indices_TS_total = np.concatenate((indices_TS_total, indices_TS), axis = 0)
        indices_rem_total = np.concatenate((indices_rem_total, indices_rem), axis = 0)
      for i in rat_indices_CBD:
        array_scores = np.array([])
        array_features = np.empty((0, 10))
        for group_name in hdf["CBD"][f"Rat{i}"]:
            group = hdf["CBD"][f"Rat{i}"][group_name]
            array_scores = np.concatenate((array_scores, group['Mapped_scores'][:]), axis = 0)
            array_features = np.concatenate((array_features, group['Features'][:]), axis = 0)
        indices_artefact = np.where(array_scores == 0)[0] # Artefact
        indices_wake = np.where(array_scores == 1)[0] # Wake
        indices_nrem = np.where(array_scores == 3)[0] # NREM
        indices_TS = np.where(array_scores  == 4)[0] #TS
        indices_rem = np.where(array_scores == 5)[0] # REM
        len_scores = len(array_scores_total)
        array_scores_total = np.concatenate((array_scores_total, array_scores), axis = 0)
        array_features_total = np.concatenate((array_features_total, array_features), axis = 0)
        indices_artefact = np.array([el + len_scores for el in indices_artefact])
        indices_wake = np.array([el + len_scores for el in indices_wake])
        indices_nrem = np.array([el + len_scores for el in indices_nrem])
        indices_TS = np.array([el + len_scores for el in indices_TS])
        indices_rem = np.array([el + len_scores for el in indices_rem])
        indices_artefact_total = np.concatenate((indices_artefact_total, indices_artefact), axis = 0)
        indices_wake_total = np.concatenate((indices_wake_total, indices_wake), axis = 0)
        indices_nrem_total = np.concatenate((indices_nrem_total, indices_nrem), axis = 0)
        indices_TS_total = np.concatenate((indices_TS_total, indices_TS), axis = 0)
        indices_rem_total = np.concatenate((indices_rem_total, indices_rem), axis = 0)

    selected_indices_artefact = random.sample(list(indices_artefact_total), n_artefact)
    selected_indices_wake = random.sample(list(indices_wake_total), n_wake)
    selected_indices_nrem = random.sample(list(indices_nrem_total), n_nrem)
    selected_indices_TS = random.sample(list(indices_TS_total), n_TS)
    selected_indices_rem = random.sample(list(indices_rem_total), n_rem)
    selected_indices_artefact = np.ravel([[el]* n_repeat_artefact for el in selected_indices_artefact])
    selected_indices_wake = np.ravel([[el]* n_repeat_wake for el in selected_indices_wake])
    selected_indices_nrem = np.ravel([[el]* n_repeat_nrem for el in selected_indices_nrem])
    selected_indices_TS = np.ravel([[el]* n_repeat_TS for el in selected_indices_TS])
    selected_indices_rem = np.ravel([[el]* n_repeat_rem for el in selected_indices_rem])

    features_artefact = array_features_total[selected_indices_artefact.astype(int)]
    scores = array_scores_total[selected_indices_artefact.astype(int)]
    training_dataset = np.concatenate((training_dataset, features_artefact), axis=0)
    manual_scoring = np.concatenate((manual_scoring, scores), axis=0)

    features_wake = array_features_total[selected_indices_wake.astype(int)]
    scores = array_scores_total[selected_indices_wake.astype(int)]
    training_dataset = np.concatenate((training_dataset, features_wake), axis=0)          # Filling up training dataset
    manual_scoring = np.concatenate((manual_scoring, scores), axis=0)                     # Filling up corresponding manual scores

    features_nrem = array_features_total[selected_indices_nrem.astype(int)]
    scores = array_scores_total[selected_indices_nrem.astype(int)]
    training_dataset = np.concatenate((training_dataset, features_nrem), axis=0)
    manual_scoring = np.concatenate((manual_scoring, scores), axis=0)

    features_TS = array_features_total[selected_indices_TS.astype(int)]
    scores = array_scores_total[selected_indices_TS.astype(int)]
    training_dataset = np.concatenate((training_dataset, features_TS), axis=0)
    manual_scoring = np.concatenate((manual_scoring, scores), axis=0)

    features_rem = array_features_total[selected_indices_rem.astype(int)]
    scores = array_scores_total[selected_indices_rem.astype(int)]
    training_dataset = np.concatenate((training_dataset, features_rem), axis=0)
    manual_scoring = np.concatenate((manual_scoring, scores), axis=0)

    paired = list(zip(training_dataset, manual_scoring))

    # Shuffle the paired list
    random.shuffle(paired)

    # Unzip the shuffled, paired list back into features and scores
    training_dataset, manual_scoring = zip(*paired)
    # training_dataset = np.array()
    training_dataset = np.array(training_dataset)
    manual_scoring = np.array(manual_scoring)
  
    return training_dataset, manual_scoring.astype(int)

def dataset_OS_RGS_CBD(file_path, rat_indices_OS, rat_indices_RGS, rat_indices_CBD, states):
    """
    Extract a dataset from the OS_RGS_CBD.h5 file.
    
    Parameters : 
    file_path (str): the file_path to the OS_RGS_CBD.h5 file
    rat_indices_OS (int array): the indices of the rats from the OS_Basic dataset that would be included in the training dataset. #Included in [1, 3, 4, 6, 9, 11, 13]
    rat_indices_RGS (int array): the indices of the rats from the RGS14 dataset that would be included in the training dataset. #Included in [1, 2, 3, 4, 6, 7, 8, 9] (RGS control : [1, 2, 6, 9] and RGS positive : [3, 4, 7, 8])
    rat_indices_CBD (int array): the indices of the rats from the CBD dataset that would be included in the training dataset. #Included in [3, 4, 5, 6]
    states : (str array) : an array containing the names of the states wanted in the dataset #Included in  ["artefact", "Wake", "NREM", "TS", "REM]
    
    Returns :
    float numpy.array : an array containing the features for each epoch of the dataset.
    int numpy.array : an array containing the manual sleep state scoring of each epoch of the dataset.

    Notes:
    The dataset obtained is not in a chronological order.
    """
    dataset = np.empty((0, 10))
    manual_scoring = np.array([])

    array_scores_total = np.array([])
    array_features_total = np.empty((0, 10))
    indices_artefact_total = np.array([])
    indices_wake_total = np.array([])
    indices_nrem_total = np.array([])
    indices_TS_total = np.array([])
    indices_rem_total = np.array([])
    with h5py.File(file_path, 'r') as hdf:
      for i in rat_indices_OS:
        array_scores = np.array([])
        array_features = np.empty((0, 10))
        for group_name in hdf["OS_basic"][f"Rat{i}"]:
            group = hdf["OS_basic"][f"Rat{i}"][group_name]
            array_scores = np.concatenate((array_scores, group['Mapped_scores'][:]), axis = 0)
            array_features = np.concatenate((array_features, group['Features'][:]), axis = 0)
        indices_artefact = np.where(array_scores == 0)[0] # Artefact
        indices_wake = np.where(array_scores == 1)[0] # Wake
        indices_nrem = np.where(array_scores == 3)[0] # NREM
        indices_TS = np.where(array_scores  == 4)[0] #TS
        indices_rem = np.where(array_scores == 5)[0] # REM
        len_scores = len(array_scores_total)

        array_scores_total = np.concatenate((array_scores_total, array_scores), axis = 0)
        array_features_total = np.concatenate((array_features_total, array_features), axis = 0)
        indices_artefact = np.array([el + len_scores for el in indices_artefact])
        indices_wake = np.array([el + len_scores for el in indices_wake])
        indices_nrem = np.array([el + len_scores for el in indices_nrem])
        indices_TS = np.array([el + len_scores for el in indices_TS])
        indices_rem = np.array([el + len_scores for el in indices_rem])
        indices_artefact_total = np.concatenate((indices_artefact_total, indices_artefact), axis = 0)
        indices_wake_total = np.concatenate((indices_wake_total, indices_wake), axis = 0)
        indices_nrem_total = np.concatenate((indices_nrem_total, indices_nrem), axis = 0)
        indices_TS_total = np.concatenate((indices_TS_total, indices_TS), axis = 0)
        indices_rem_total = np.concatenate((indices_rem_total, indices_rem), axis = 0)

      for i in rat_indices_RGS:
        array_scores = np.array([])
        array_features = np.empty((0, 10))
        for group_name in hdf["RGS14"][f"Rat{i}"]:
            group = hdf["RGS14"][f"Rat{i}"][group_name]
            array_scores = np.concatenate((array_scores, group['Mapped_scores'][:]), axis = 0)
            array_features = np.concatenate((array_features, group['Features'][:]), axis = 0)
        indices_artefact = np.where(array_scores == 0)[0] # Artefact
        indices_wake = np.where(array_scores == 1)[0] # Wake
        indices_nrem = np.where(array_scores == 3)[0] # NREM
        indices_TS = np.where(array_scores  == 4)[0] #TS
        indices_rem = np.where(array_scores == 5)[0] # REM
        len_scores = len(array_scores_total)

        array_scores_total = np.concatenate((array_scores_total, array_scores), axis = 0)
        array_features_total = np.concatenate((array_features_total, array_features), axis = 0)
        indices_artefact = np.array([el + len_scores for el in indices_artefact])
        indices_wake = np.array([el + len_scores for el in indices_wake])
        indices_nrem = np.array([el + len_scores for el in indices_nrem])
        indices_TS = np.array([el + len_scores for el in indices_TS])
        indices_rem = np.array([el + len_scores for el in indices_rem])
        indices_artefact_total = np.concatenate((indices_artefact_total, indices_artefact), axis = 0)
        indices_wake_total = np.concatenate((indices_wake_total, indices_wake), axis = 0)
        indices_nrem_total = np.concatenate((indices_nrem_total, indices_nrem), axis = 0)
        indices_TS_total = np.concatenate((indices_TS_total, indices_TS), axis = 0)
        indices_rem_total = np.concatenate((indices_rem_total, indices_rem), axis = 0)

      for i in rat_indices_CBD:
        array_scores = np.array([])
        array_features = np.empty((0, 10))
        for group_name in hdf["CBD"][f"Rat{i}"]:
            group = hdf["CBD"][f"Rat{i}"][group_name]
            array_scores = np.concatenate((array_scores, group['Mapped_scores'][:]), axis = 0)
            array_features = np.concatenate((array_features, group['Features'][:]), axis = 0)
        indices_artefact = np.where(array_scores == 0)[0] # Artefact
        indices_wake = np.where(array_scores == 1)[0] # Wake
        indices_nrem = np.where(array_scores == 3)[0] # NREM
        indices_TS = np.where(array_scores  == 4)[0] #TS
        indices_rem = np.where(array_scores == 5)[0] # REM
        len_scores = len(array_scores_total)
        array_scores_total = np.concatenate((array_scores_total, array_scores), axis = 0)
        array_features_total = np.concatenate((array_features_total, array_features), axis = 0)
        indices_artefact = np.array([el + len_scores for el in indices_artefact])
        indices_wake = np.array([el + len_scores for el in indices_wake])
        indices_nrem = np.array([el + len_scores for el in indices_nrem])
        indices_TS = np.array([el + len_scores for el in indices_TS])
        indices_rem = np.array([el + len_scores for el in indices_rem])
        indices_artefact_total = np.concatenate((indices_artefact_total, indices_artefact), axis = 0)
        indices_wake_total = np.concatenate((indices_wake_total, indices_wake), axis = 0)
        indices_nrem_total = np.concatenate((indices_nrem_total, indices_nrem), axis = 0)
        indices_TS_total = np.concatenate((indices_TS_total, indices_TS), axis = 0)
        indices_rem_total = np.concatenate((indices_rem_total, indices_rem), axis = 0)

    states = [el.lower() for el in states]

    if "artefact" in states:
      features_artefact = array_features_total[indices_artefact_total.astype(int)]
      scores = array_scores_total[indices_artefact_total.astype(int)]
      dataset = np.concatenate((dataset, features_artefact), axis=0)
      manual_scoring = np.concatenate((manual_scoring, scores), axis=0)

    if "wake" in states:
      features_wake = array_features_total[indices_wake_total.astype(int)]
      scores = array_scores_total[indices_wake_total.astype(int)]
      dataset = np.concatenate((dataset, features_wake), axis=0)
      manual_scoring = np.concatenate((manual_scoring, scores), axis=0)

    if "nrem" in states:
      features_nrem = array_features_total[indices_nrem_total.astype(int)]
      scores = array_scores_total[indices_nrem_total.astype(int)]
      dataset = np.concatenate((dataset, features_nrem), axis=0)
      manual_scoring = np.concatenate((manual_scoring, scores), axis=0)

    if "ts" in states:
      features_TS = array_features_total[indices_TS_total.astype(int)]
      scores = array_scores_total[indices_TS_total.astype(int)]
      dataset = np.concatenate((dataset, features_TS), axis=0)
      manual_scoring = np.concatenate((manual_scoring, scores), axis=0)

    if "rem" in states:
      features_rem = array_features_total[indices_rem_total.astype(int)]
      scores = array_scores_total[indices_rem_total.astype(int)]
      dataset = np.concatenate((dataset, features_rem), axis=0)
      manual_scoring = np.concatenate((manual_scoring, scores), axis=0)

    dataset = np.array(dataset)
    manual_scoring = np.array(manual_scoring)

    return dataset, manual_scoring.astype(int)

def dataset_in_order_OS_RGS_CBD(file_path, rat_indexes_OS, rat_indexes_RGS, rat_indexes_CBD, states):
    """
    Extract a dataset from the OS_RGS_CBD.h5 file while keeping the epochs in a chronolical order for each posttrial.
    
    Parameters : 
    file_path (str): the file_path to the OS_RGS_CBD.h5 file
    rat_indices_OS (int array): the indices of the rats from the OS_Basic dataset that would be included in the training dataset. #Included in [1, 3, 4, 6, 9, 11, 13]
    rat_indices_RGS (int array): the indices of the rats from the RGS14 dataset that would be included in the training dataset. #Included in [1, 2, 3, 4, 6, 7, 8, 9] (RGS control : [1, 2, 6, 9] and RGS positive : [3, 4, 7, 8])
    rat_indices_CBD (int array): the indices of the rats from the CBD dataset that would be included in the training dataset. #Included in [3, 4, 5, 6]
    states : (str array) : an array containing the names of the states wanted in the dataset #Included in  ["artefact", "Wake", "NREM", "TS", "REM]
    
    Returns :
    float numpy.array : an array containing the features for each epoch of the dataset.
    int numpy.array : an array containing the manual sleep state scoring of each epoch of the dataset.

    Notes:
    The dataset obtained is in a chronological order.
    If not all the states are included in the posttrial, this function will take a longer time than the dataset_OS_RGS_CBD function.
    """
    dataset = np.empty((0, 10))
    manual_scoring = np.array([])
    states = [el.lower() for el in states]
    int_states = []
    if "artefact" in states:
      int_states.append(0)
    if "wake" in states:
      int_states.append(1)
    if "nrem" in states:
      int_states.append(3)
    if "ts" in states:
      int_states.append(4)
    if "rem" in states:
      int_states.append(5)

    array_scores_total = np.array([])
    array_features_total = np.empty((0, 10))
    indices_artefact_total = np.array([])
    indices_wake_total = np.array([])
    indices_nrem_total = np.array([])
    indices_TS_total = np.array([])
    indices_rem_total = np.array([])
    with h5py.File(file_path, 'r') as hdf:
      for i in rat_indexes_OS:
        array_scores = np.array([])
        array_features = np.empty((0, 10))
        for group_name in hdf["OS_basic"][f"Rat{i}"]:
            group = hdf["OS_basic"][f"Rat{i}"][group_name]
            array_scores = group['Mapped_scores'][:]
            array_features = group['Features'][:]
            if int_states == [0, 1, 3, 4, 5]:
              array_features_total = np.concatenate((array_features_total, array_features), axis = 0)
              array_scores_total = np.concatenate((array_scores_total, array_scores), axis = 0)
            else:
              for j, el in enumerate(array_scores):
                  if el in int_states:
                    array_features_total = np.concatenate((array_features_total, [array_features[j]]), axis = 0)
                    array_scores_total = np.append(array_scores_total ,el)

      for i in rat_indexes_RGS:
        array_scores = np.array([])
        array_features = np.empty((0, 10))
        for group_name in hdf["RGS14"][f"Rat{i}"]:
            group = hdf["RGS14"][f"Rat{i}"][group_name]
            array_scores = group['Mapped_scores'][:]
            array_features = group['Features'][:]
            if int_states == [0, 1, 3, 4, 5]:
              array_features_total = np.concatenate((array_features_total, array_features), axis = 0)
              array_scores_total = np.concatenate((array_scores_total, array_scores), axis = 0)
            else:
              for j, el in enumerate(array_scores):
                  if el in int_states:
                    array_features_total = np.concatenate((array_features_total, [array_features[j]]), axis = 0)
                    array_scores_total = np.append(array_scores_total ,el)

      for i in rat_indexes_CBD:
        array_scores = np.array([])
        array_features = np.empty((0, 10))
        for group_name in hdf["CBD"][f"Rat{i}"]:
            group = hdf["CBD"][f"Rat{i}"][group_name]
            array_scores = group['Mapped_scores'][:]
            array_features = group['Features'][:]
            if int_states == [0, 1, 3, 4, 5]:
              array_features_total = np.concatenate((array_features_total, array_features), axis = 0)
              array_scores_total = np.concatenate((array_scores_total, array_scores), axis = 0)
            else:
              for j, el in enumerate(array_scores):
                  if el in int_states:
                    array_features_total = np.concatenate((array_features_total, [array_features[j]]), axis = 0)
                    array_scores_total = np.append(array_scores_total ,el)
                    
    return array_features_total, array_scores_total.astype(int)

def posttrial_OS_RGS_CBD(file_path, rat_dataset, rat_index, posttrial_name, states):
    """
    Extract a posttrial from the OS_RGS_CBD.h5 file.
    
    Parameters : 
    file_path (str): the file_path to the OS_RGS_CBD.h5 file
    rat_dataset (str) : the name of the rat dataset #"OS", "RGS" or"CBD"
    rat_index (int) : the index of the rat
    posttrial_name (str) : the name of the posttrial
    states : (str array) : an array containing the names of the states wanted in the dataset #Included in  ["artefact", "Wake", "NREM", "TS", "REM]
    
    Returns :
    float numpy.array : an array containing the features for each epoch of the dataset.
    int numpy.array : an array containing the manual sleep state scoring of each epoch of the dataset.

    Notes:
    The dataset obtained is in a chronological order.
    """
    dataset = np.empty((0, 10))
    manual_scoring = np.array([])

    states = [el.lower() for el in states]
    int_states = []
    if "artefact" in states:
      int_states.append(0)
    if "wake" in states:
      int_states.append(1)
    if "nrem" in states:
      int_states.append(3)
    if "ts" in states:
      int_states.append(4)
    if "rem" in states:
      int_states.append(5)

    with h5py.File(file_path, 'r') as hdf:
      if rat_dataset == "OS":
        group = hdf["OS_basic"][f"Rat{rat_index}"][posttrial_name]
        array_scores = group['Mapped_scores'][:]
        array_features = group['Features'][:]
        if int_states == [0, 1, 3, 4, 5]:
              dataset = np.concatenate((dataset, array_features), axis = 0)
              manual_scoring = np.concatenate((manual_scoring, array_scores), axis = 0)
        else:
          for i, el in enumerate(array_scores):
              if el in int_states:
                dataset = np.concatenate((dataset, [array_features[i]]), axis = 0)
                manual_scoring = np.append(manual_scoring ,el)


      elif rat_dataset == "RGS":
        group = hdf["RGS14"][f"Rat{rat_index}"][posttrial_name]
        array_scores = group['Mapped_scores'][:]
        array_features = group['Features'][:]
        if int_states == [0, 1, 3, 4, 5]:
              dataset = np.concatenate((dataset, array_features), axis = 0)
              manual_scoring = np.concatenate((manual_scoring, array_scores), axis = 0)
        else:
          for i, el in enumerate(array_scores):
              if el in int_states:
                dataset = np.concatenate((dataset, [array_features[i]]), axis = 0)
                manual_scoring = np.append(manual_scoring ,el)

      elif rat_dataset == "CBD":
        group = hdf["CBD"][f"Rat{rat_index}"][posttrial_name]
        array_scores = group['Mapped_scores'][:]
        array_features = group['Features'][:]
        if int_states == [0, 1, 3, 4, 5]:
              dataset = np.concatenate((dataset, array_features), axis = 0)
              manual_scoring = np.concatenate((manual_scoring, array_scores), axis = 0)
        else:
          for i, el in enumerate(array_scores):
              if el in int_states:
                dataset = np.concatenate((dataset, [array_features[i]]), axis = 0)
                manual_scoring = np.append(manual_scoring ,el)

    return dataset, manual_scoring.astype(int)
