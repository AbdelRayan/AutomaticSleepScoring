def GetInferredStates(uniqueStates, obsKeys, uniqueStatesStr, obsKeys_pt, uniqueStates_pt, uniqueStatesStr_pt):
    majority_inferred_states = []

    # Iterate through each unique state in the training data
    for i in uniqueStates[:, 0]:
        # Find indices where the second column of obsKeys matches the current unique state
        idx = np.where(obsKeys[:, 1] == i)[0]

        # Extract frames corresponding to the current latent state
        latent_frames = obsKeys[idx, :]

        # Calculate the proportion of each state (awake, nrem, rem) in the latent frames
        awake_pct = round(len(np.where(latent_frames[:, 3] == 1)[0]) / float(len(latent_frames)), 3)
        nrem_pct = round(len(np.where(latent_frames[:, 3] == 3)[0]) / float(len(latent_frames)), 3)
        rem_pct = round(len(np.where(latent_frames[:, 3] == 5)[0]) / float(len(latent_frames)), 3)

        # Determine the majority state based on the proportions and append to the list
        if awake_pct >= nrem_pct and awake_pct >= rem_pct:
            majority_inferred_states.append(1)
        elif nrem_pct >= rem_pct and nrem_pct >= awake_pct:
            majority_inferred_states.append(3)
        elif rem_pct >= nrem_pct and rem_pct >= awake_pct:
            majority_inferred_states.append(5)

    # Initialize an array to store the inferred states for the test dataset
    inferred_states = np.zeros(len(obsKeys_pt[:, 1]))

    # Iterate through each unique state in the test data
    for i in range(len(uniqueStates_pt[:, 0])):
        # Find indices where the second column of obsKeys_pt matches the current unique state
        idx_pt = np.where(obsKeys_pt[:, 1] == uniqueStates_pt[i, 0])[0]

        # If the unique state string is not found in the training data, set inferred state to -1
        if uniqueStatesStr_pt[i] not in uniqueStatesStr:
            inferred_states[idx_pt] = -1
        else:
            # Otherwise, find the corresponding majority inferred state from training data
            for k in range(len(uniqueStatesStr)):
                if uniqueStatesStr[k] == uniqueStatesStr_pt[i]:
                    inferred_states[idx_pt] = majority_inferred_states[k]
    
    return inferred_states