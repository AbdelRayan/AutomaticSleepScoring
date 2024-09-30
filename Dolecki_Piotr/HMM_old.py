# THIS IS AN OLDER VERSION
# less accurate and a little bit different method
# install hmmlearn first time running on colab : (line below)
!pip install hmmlearn

import numpy as np
from hmmlearn import hmm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# SEQUENCES
#original_sequence = [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5, 1, 1, 2, 2, 3, 3, 4, 4, 5, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 5, 5, 1, 1, 2, 2, 3, 4, 4, 5] * 5
original_sequence = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5]
#original_sequence = [1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 3, 3, 5, 5, 5, 3, 2, 2, 1, 1, 2, 3, 3, 4, 4, 4, 4, 3, 3, 5, 5, 5, 3, 2, 1, 1, 2, 3, 3, 4, 3, 3, 5, 5, 5, 5, 3, 2, 1, 2, 3, 3, 3, 5, 5, 5, 5, 2, 1, 1, 2, 3, 5, 5, 5, 5, 3, 2, 1, 1, 1, 1] * 3 

original_sequence = np.array(original_sequence)
n_states = 5

# TRANSITION MATRIX
transition_matrix = np.zeros((n_states, n_states))
for (current_state, next_state) in zip(original_sequence[:-1], original_sequence[1:]):
    transition_matrix[current_state - 1, next_state - 1] += 1
transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True) # normalization of the transition matrix (row-wise)

# EMISSION MATRIX
emission_matrix = np.eye(n_states)

# Preprocess the observed sequence: convert to 0-based indexing
observed_sequence = (original_sequence - 1).reshape(-1, 1)

# Set up observed counts for HMM (one-hot encoded observations)
observed_counts = np.zeros((observed_sequence.shape[0], n_states), dtype=int)
for i in range(observed_sequence.shape[0]):
    observed_counts[i, observed_sequence[i, 0]] = 1

# Initialize the HMM model with init_params='e' to keep the manual transition and start probabilities
model = hmm.MultinomialHMM(n_components=n_states, n_iter=1, tol=0.01, random_state=42, init_params='e')

# START PROBABILITIES
start_state = original_sequence[0] - 1  # Convert to 0-based index
model.startprob_ = np.zeros(n_states)
model.startprob_[start_state] = 1  # Set 100% start probability for the initial state of the original sequence

# TRANS & EMISSION MATRIXES
model.transmat_ = transition_matrix
model.emissionprob_ = emission_matrix

# TRAINING
iterations = 200
for _ in range(iterations):
    model.fit(observed_counts)

# PREDICTED SEQUENCE
predicted_sequence = model.predict(observed_counts)
predicted_sequence = predicted_sequence + 1  # Adjust back to 1-based indexing

# ACCURACY (%)
accuracy = accuracy_score(original_sequence, predicted_sequence)
print(f"Original Sequence: {original_sequence}")
print(f"Predicted Sequence: {predicted_sequence}")
print(f"Accuracy: {accuracy * 100:.2f}%")

# FIGURES
plt.figure(figsize=(14, 6))

plt.subplot(2, 1, 1)
plt.plot(original_sequence, color='red', marker='o', linestyle='dashed')
plt.title("Original Sequence")
plt.xlabel("Time Steps")
plt.ylabel("Sleep States")
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(predicted_sequence, color='green', marker='o', linestyle='dashed')
plt.title("Predicted Sequence")
plt.xlabel("Time Steps")
plt.ylabel("Sleep States")
plt.grid(True)

plt.tight_layout()
plt.show()
