# BEST VERSION SO FAR

# install hmmlearn first time running on colab : (line below)
!pip install hmmlearn

import numpy as np
from hmmlearn import hmm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# DIFFERENT SEQUENCES TO TRY ON
#original_sequence = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5] * 5  # Example long sequence
#original_sequence = [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5, 1, 1, 2, 2, 3, 3, 4, 4, 5, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 5, 5, 1, 1, 2, 2, 3, 4, 4, 5] * 5
#original_sequence = [1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 3, 3, 5, 5, 5, 3, 2, 2, 1, 1, 2, 3, 3, 4, 4, 4, 4, 3, 3, 5, 5, 5, 3, 2, 1, 1, 2, 3, 3, 4, 3, 3, 5, 5, 5, 5, 3, 2, 1, 2, 3, 3, 3, 5, 5, 5, 5, 2, 1, 1, 2, 3, 5, 5, 5, 5, 3, 2, 1, 1, 1, 1] * 3 
original_sequence = [3, 2, 2, 5, 5, 5, 5, 4, 4, 4, 4, 1, 4, 4, 4, 4, 4, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 1, 3, 3, 3, 3, 3, 5, 5, 5, 2, 2, 4, 4, 4, 4, 5, 5, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 5, 5, 5, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 4, 5, 5, 5, 5, 5, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 5, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 5, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 5, 5, 5, 5, 5, 5, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 1, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 2, 2, 2, 2, 2, 5, 5, 5, 5, 5, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 4, 4, 4]

original_sequence = np.array(original_sequence)
n_states = 5  # hidden states

# TRANSITION MATRIX
transition_matrix = np.zeros((n_states, n_states))
for (current_state, next_state) in zip(original_sequence[:-1], original_sequence[1:]):
    transition_matrix[current_state - 1, next_state - 1] += 1
transition_matrix = (transition_matrix + 0.01)  # add small smoothing value to avoid zero probabilities
transition_matrix /= transition_matrix.sum(axis=1, keepdims=True)  # normalize row-wise

# EMISSION MATRIX (slightly noisy diagonal)
emission_matrix = np.eye(n_states) * 0.9 + 0.1 / n_states

# Convert observed sequence to 0-based indexing & set up observed counts for HMM (one-hot encoded observations)
observed_sequence = (original_sequence - 1).reshape(-1, 1)
observed_counts = np.zeros((observed_sequence.shape[0], n_states), dtype=int)
for i in range(observed_sequence.shape[0]):
    observed_counts[i, observed_sequence[i, 0]] = 1

# HMM MODEL
model = hmm.MultinomialHMM(n_components=n_states, n_iter=1, tol=0.01, random_state=42, init_params='')
start_state = original_sequence[0] - 1  # convert to 0-based index
model.startprob_ = np.zeros(n_states)
model.startprob_[start_state] = 1  # start with the same state as original sequence
model.transmat_ = transition_matrix
model.emissionprob_ = emission_matrix

# TRAINING
iterations = 10 
for _ in range(iterations):
    model.fit(observed_counts)

# PREDICTED SEQUENCE
predicted_sequence = model.predict(observed_counts)
predicted_sequence = predicted_sequence + 1  # adjust back to 1-based indexing

# REDUCTION OF LATENT STATES
reduced_sequence = [predicted_sequence[0]]
for i in range(1, len(predicted_sequence)):
    if predicted_sequence[i] != predicted_sequence[i - 1]:
        reduced_sequence.append(predicted_sequence[i])

# ACCURACY (%)
accuracy = accuracy_score(original_sequence, predicted_sequence)
print(f"Original Sequence: {original_sequence}")
print(f"Predicted Sequence: {predicted_sequence}")
print("Reduced Sequence:", reduced_sequence)
print(f"Original sequence length: {len(original_sequence)}")
print(f"Reduced sequence length: {len(reduced_sequence)}")
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
