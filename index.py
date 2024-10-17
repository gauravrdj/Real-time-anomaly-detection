#Real time Anomaly Detection
#Cobblestone Energy Assignment

#Adding libraries

import numpy as np
import matplotlib.pyplot as plt
import rrcf


# Data Stream Simulation
def generate_data_stream(pattern_length):
    try:
         # Check if pattern_length is a positive integer
        if not isinstance(pattern_length, int) or pattern_length <= 0:
            raise ValueError("Pattern length must be a positive integer.")
         # Set the pattern length
        n = pattern_length
         # Set amplitude
        A = 50 
        center = 100  # center value
        phi = 30  # phase angle
        T = 2 * np.pi / 100  # period
        #Generate time value
        t = np.arange(n)
         # Generate a sinusoidal pattern with specified parameters
        sin = A * np.sin(T * t - phi * T) + center
        return sin
    except ValueError as e:
        print(f"Error in generate_data_stream: {e}")
        
def add_anomalies_to_data(data_stream):
    try:
        # Check if data_stream is a list or NumPy array
        if not isinstance(data_stream, (list, np.ndarray)):
            raise TypeError("Data stream must be a list or NumPy array.")
         # Introduce anomalies by setting values at specific indices to 100, 123, 345, 456, 460, 622, 700
        data_stream[100] = 28
        data_stream[123] = 63
        data_stream[345] = 45
        data_stream[456] = 66
        data_stream[460] = 3
        data_stream[622] = 36
        data_stream[700] = 59
    except TypeError as e:
        print(f"Error in add_anomalies_to_data: {e}")



# Anomaly Detection with RCTreeForest
class RCTreeForest:
    def __init__(self, num_trees, tree_size, window_size):
        try:
            # Check if input parameters are positive integers
            if not all(isinstance(param, int) and param > 0 for param in [num_trees, tree_size, window_size]):
                raise ValueError("All parameters must be positive integers.")
            # Initialize the Random Cut Tree (RCT) forest with a specified number of trees and tree size
            self.num_trees = num_trees
            self.tree_size = tree_size
            self.shingle_size = window_size
            # Create a list of RCTree instances to form the forest
            self.forest = [rrcf.RCTree() for _ in range(num_trees)]
        except ValueError as e:
            print(f"Error in initialization: {e}")

    def anomaly_detector(self, index, point):
        try:
               # Check if index is a positive integer
            if not isinstance(index, int) or index < 0:
                raise ValueError("Index must be a non-negative integer.")
            
            # Check if point is a list of numeric values with the correct length
            if not isinstance(point, list) or len(point) != self.shingle_size or not all(isinstance(p, (int, float)) for p in point):
                raise ValueError(f"Point must be a list of {self.shingle_size} numeric values.")
              # Initialize average codisplacement to zero
            avg_codisplacement = 0
            for tree in self.forest:
                # If the tree size exceeds the specified limit, forget the oldest point (FIFO)
                if len(tree.leaves) > self.tree_size:
                    tree.forget_point(index - self.tree_size)
                    # Insert the new point into the tree
                tree.insert_point(point, index=index)
                # Compute the codisplacement for the new point
                new_codisplacement = tree.codisp(index)
                # Accumulate the codisplacement across all trees
                avg_codisplacement += new_codisplacement / self.num_trees
                # Return the average codisplacement for the given point
            return avg_codisplacement
        except ValueError as e:
            print(f"Error in anomaly detection: {e}")

# Generate a data stream with a sinusoidal pattern of length 1000
data_stream = generate_data_stream(1000)
# Introduce anomalies to the generated data stream
add_anomalies_to_data(data_stream)
# Define the number of trees in the Random Cut Tree (RCT) forest
num_trees = 40
# Define the size limit for each tree in the RCT forest
tree_size = 256
# Define the size of the window
shingle_size = 4
# Create an instance of the RCTreeForest class with the specified number of trees and tree size
forest = RCTreeForest(num_trees, tree_size, shingle_size)

# Anomaly detection loop
# Initialize empty lists to store anomaly scores and the current data window
anomaly_score = []
current_window = []
# Iterate through the data stream
for i in range(len(data_stream)):
    # If the index is within the shingle size, populate the initial window with data_stream values
    if i < forest.shingle_size:
        current_window.append(data_stream[i])
        # Initialize anomaly score to 0 for the initial window
        anomaly_score.append(0)
        continue
    else:
        # Update the current window by adding the latest data_stream value and removing the oldest
        current_window.append(data_stream[i])
        current_window = current_window[1:]
    # Calculate anomaly score using the RCT forest for the current window
    score = forest.anomaly_detector(i, current_window)
     # Append the calculated anomaly score to the list
    anomaly_score.append(score)
    #If there is a sudden peak we can say it is a anomaly
    if i > forest.shingle_size + 1 and (score >= 1.7 * anomaly_score[i - 1] or score <= -1.7 * anomaly_score[i - 1]):
        print("Anomaly Detected at index:", i)

# Plotting both the data stream and anomaly scores in one frame
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))  # Two rows, one column

# Plot the data stream in the first subplot
# Plot the original data stream using Matplotlib
ax1.plot(np.arange(1000), data_stream, label='Data Stream', color='blue')
ax1.set_title("Data Stream with Anomalies")
ax1.set_xlabel("Index")
ax1.set_ylabel("Value")
ax1.legend()


# Plot the calculated anomaly scores using Matplotlib
ax2.plot(np.arange(1000), anomaly_score, label='Anomaly Score', color='red')
ax2.set_title("Anomaly Score")
ax2.set_xlabel("Index")
ax2.set_ylabel("Anomaly Score")
ax2.legend()

# Adjust layout
plt.tight_layout()

# Display the plot
plt.show()

# Efficiency code snippet for further anomaly detection
forest = []
for _ in range(num_trees):
    tree = rrcf.RCTree()
    forest.append(tree)

def anomaly_detector(index, point):
    avg = 0
    for tree in forest:
          # If tree is above permitted size...
        if len(tree.leaves) > tree_size:
            # Drop the oldest point (FIFO)
            tree.forget_point(index - tree_size)
            # Insert the new point into the tree
        tree.insert_point(point, index=index)
        # Compute codisp on the new point...
        new_codisplacement = tree.codisp(index)
        # And take the average over all trees
        avg += new_codisplacement / num_trees
    return avg

anomaly_score = []
current_window = []
for i in range(len(data_stream)):
    if i < shingle_size:
        current_window.append(data_stream[i])
        anomaly_score.append(0)
        continue
    else:
        current_window.append(data_stream[i])
        current_window = current_window[1:]
        
    score = anomaly_detector(i, current_window)
    print(i, end=' ')
    anomaly_score.append(score)



