#Real time Anomaly Detection

#Adding libraries

import numpy as np
import matplotlib.pyplot as plt
import rrcf


# Data Stream Simulation
def generate_data_stream(pattern_length):
    try:
        if not isinstance(pattern_length, int) or pattern_length <= 0:
            raise ValueError("Pattern length must be a positive integer.")
        n = pattern_length
        A = 50  # amplitude
        center = 100  # center value
        phi = 30  # phase angle
        T = 2 * np.pi / 100  # period
        t = np.arange(n)
        sin = A * np.sin(T * t - phi * T) + center
        return sin
    except ValueError as e:
        print(f"Error in generate_data_stream: {e}")
        
def add_anomalies_to_data(data_stream):
    try:
        if not isinstance(data_stream, (list, np.ndarray)):
            raise TypeError("Data stream must be a list or NumPy array.")
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
            if not all(isinstance(param, int) and param > 0 for param in [num_trees, tree_size, window_size]):
                raise ValueError("All parameters must be positive integers.")
            self.num_trees = num_trees
            self.tree_size = tree_size
            self.shingle_size = window_size
            self.forest = [rrcf.RCTree() for _ in range(num_trees)]
        except ValueError as e:
            print(f"Error in initialization: {e}")

    def anomaly_detector(self, index, point):
        try:
            if not isinstance(index, int) or index < 0:
                raise ValueError("Index must be a non-negative integer.")
            if not isinstance(point, list) or len(point) != self.shingle_size or not all(isinstance(p, (int, float)) for p in point):
                raise ValueError(f"Point must be a list of {self.shingle_size} numeric values.")
            
            avg_codisplacement = 0
            for tree in self.forest:
                if len(tree.leaves) > self.tree_size:
                    tree.forget_point(index - self.tree_size)
                tree.insert_point(point, index=index)
                new_codisplacement = tree.codisp(index)
                avg_codisplacement += new_codisplacement / self.num_trees
            return avg_codisplacement
        except ValueError as e:
            print(f"Error in anomaly detection: {e}")

# Data stream generation and anomaly detection
data_stream = generate_data_stream(1000)
add_anomalies_to_data(data_stream)

num_trees = 40
tree_size = 256
shingle_size = 4
forest = RCTreeForest(num_trees, tree_size, shingle_size)

# Anomaly detection loop
anomaly_score = []
current_window = []
for i in range(len(data_stream)):
    if i < forest.shingle_size:
        current_window.append(data_stream[i])
        anomaly_score.append(0)
        continue
    else:
        current_window.append(data_stream[i])
        current_window = current_window[1:]
    
    score = forest.anomaly_detector(i, current_window)
    anomaly_score.append(score)
    
    if i > forest.shingle_size + 1 and (score >= 1.7 * anomaly_score[i - 1] or score <= -1.7 * anomaly_score[i - 1]):
        print("Anomaly Detected at index:", i)

# Plotting both the data stream and anomaly scores in one frame
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))  # Two rows, one column

# Plot the data stream in the first subplot
ax1.plot(np.arange(1000), data_stream, label='Data Stream', color='blue')
ax1.set_title("Data Stream with Anomalies")
ax1.set_xlabel("Index")
ax1.set_ylabel("Value")
ax1.legend()

# Plotting the anomaly score in the second subplot
ax2.plot(np.arange(1000), anomaly_score, label='Anomaly Score', color='red')
ax2.set_title("Anomaly Score")
ax2.set_xlabel("Index")
ax2.set_ylabel("Anomaly Score")
ax2.legend()

# Adjust layout
plt.tight_layout()

# Both plots
plt.show()

# Efficiency code snippet for further anomaly detection
forest = []
for _ in range(num_trees):
    tree = rrcf.RCTree()
    forest.append(tree)

def anomaly_detector(index, point):
    avg = 0
    for tree in forest:
        if len(tree.leaves) > tree_size:
            tree.forget_point(index - tree_size)
        tree.insert_point(point, index=index)
        new_codisplacement = tree.codisp(index)
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



