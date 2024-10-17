
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# Random data generation
random_seed = np.random.RandomState(12)

# Generating a set of normal observations for training data
x_train = 0.5 * random_seed.randn(500, 2)
x_train = np.r_[x_train + 3, x_train]
x_train = pd.DataFrame(x_train, columns=["x", "y"])

# Generating a testing set with normal observations
x_test = 0.5 * random_seed.randn(500, 2)
x_test = np.r_[x_test + 3, x_test]
x_test = pd.DataFrame(x_test, columns=["x", "y"])

# Generating a set of outlier observations
x_outliers = random_seed.uniform(low=-5, high=5, size=(50, 2))
x_outliers = pd.DataFrame(x_outliers, columns=["x", "y"])

# Creating subplots to show both plots side by side
fig, (ax1, ax2,ax3) = plt.subplots(1, 3, figsize=(12, 6))

# First plot - Generated data
p1 = ax1.scatter(x_train.x, x_train.y, c="white", s=50, edgecolors="black")
p2 = ax1.scatter(x_test.x, x_test.y, c="green", s=50, edgecolors="black")
p3 = ax1.scatter(x_outliers.x, x_outliers.y, c="blue", s=50, edgecolors="black")
ax1.set_xlim((-6, 6))
ax1.set_ylim((-6, 6))
ax1.legend([p1, p2, p3], ["Training set", "Normal testing set", "Anomalous testing set"], loc="lower right")
ax1.set_title("Generated Data")

# Now train an Isolation Forest model on training data
clf = IsolationForest()
clf.fit(x_train)
y_pred_train = clf.predict(x_train)
y_pred_test = clf.predict(x_test)
# Predict outliers
y_pred_outliers = clf.predict(x_outliers)
x_outliers = x_outliers.assign(pred=y_pred_outliers)

# Second plot - Outlier detection
p1 = ax2.scatter(x_train.x, x_train.y, c="white", s=50, edgecolors="black")
p2 = ax2.scatter(
    x_outliers.loc[x_outliers.pred == -1, "x"],
    x_outliers.loc[x_outliers.pred == -1, "y"],
    c="blue", s=50, edgecolors="black")
p3 = ax2.scatter(
    x_outliers.loc[x_outliers.pred == 1, "x"],
    x_outliers.loc[x_outliers.pred == 1, "y"],
    c="red", s=50, edgecolors="black")
ax2.set_xlim((-6, 6))
ax2.set_ylim((-6, 6))
ax2.legend([p1, p2, p3], ["Training observations", "Normal points", "Detected outliers"], loc="lower right")
ax2.set_title("Outlier Detection")



# Now perform on normal data, to check the performance
x_test = x_test.assign(pred=y_pred_test)
x_test.head()

#plot
p1 = ax3.scatter(x_train.x, x_train.y, c = "white", s=50, edgecolor = "black")
p2 = ax3.scatter(
    x_test.loc[x_test.pred==1, ["x"]],
    x_test.loc[x_test.pred==1, ["y"]],
    c = "blue",
    s=50,
    edgecolor = "black",
)
p3 = ax3.scatter(
    x_test.loc[x_test.pred==-1, ["x"]],
    x_test.loc[x_test.pred==-1, ["y"]],
    c = "red",
    s=50,
    edgecolor = "black",
)
ax3.set_xlim((-6,6))
ax3.set_ylim((-6,6))
ax3.legend(
    [p1,p2,p3],
    [
        "Training observations",
        "Correctly labeled test observations",
        "Incorrectly labeled test observations",
    ],
    loc = "lower right",
)
ax3.set_title("Overall Plot")
plt.tight_layout()
plt.show()