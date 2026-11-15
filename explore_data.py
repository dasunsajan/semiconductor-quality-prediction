
import pandas as pd

# Load the SECOM dataset
X = pd.read_csv("secom.data", sep=r"\s+", header=None)

# Load the labels
labels = pd.read_csv(
    "secom_labels.data",
    sep=r"\s+",
    header=None
)

y = labels.iloc[:, 0]

# Display dataset information
print("Dataset Shape:", X.shape)

print("\nFirst 5 Records:")
print(X.head())

print("\nPass/Fail Distribution:")
print(y.value_counts())

print("\nTotal Missing Values:")
print(X.isnull().sum().sum())


import matplotlib.pyplot as plt

# Create folder for saving graphs
from pathlib import Path
Path("results").mkdir(exist_ok=True)

# Graph 1: Pass/Fail Distribution
counts = y.value_counts().reindex([-1, 1], fill_value=0)

plt.figure(figsize=(7, 5))
plt.bar(["Pass", "Fail"], counts.values)

plt.title("SECOM Dataset - Class Distribution")
plt.xlabel("Quality Result")
plt.ylabel("Number of Records")

for i, count in enumerate(counts.values):
    plt.text(i, count + 10, str(count), ha="center")

plt.tight_layout()
plt.savefig("results/class_distribution.png")
plt.show()

# Graph 2: Missing Values per Feature
missing = X.isnull().sum()

plt.figure(figsize=(10, 5))
plt.hist(missing, bins=30)

plt.title("Missing Values per Feature")
plt.xlabel("Number of Missing Values")
plt.ylabel("Number of Features")

plt.tight_layout()
plt.savefig("results/missing_values.png")
plt.show()

print("EDA graphs saved in results folder!")
