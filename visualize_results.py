
# Group 29 - Logistic Regression Result Visualization

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Create results directory
Path("results").mkdir(exist_ok=True)

# Results from 5-Fold Stratified Cross Validation
metrics = [
    "Accuracy",
    "Fail Precision",
    "Fail Recall",
    "Fail F1-score",
    "Balanced Accuracy",
    "Average Precision"
]

baseline = [0.8864, 0.1504, 0.1538, 0.1508, 0.5461, 0.1485]
weighted = [0.8443, 0.1147, 0.2024, 0.1456, 0.5462, 0.1488]

comparison = pd.DataFrame({
    "Metric": metrics,
    "Baseline": baseline,
    "Class Weighted": weighted
})

# Save results as CSV
comparison.to_csv(
    "results/model_comparison.csv",
    index=False
)

# Create comparison bar chart
x = range(len(metrics))
width = 0.35

plt.figure(figsize=(12, 6))

plt.bar(
    [i - width / 2 for i in x],
    [v * 100 for v in baseline],
    width,
    label="Baseline",
    color="steelblue"
)

plt.bar(
    [i + width / 2 for i in x],
    [v * 100 for v in weighted],
    width,
    label="Class Weighted",
    color="darkorange"
)

plt.xticks(
    list(x),
    metrics,
    rotation=25,
    ha="right"
)

plt.ylabel("Score (%)")
plt.title(
    "Logistic Regression - 5-Fold CV Comparison"
)
plt.legend()
plt.tight_layout()

plt.savefig(
    "results/model_comparison.png",
    dpi=300
)
plt.show()

# Nested CV results - separate visualization
nested_scores = [
    0.8762, 0.1515, 0.2224,
    0.1711, 0.5726, 0.1228
]

plt.figure(figsize=(10, 5))

plt.bar(
    metrics,
    [v * 100 for v in nested_scores],
    color="seagreen"
)

plt.xticks(rotation=25, ha="right")
plt.ylabel("Mean Score (%)")
plt.title(
    "Logistic Regression - Nested CV Results"
)

plt.tight_layout()

plt.savefig(
    "results/nested_cv_performance.png",
    dpi=300
)
plt.show()

print("Result visualizations completed!")
print(comparison)
