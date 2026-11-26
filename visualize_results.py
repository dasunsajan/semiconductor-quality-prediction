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
# Load actual results from nested cross-validation

nested_results = pd.read_csv(
    "results/nested_cv_results.csv"
)

nested_metrics = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "balanced_accuracy",
    "average_precision"
]

nested_labels = [
    "Accuracy",
    "Fail Precision",
    "Fail Recall",
    "Fail F1-score",
    "Balanced Accuracy",
    "Average Precision"
]

# Calculate mean and standard deviation across outer folds
nested_means = nested_results[nested_metrics].mean()
nested_stds = nested_results[nested_metrics].std(ddof=0)

# Create nested CV performance chart
plt.figure(figsize=(12, 6))

bars = plt.bar(
    nested_labels,
    nested_means.values,
    yerr=nested_stds.values,
    capsize=5,
    color="seagreen",
    edgecolor="black",
    alpha=0.85
)

# Add metric values above bars
for bar, mean, std in zip(
    bars,
    nested_means.values,
    nested_stds.values
):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        mean + std + 0.015,
        f"{mean:.3f}",
        ha="center",
        va="bottom",
        fontsize=9
    )

plt.ylabel("Score")
plt.ylim(0, 1.0)

plt.title(
    "Logistic Regression - Nested 5-Fold Cross-Validation"
)

plt.xticks(
    rotation=25,
    ha="right"
)

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.4
)

plt.tight_layout()

plt.savefig(
    "results/nested_cv_performance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

print("\nNested CV Mean Results:")
print(nested_means)

print("\nNested CV Standard Deviations:")
print(nested_stds)

print("\nResult visualizations completed!")
