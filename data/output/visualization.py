import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from zipfile import ZipFile
import sys
from pathlib import Path

# Load the results
if len(sys.argv) > 1:
    csv_path = sys.argv[1]
else:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent if script_dir.name == "output" else script_dir
    output_dir_path = project_root / "data" / "output"
    output_files = list(output_dir_path.glob("*.csv"))
    if not output_files:
        raise FileNotFoundError("Nenhum arquivo CSV encontrado em data/output/")
    csv_path = max(output_files, key=lambda f: f.stat().st_ctime)

print(f"Loaded data from: {csv_path}")
df = pd.read_csv(csv_path)
filtered_df = df[df["Feature Selection Algorithm"] != "Low Variance"]

# Prepare output directory
output_dir = project_root / "figures"
output_dir.mkdir(parents=True, exist_ok=True)

sns.set(style="whitegrid")

# 1. Balanced Accuracy by Feature Selection Algorithm
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Feature Selection Algorithm", y="Balanced Accuracy")
plt.title("Balanced Accuracy by Feature Selection Algorithm")
plt.tight_layout()
plt.savefig(output_dir / "balanced_accuracy_by_selector.png")
plt.close()

# 2. Accuracy vs Number of Features
plt.figure(figsize=(8, 5))
sns.lineplot(data=filtered_df, x="Number of Features", y="Balanced Accuracy", hue="Feature Selection Algorithm", marker="o")
plt.title("Accuracy vs Number of Features by Feature Selector")
plt.tight_layout()
plt.savefig(output_dir / "accuracy_vs_num_features.png")
plt.close()

# 3. F1-Score Boxplot
plt.figure(figsize=(8, 5))
sns.boxplot(data=filtered_df, x="Feature Selection Algorithm", y="F1-Score")
plt.title("F1-Score by Feature Selection Algorithm")
plt.tight_layout()
plt.savefig(output_dir / "f1_score_by_selector.png")
plt.close()

# 4. Training Time vs Accuracy
plt.figure(figsize=(8, 5))
sns.scatterplot(data=filtered_df, x="Train Time (s)", y="Balanced Accuracy", hue="Feature Selection Algorithm", style="Number of Features")
plt.title("Training Time vs Balanced Accuracy")
plt.tight_layout()
plt.savefig(output_dir / "training_time_vs_accuracy.png")
plt.close()

# 5. Precision and Recall per User
agg = df.groupby("User")[["Precision", "Recall"]].mean().reset_index().melt(id_vars="User", var_name="Metric", value_name="Score")
plt.figure(figsize=(10, 5))
sns.barplot(data=agg, x="User", y="Score", hue="Metric")
plt.title("Average Precision and Recall per User")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(output_dir / "precision_recall_per_user.png")
plt.close()

zip_path = project_root / "figures.zip"
with ZipFile(zip_path, "w") as zipf:
    for root, _, files in os.walk(output_dir):
        for file in files:
            zipf.write(os.path.join(root, file), arcname=os.path.join(os.path.basename(root), file))
