# Re-import necessary libraries and reinitialize the process after the reset.
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from itertools import combinations

# Load the dataset again after the reset
file_path = '/mnt/data/test.tsv'
df = pd.read_csv(file_path, sep='\t')

# Ensure 'reasoning_types' column exists and preprocess
df["reasoning_types"] = df["reasoning_types"].astype(str)
df["reasoning_types_list"] = df["reasoning_types"].apply(lambda x: x.split(" | "))

# Count occurrences of each reasoning type
all_reasoning_types = [reasoning for sublist in df["reasoning_types_list"] for reasoning in sublist]
reasoning_counts = Counter(all_reasoning_types)
reasoning_counts_df = pd.DataFrame(list(reasoning_counts.items()), columns=["Reasoning Type", "Count"])

# Calculate overlaps between reasoning types
overlap_counts = Counter()
for reasoning_list in df["reasoning_types_list"]:
    for combination in combinations(reasoning_list, 2):
        overlap_counts[frozenset(combination)] += 1

# Convert frozensets to strings for visualization
overlap_data = [(tuple(k), v) for k, v in overlap_counts.items()]
overlap_counts_df = pd.DataFrame(overlap_data, columns=["Reasoning Pair", "Count"])
overlap_counts_df["Reasoning Pair"] = overlap_counts_df["Reasoning Pair"].apply(lambda x: " | ".join(x))

# Plot the distribution of reasoning types
plt.figure(figsize=(10, 6))
reasoning_counts_df = reasoning_counts_df.sort_values("Count", ascending=False)
plt.bar(reasoning_counts_df["Reasoning Type"], reasoning_counts_df["Count"], color="skyblue")
plt.title("Distribution of Reasoning Types")
plt.xlabel("Reasoning Type")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

import seaborn as sns

# Pivot the data for a heatmap visualization
overlap_matrix = overlap_counts_df.copy()
overlap_matrix["Reasoning Pair"] = overlap_matrix["Reasoning Pair"].apply(lambda x: x.split(" | "))
overlap_matrix = overlap_matrix.explode("Reasoning Pair").pivot_table(
    index="Reasoning Pair", 
    columns="Reasoning Pair", 
    values="Count", 
    aggfunc="sum", 
    fill_value=0
)

# Create a heatmap of overlaps
plt.figure(figsize=(12, 8))
sns.heatmap(overlap_matrix, annot=True, fmt="d", cmap="Blues", cbar_kws={'label': 'Overlap Count'})
plt.title("Reasoning Type Pair Overlap Heatmap")
plt.ylabel("Reasoning Type 1")
plt.xlabel("Reasoning Type 2")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Output data for overlap counts
import ace_tools as tools; tools.display_dataframe_to_user(name="Reasoning Overlap Counts", dataframe=overlap_counts_df)