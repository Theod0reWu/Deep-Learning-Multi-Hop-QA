import matplotlib.pyplot as plt
import numpy as np

# Data for GPT and Gemini models
categories = ['Multiple constraints', 'Temporal reasoning', 'Numerical reasoning', 
              'Post processing', 'Tabular reasoning']

gpt_accuracies = [0.06792452830188679, 0.08, 0.08620689655172414, 
                  0.0, 0.04878048780487805]

gemini_accuracies = [0.1471698113207547, 0.2, 0.1724137931034483,
                    0.14285714285714285, 0.0975609756097561]

# Set up the bar positions
x = np.arange(len(categories))
width = 0.35  # Width of the bars

# Create the figure and axis
fig, ax = plt.subplots(figsize=(12, 6))

# Create the bars
rects1 = ax.bar(x - width/2, gpt_accuracies, width, label='GPT', color='blue', alpha=0.7)
rects2 = ax.bar(x + width/2, gemini_accuracies, width, label='Gemini', color='red', alpha=0.7)

# Customize the plot
ax.set_ylabel('Accuracy')
ax.set_title('Model Performance Comparison by Category')
ax.set_xticks(x)
ax.set_xticklabels(categories, rotation=45, ha='right')
ax.legend()

# Add value labels on top of each bar
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', rotation=0)

autolabel(rects1)
autolabel(rects2)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot
plt.savefig('model_comparison.png')
