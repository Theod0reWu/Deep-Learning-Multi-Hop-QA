import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Create figure and axis
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_xlim(0, 12)
ax.set_ylim(0, 6)

# Function to create a box with text
def create_box(x, y, width, height, text, color='lightblue'):
    box = patches.Rectangle((x, y), width, height, facecolor=color, edgecolor='black', alpha=0.3)
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text, ha='center', va='center', wrap=True)

# Function to create an arrow
def create_arrow(start, end, text=''):
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->'))
    if text:
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        ax.text(mid_x, mid_y + 0.2, text, ha='center', va='bottom')

# Create boxes
create_box(1, 2, 2, 1, 'Input Prompt\n(Text)')
create_box(4, 2, 2, 1, 'TF-IDF\nVectorizer')
create_box(7, 2, 2, 1, 'Neural\nNetwork')
create_box(10, 2, 1.5, 1, 'Predicted\nHop Count')

# Create training label box
create_box(7, 4, 2, 1, 'Training Labels\n(Hop Counts)', color='lightgreen')

# Create arrows
create_arrow((3, 2.5), (4, 2.5), 'Text input')
create_arrow((6, 2.5), (7, 2.5), 'Feature Vector')
create_arrow((9, 2.5), (10, 2.5), 'Prediction')
create_arrow((8, 4), (8, 3), 'Training')

# Remove axes
ax.set_xticks([])
ax.set_yticks([])

# Add title
plt.title('TFIDF-Neural Network Architecture for Hop Count Prediction')

# Save the diagram
plt.savefig('/Users/avinair/Desktop/DeepLearningProject/Deep-Learning-Multi-Hop-QA/src/architecture_diagram.png', 
            bbox_inches='tight', dpi=300)
plt.close()
