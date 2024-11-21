import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Function to draw a box with text
def draw_box(ax, x, y, width, height, text, color):
    box = FancyBboxPatch(
        (x, y), width, height, boxstyle="round,pad=0.3",
        edgecolor="black", facecolor=color, linewidth=1.5
    )
    ax.add_patch(box)
    ax.text(
        x + width / 2, y + height / 2, text,
        color="black", fontsize=10, ha="center", va="center", weight="bold"
    )

# Function to draw arrows between boxes
def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(facecolor='black', arrowstyle='->', linewidth=1.5)
    )

# Create the diagram
fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(0, 12)
ax.set_ylim(0, 8)
ax.axis('off')

# Step 1: Input Data Preparation
draw_box(ax, 1, 6.5, 2.5, 1, "Input Data\nPreparation", "lightblue")
ax.text(1.25, 6.2, "21 Key Points\n(X, Y, Z)", fontsize=8, ha="center", color="blue")
ax.text(1.25, 5.8, "Check for hand detection\n& calculate dimensions", fontsize=7, ha="center", color="blue")
draw_arrow(ax, 3.5, 7, 4, 7)

# Step 2: Feature Engineering
draw_box(ax, 4, 6.5, 3, 1, "Feature\nEngineering", "lightgreen")
ax.text(5.5, 6.2, "Angle Calculation\nNormalize Key Points", fontsize=8, ha="center", color="green")
draw_arrow(ax, 7.5, 7, 8, 7)

# Step 3: Temporal Data Creation
draw_box(ax, 8, 6.5, 3, 1, "Temporal Data\nCreation", "lightpink")
ax.text(9.5, 6.2, "Create Time Series\n& Handle Missing Data", fontsize=8, ha="center", color="purple")
draw_arrow(ax, 5.5, 5.5, 5.5, 4)

# Step 4: Model Training (CNN + LSTM)
draw_box(ax, 4, 3, 4, 1.5, "Model Training\n(CNN + LSTM)", "gold")
ax.text(6, 2.6, "Spatial & Temporal Feature\nExtraction", fontsize=8, ha="center", color="orange")
draw_arrow(ax, 6, 2.5, 6, 1.5)

# Step 5: Classification or Regression
draw_box(ax, 5, 1, 4, 1.5, "Classification or\nRegression", "salmon")
ax.text(7, 0.6, "Softmax for classification\nLinear for regression", fontsize=8, ha="center", color="red")
draw_arrow(ax, 9, 1.75, 10, 1.75)

# Step 6: Output Layer
draw_box(ax, 10, 1, 2.5, 1.5, "Output Layer\nResults", "violet")
ax.text(11.25, 0.6, "Real-time feedback\n& hand pose analysis", fontsize=8, ha="center", color="purple")

# Connecting arrows for intermediate steps
draw_arrow(ax, 2.5, 5.5, 2.5, 5)
draw_arrow(ax, 9.5, 5.5, 9.5, 5)

# Annotations for architecture
ax.text(1, 7.5, "Step 1: Input", fontsize=9, weight="bold", color="black")
ax.text(4.5, 7.5, "Step 2: Features", fontsize=9, weight="bold", color="black")
ax.text(8.5, 7.5, "Step 3: Temporal Data", fontsize=9, weight="bold", color="black")
ax.text(5.5, 4.5, "Step 4: Model Training", fontsize=9, weight="bold", color="black")
ax.text(7, 2.5, "Step 5: Prediction", fontsize=9, weight="bold", color="black")
ax.text(11, 2.5, "Step 6: Output", fontsize=9, weight="bold", color="black")

# Display the diagram
plt.tight_layout()
plt.show()
