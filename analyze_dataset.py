import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# Define the class names
class_names = ['battery', 'can', 'cardboard_bowl', 'cardboard_box', 'chemical_plastic_bottle',
               'chemical_plastic_gallon', 'chemical_spray_can', 'light_bulb', 'paint_bucket',
               'plastic_bag', 'plastic_bottle', 'plastic_bottle_cap', 'plastic_box',
               'plastic_cultery', 'plastic_cup', 'plastic_cup_lid', 'reuseable_paper',
               'scrap_paper', 'scrap_plastic', 'snack_bag', 'stick', 'straw', 'Beef',
               'Cabbage', 'Carrot', 'Chicken', 'Cucumber', 'Egg', 'Eggplant', 'Leek',
               'Onion', 'Pork', 'Potato', 'Radish', 'Tomato']

# Base directory
base_dir = Path(
    r'c:\Users\ayush\OneDrive\Pictures\Documents\Desktop\waste dataset 10')

# Function to count images per class in a split


def count_images_per_class(label_dir):
    class_counter = Counter()

    if not label_dir.exists():
        return class_counter

    # Go through all label files
    for label_file in label_dir.glob('*.txt'):
        with open(label_file, 'r') as f:
            for line in f:
                if line.strip():
                    class_id = int(line.split()[0])
                    class_counter[class_id] += 1

    return class_counter


# Count for each split
train_counts = count_images_per_class(base_dir / 'train' / 'labels')
valid_counts = count_images_per_class(base_dir / 'valid' / 'labels')
test_counts = count_images_per_class(base_dir / 'test' / 'labels')

# Combine all counts
total_counts = Counter()
for c in [train_counts, valid_counts, test_counts]:
    total_counts.update(c)

# Prepare data for visualization
class_ids = list(range(len(class_names)))
train_data = [train_counts.get(i, 0) for i in class_ids]
valid_data = [valid_counts.get(i, 0) for i in class_ids]
test_data = [test_counts.get(i, 0) for i in class_ids]
total_data = [total_counts.get(i, 0) for i in class_ids]

# Print statistics
print("Dataset Distribution Analysis")
print("=" * 80)
print(f"\nTotal classes: {len(class_names)}")
print(f"\nInstances per split:")
print(f"  Train: {sum(train_data)}")
print(f"  Valid: {sum(valid_data)}")
print(f"  Test: {sum(test_data)}")
print(f"  Total: {sum(total_data)}")

print("\n" + "=" * 80)
print("\nInstances per class (Total):")
print("-" * 80)
for i, name in enumerate(class_names):
    print(
        f"{name:30s}: {total_data[i]:4d} (Train: {train_data[i]:3d}, Valid: {valid_data[i]:3d}, Test: {test_data[i]:3d})")

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(24, 16))

# 1. Total distribution - Bar chart
ax1 = axes[0, 0]
bars = ax1.barh(class_names, total_data, color='steelblue')
ax1.set_xlabel('Number of Instances', fontsize=12, fontweight='bold')
ax1.set_title('Total Instance Distribution per Class',
              fontsize=14, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)
# Add value labels on bars
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax1.text(width, bar.get_y() + bar.get_height()/2, f' {int(width)}',
             ha='left', va='center', fontsize=9)

# 2. Stacked bar chart for train/valid/test split
ax2 = axes[0, 1]
x_pos = np.arange(len(class_names))
p1 = ax2.bar(x_pos, train_data, label='Train', color='#2ecc71')
p2 = ax2.bar(x_pos, valid_data, bottom=train_data,
             label='Valid', color='#3498db')
p3 = ax2.bar(x_pos, test_data, bottom=np.array(train_data) + np.array(valid_data),
             label='Test', color='#e74c3c')
ax2.set_xlabel('Class', fontsize=12, fontweight='bold')
ax2.set_ylabel('Number of Instances', fontsize=12, fontweight='bold')
ax2.set_title('Train/Valid/Test Split per Class',
              fontsize=14, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(class_names, rotation=45, ha='right', fontsize=8)
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# 3. Pie chart showing proportion of total instances
ax3 = axes[1, 0]
# Only show classes with instances
non_zero_indices = [i for i, count in enumerate(total_data) if count > 0]
non_zero_names = [class_names[i] for i in non_zero_indices]
non_zero_counts = [total_data[i] for i in non_zero_indices]

colors = plt.cm.tab20(np.linspace(0, 1, len(non_zero_names)))
wedges, texts, autotexts = ax3.pie(non_zero_counts, labels=non_zero_names, autopct='%1.1f%%',
                                   colors=colors, startangle=90)
for text in texts:
    text.set_fontsize(8)
for autotext in autotexts:
    autotext.set_fontsize(7)
    autotext.set_color('white')
    autotext.set_fontweight('bold')
ax3.set_title('Proportion of Total Instances per Class',
              fontsize=14, fontweight='bold')

# 4. Sorted bar chart
ax4 = axes[1, 1]
sorted_indices = np.argsort(total_data)[::-1]
sorted_names = [class_names[i] for i in sorted_indices]
sorted_counts = [total_data[i] for i in sorted_indices]
colors_sorted = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(sorted_names)))
bars = ax4.barh(sorted_names, sorted_counts, color=colors_sorted)
ax4.set_xlabel('Number of Instances', fontsize=12, fontweight='bold')
ax4.set_title('Classes Sorted by Instance Count',
              fontsize=14, fontweight='bold')
ax4.grid(axis='x', alpha=0.3)
# Add value labels
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax4.text(width, bar.get_y() + bar.get_height()/2, f' {int(width)}',
             ha='left', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('dataset_analysis.png', dpi=300, bbox_inches='tight')
print("\n" + "=" * 80)
print("\n✅ Visualization saved as 'dataset_analysis.png'")
plt.show()
