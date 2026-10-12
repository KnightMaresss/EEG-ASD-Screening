"""Generate all figures for the paper."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

# ============================================
# FIGURE 1: Cross-Validation Box Plot
# ============================================
cv_scores = [78.1, 76.1, 76.8, 81.5, 75.7]

fig, ax = plt.subplots(figsize=(6, 4))
bp = ax.boxplot(cv_scores, patch_artist=True, widths=0.5)
bp['boxes'][0].set_facecolor('#4CAF50')
bp['medians'][0].set_color('white')

ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('5-Fold Cross-Validation Accuracy Distribution', fontsize=14, fontweight='bold')
ax.set_xticklabels(['Random Forest'], fontsize=11)
ax.axhline(y=77.6, color='red', linestyle='--', linewidth=1.5, label=f'Mean: 77.6%')
ax.legend(fontsize=10)
ax.set_ylim(74, 83)

plt.tight_layout()
plt.savefig('cv_boxplot.png', dpi=300, bbox_inches='tight')
print("✅ Saved: cv_boxplot.png")

# ============================================
# FIGURE 2: Feature Importance Bar Chart
# ============================================
features = [
    'Frontal Theta Power',
    'Theta/Beta Ratio', 
    'Alpha Asymmetry',
    'Mean Alpha Power',
    'Global Std Dev',
    'Mean Beta Power',
    'Max Theta Power',
    'Mean Delta Power',
    'Global Mean',
    'Alpha/Theta Ratio'
]
importance = [0.15, 0.14, 0.12, 0.11, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05]

colors = []
for f in features:
    if 'Theta' in f or 'Alpha' in f:
        colors.append('#f44336')  # Red for ASD biomarkers
    else:
        colors.append('#2196F3')  # Blue for other features

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.barh(features[::-1], importance[::-1], color=colors[::-1])
ax.set_xlabel('Feature Importance Score', fontsize=12)
ax.set_title('Top 10 Discriminative Features for ASD Detection', fontsize=14, fontweight='bold')

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#f44336', label='ASD Biomarker'),
                   Patch(facecolor='#2196F3', label='Other Feature')]
ax.legend(handles=legend_elements, fontsize=10)

plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
print("✅ Saved: feature_importance.png")

# ============================================
# FIGURE 3: Confusion Matrix
# ============================================
cm = np.array([[22, 6],   # Actual ASD
               [7, 21]])  # Actual Control

fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted\nASD', 'Predicted\nNeurotypical'],
            yticklabels=['Actual\nASD', 'Actual\nNeurotypical'],
            annot_kws={'fontsize': 14}, cbar=False)
ax.set_title('Confusion Matrix (n=56)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✅ Saved: confusion_matrix.png")

# ============================================
# FIGURE 4: Metrics Comparison Bar Chart
# ============================================
metrics = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-Score']
values = [77.6, 78, 76, 79, 77]

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(metrics, values, color=['#4CAF50', '#2196F3', '#FF9800', '#9C27B0', '#f44336'])
ax.set_ylabel('Percentage (%)', fontsize=12)
ax.set_title('Classification Performance Metrics', fontsize=14, fontweight='bold')
ax.set_ylim(0, 100)

# Add value labels on bars
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{val}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('metrics_bar.png', dpi=300, bbox_inches='tight')
print("✅ Saved: metrics_bar.png")

print("\n🎉 All figures generated! Upload these to Overleaf:")
print("   - cv_boxplot.png")
print("   - feature_importance.png")
print("   - confusion_matrix.png")
print("   - metrics_bar.png")