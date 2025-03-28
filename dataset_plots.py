import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set modern style with white background
plt.style.use('default')
sns.set_palette("husl", 8)
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12

# Load the dataset
df = pd.read_csv('dataset.csv')

# === 1. Bar Plot - Percentage of True values per feature ===
feature_cols = [col for col in df.columns if any(prefix in col for prefix in ['clarity', 'value', 'experience', 'quality'])]
feature_percentages = df[feature_cols].mean() * 100

plt.figure(figsize=(14, 7))
ax = feature_percentages.plot(kind='bar', color=sns.color_palette("husl", len(feature_percentages)))
plt.title('Percentage of True Values per Feature', pad=20, fontsize=16, fontweight='bold')
plt.xlabel('Features', fontsize=12)
plt.ylabel('Percentage (%)', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
for i, v in enumerate(feature_percentages):
    plt.text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=8)
plt.tight_layout()
plt.savefig('bar_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# === 2. Heatmap - Feature correlations ===
plt.figure(figsize=(12, 10))
correlation_matrix = df[feature_cols].corr()
ax = sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0,
                 square=True, linewidths=0.5, cbar_kws={'shrink': .5})
plt.title('Correlation Between Binary Features', pad=20, fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# === 3. Pie Chart - Rows with ALL Quality Features True ===
quality_cols = [col for col in df.columns if 'quality' in col]
has_all_quality = df[quality_cols].all(axis=1).value_counts(normalize=True) * 100

# Reverse colors so "All True" is second color
colors = sns.color_palette("husl", 2)[::-1]

plt.figure(figsize=(8, 8))
plt.pie(has_all_quality, labels=['All Quality Features True', 'Not All True'],
        autopct='%1.1f%%', colors=colors,
        wedgeprops={'edgecolor': 'white', 'linewidth': 2},
        textprops={'fontsize': 12, 'fontweight': 'bold'})
plt.title('Rows with All Quality Features True', pad=20, fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('pie_chart_all_quality_true.png', dpi=300, bbox_inches='tight')
plt.show()

# === 3b. Pie Chart - Distribution of Defects Among Quality Features ===
false_counts = (df[quality_cols] == False).sum()
false_distribution = false_counts / false_counts.sum() * 100

plt.figure(figsize=(8, 8))
plt.pie(false_distribution, labels=false_distribution.index,
        autopct='%1.1f%%', colors=sns.color_palette("husl", len(false_distribution)),
        wedgeprops={'edgecolor': 'white', 'linewidth': 2},
        textprops={'fontsize': 12, 'fontweight': 'bold'})
plt.title('Percentage Distribution of Defects in Quality Features', pad=20, fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('pie_chart_defect_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# === 4. Stacked Bar Chart - Feature Categories ===
categories = {
    'Clarity': [col for col in df.columns if 'clarity' in col],
    'Value': [col for col in df.columns if 'value' in col],
    'Experience': [col for col in df.columns if 'experience' in col],
    'Quality': [col for col in df.columns if 'quality' in col]
}
category_means = {cat: df[cols].mean().mean() * 100 for cat, cols in categories.items()}
category_df = pd.DataFrame(category_means.items(), columns=['Category', 'Percentage'])

plt.figure(figsize=(10, 6))
ax = sns.barplot(data=category_df, x='Category', y='Percentage',
                 palette="husl", edgecolor='white', linewidth=2)
plt.title('Average Percentage of True Values by Category', pad=20, fontsize=16, fontweight='bold')
plt.ylabel('Percentage (%)', fontsize=12)
plt.xlabel('Category', fontsize=12)
for i, v in enumerate(category_df['Percentage']):
    plt.text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig('category_bar.png', dpi=300, bbox_inches='tight')
plt.show()
