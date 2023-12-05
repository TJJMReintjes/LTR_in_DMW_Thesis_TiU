import matplotlib.pyplot as plt
import pandas as pd

#Load the datasets
baseline_relevance_scores = pd.read_csv('path_to_baseline_relevance_scores')
dynamic_relevance_scores = pd.read_csv('path_dynamic_relevance_scores.csv')

#Overlay Histogram
plt.figure(figsize=(12, 8))
plt.hist(baseline_relevance_scores['relevance_score'], bins=50, alpha=0.5, label='Baseline Model', color='lightcoral', edgecolor='black')
plt.hist(dynamic_relevance_scores['relevance_score'], bins=50, alpha=0.5, label='Dynamic Model', color='skyblue', edgecolor='black')
plt.title('Relevance Score Baseline vs. Relevance Score Dynamic Model')
plt.xlabel('Total Relevance')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.show()

#Density Plot
plt.figure(figsize=(10, 6))
sns.kdeplot(baseline_relevance_scores['relevance_score'], shade=True, color="lightcoral", label='Baseline Model')
sns.kdeplot(dynamic_relevance_scores['relevance_score'], shade=True, color="skyblue", label='Dynamic Model')
plt.title('Relevance Score Baseline vs. Relevance Score Dynamic Model')
plt.xlabel('Total Relevance')
plt.ylabel('Density')
plt.legend()
plt.show()


#Print Basic Statistics
statistics = {
    'Mean': baseline_relevance_scores['relevance_score'].mean(),
    'Median': baseline_relevance_scores['relevance_score'].median(),
    'Std Deviation': baseline_relevance_scores['relevance_score'].std(),
    'Min': baseline_relevance_scores['relevance_score'].min(),
    'Max': baseline_relevance_scores['relevance_score'].max(),
}

statistics_dynamic = {
    'Mean': dynamic_relevance_scores['relevance_score'].mean(),
    'Median': dynamic_relevance_scores['relevance_score'].median(),
    'Std Deviation': dynamic_relevance_scores['relevance_score'].std(),
    'Min': dynamic_relevance_scores['relevance_score'].min(),
    'Max': dynamic_relevance_scores['relevance_score'].max(),
}

stats_df = pd.DataFrame({'Baseline': statistics, 'Dynamic': statistics_dynamic})

#Export to latex table
stats_df.to_latex()
print(stats_df.to_latex())
