import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('path_to_filename.csv', delimiter=",", header=0)

feature_1_scores = data['feature_1']
feature_2_scores = data['feature_2']
feature_3_scores = data['feature_3']
relevance_scores = data['relevance_score']

plt.figure(figsize=(12,8))

plt.subplot(2, 2, 1)
plt.hist(feature_1_scores, bins=100, color='bisque', edgecolor='black')
plt.title('Feature 1 distribution')
plt.xlabel('Feature 1 Score')
plt.ylabel('Frequency')

plt.subplot(2, 2, 2)
plt.hist(feature_2_scores, bins=30, color='skyblue', edgecolor='black')
plt.title('feature_2 Score distribution')
plt.xlabel('feature_2 Score')
plt.ylabel('Frequency')

plt.subplot(2, 2, 3)
plt.hist(feature_3_scores, bins=30, color='mediumseagreen', edgecolor='black')
plt.title('feature_3 Score distribution')
plt.xlabel('feature_3 Score')
plt.ylabel('Frequency')

plt.subplot(2, 2, 4 )
plt.hist(relevance_scores, bins=100, color='lightcoral', edgecolor='black')
plt.title('Relevance Score distribution')
plt.xlabel('Relevance Score')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
