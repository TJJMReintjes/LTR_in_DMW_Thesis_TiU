import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

#Load the dataset
df = pd.read_csv('feature_importances_path.csv')  # Replace with your CSV file path

outliers = df[df['feature_2'] > 0.5]
averages = df.mean()

#Set the 'usergroup_id' as the index
df.set_index('usergroup_id', inplace=True)

#Plot the feature importances for each user
plt.figure(figsize=(10, 15))
sns.heatmap(df.T, cmap='YlGnBu', cbar_kws={'label': 'Feature Importance'})
plt.title('Average Feature Importances per Usergroup')
plt.xlabel('Usergroup)
plt.ylabel('Features')
plt.yticks(rotation=0)
plt.show()

print(averages, outliers.head())
