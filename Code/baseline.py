import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt

def baseline(df, weights):
    """Calculate the baseline relevance score with fixed weights.
    :param df: DataFrame containing the dataset with model scores features.
    :param weights: Dictionary with weights for each feature"""
    
    return df['feature_1'] * weights['feature_1'] + df['feature_2'] * weights['feature_2'] + df['feature_3'] * weights['feature_3']

def evaluate_model(predictions, true_labels):
    """Evaluate the model by calculating metrics and plotting a confusion matrix.
    :param predictions: Array of predicted labels.
    :param true_labels: Array of true labels """
    print("F1 Score:", f1_score(true_labels, predictions, average='binary'))
    print("Precision:", precision_score(true_labels, predictions, average='binary'))
    print("Recall:", recall_score(true_labels, predictions, average='binary'))

    # Plotting confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

#Main Logic
df = pd.read_csv('path_to_dataset.csv')
weights = {'feature_1': 0.5, 'feature_2': 0.3, 'feature_3': 0.1}
baseline_scores = baseline(df, weights)

#Binarize scores for comparison
threshold = 40  #Adjust the threshold as needed
predicted_labels = (baseline_scores > threshold).astype(int)

#Evaluate the baseline model
evaluate_model(predicted_labels, df['true_label'])
