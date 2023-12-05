import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV
from sklearn.preprocessing import MinMaxScaler

def load_data(file_path):
    """Load the dataset from a CSV file"""
    return pd.read_csv(file_path)

def custom_group_shuffle_split(df, group_column, first_test_size=0.1, second_test_size=0.222, random_state=42):
    """Perform a custom group shuffle split based on a specified column"""
    group_shuffle = GroupShuffleSplit(n_splits=2, test_size=first_test_size, random_state=random_state)
    for train_validation_idx, test_idx in group_shuffle.split(df, groups=df[group_column]):
        train_validation_set = df.iloc[train_validation_idx]
        test_set = df.iloc[test_idx]

    group_shuffle = GroupShuffleSplit(n_splits=2, test_size=second_test_size, random_state=random_state)
    for train_idx, validation_idx in group_shuffle.split(train_validation_set, groups=train_validation_set[group_column]):
        train_set = train_validation_set.iloc[train_idx]
        validation_set = train_validation_set.iloc[validation_idx]

    return train_set, validation_set, test_set

def grid_search_lambdamart(train_set, validation_set, feature_cols, target_col):
    """Perform GridSearchCV for LambdaMART model"""
    train_data = lgb.Dataset(train_set[feature_cols], label=train_set[target_col], group=train_set['query_id'])
    validation_data = lgb.Dataset(validation_set[feature_cols], label=validation_set[target_col], group=validation_set['query_id'])

    parameters = {
        'objective': ['lambdarank'],
        'boosting_type': ['gbdt'],
        'metric': ['ndcg'],
        'num_leaves': [31],
        'max_depth': [-1],
        'learning_rate': [0.1, 0.05, 0.01],
        'n_estimators': [50, 100, 150]
    }

    gsearch = GridSearchCV(estimator=lgb.LGBMRanker(), param_grid=parameters, scoring='neg_mean_squared_error')
    gsearch.fit(
        train_set[feature_cols], 
        train_set[target_col], 
        groups=train_set['query_id'],
        eval_set=[(validation_set[feature_cols], validation_set[target_col])],
        eval_group=[validation_set['query_id']],
        eval_metric='ndcg'
    )

    return gsearch.best_estimator_

def extract_and_save_feature_importances(model, df, feature_cols, usergroup_column, output_file):
    """Extract, normalize, and save feature importances"""
    importances = model.feature_importances_
    scaler = MinMaxScaler()
    normalized_importances = scaler.fit_transform(np.array(importances).reshape(-1, 1)).flatten()
    feature_weights = dict(zip(feature_cols, normalized_importances))

    #Calculate weighted scores for each usergroup
    weighted_scores = df.groupby(usergroup_column).apply(lambda x: np.dot(x[feature_cols], feature_weights))

    #Save to CSV
    weighted_scores.to_csv(output_file)

#Example usage
df = load_data('path_to_dataset.csv')
feature_cols = ['feature_1', 'feature_2', 'feature_3']
target_col = 'true_label'
train_set, validation_set, test_set = custom_group_shuffle_split(df, 'query_id')
best_model = grid_search_lambdamart(train_set, validation_set, feature_cols, target_col)

# Extract and save feature importances per user group
extract_and_save_feature_importances(best_model, df, feature_cols, 'usergroup_id', 'feature_importances.csv')
