# Functions for feature engineering

# In[1]: Libaries

# Data manipulation
import pandas as pd 
import numpy as np

# Store and organize output files
from pathlib import Path

# Saving models
import pickle

# Time management
import time


# In[2]: Utility functions I: Model saving

def save_model_fe(algorithm: str, model, fold, current_time):
    # Create a directory to store the output files
    results_path = Path('./MODELOS')
    results_path.mkdir(exist_ok=True)

    # Name experiment algorithm + current time
    experiment_name = algorithm + '_' + current_time
    experiment_dir = results_path / experiment_name
    experiment_dir.mkdir(exist_ok=True)

    print(f'Saving model in {experiment_dir}')

    # Save model with save_model() function in json format in the experiment directory
    model_path = experiment_dir / f'{algorithm}_model_{str(fold)}.json'
    model.save_model(model_path)

    print('Model saved successfully')


# In[3]: Feature engineering functions I: Feature types

def feature_types(df):
    not_used = ['customer_ID', 'S_2'] # and target
    cat_features = ['B_30', 'B_38', 'D_63', 'D_64', 'D_66', 'D_68', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126']
    # These are the 'official' binary features in Kaggle notebooks
    bin_features_1 = ['B_31', 'D_87'] 
    # These takes 0 and 1
    bin_features_2 = ['R_2', 'S_6', 'R_4', 'R_15', 'S_18', 'D_86', 'B_31', 'R_19', 'B_32', 'S_20', 'R_21', 'R_22', 'R_23', 'D_93', 'D_94', 'R_24', 'R_25', 'D_96', 'D_127', 'R_28'] 
    # This takes -1 (NaN) and 1
    bin_features_3 = ['D_87'] 

    features = df.drop(not_used, axis = 1).columns.to_list()
    num_features = [col for col in features if col not in cat_features]

    return not_used, cat_features, bin_features_1, bin_features_2, bin_features_3, num_features #, features


# In[4]: Feature engineering functions II: Dummy encoding

"""Before creating new variables, we need to encode the categorical features.
If we dummy encode them we will have k-1 new features for each categorical feature instead of k,
so there will be less features in the final dataset and the dimensionality will be reduced.
"""

# def dummy_encoding_2(train_df, test_df, cat_features): 
#     # Dummy encoding
#     train_df_oh = pd.get_dummies(train_df, columns = cat_features, drop_first = True)
#     test_df_oh = pd.get_dummies(test_df, columns = cat_features, drop_first = True)

#     # List of the new dummy encoded features
#     dummies_train = list(set(train_df_oh.columns) - set(train_df.columns))
#     dummies_test = list(set(test_df_oh.columns) - set(test_df.columns))


#     return train_df_oh, test_df_oh, dummies_train, dummies_test

# Join train and test to perform dummy encoding and then separate them again
def dummy_encoding(train_df, test_df, cat_features):
    # Join train and test
    df = pd.concat([train_df, test_df], axis = 0)

    # Dummy encoding
    df_oh = pd.get_dummies(df, columns = cat_features, drop_first = True)

    # Separate train and test
    train_df_oh = df_oh.iloc[:train_df.shape[0], :]
    test_df_oh = df_oh.iloc[train_df.shape[0]:, :]

    # List of the new dummy encoded features
    dummies_train = list(set(train_df_oh.columns) - set(train_df.columns))
    dummies_test = list(set(test_df_oh.columns) - set(test_df.columns))

    return train_df_oh, test_df_oh, dummies_train, dummies_test


# In[5]: Feature engineering functions III: Aggregations

"""
Providing the models with group statistics (mean, std, max, min, etc.) in the form of new features
is a good way to determine if a value is typical or unusual compared to the rest of the group.
We will create to functions for this purpose: one for categorical features and one for numerical features.
"""

# Function that creates aggregated features for the variables of a dataset

def feat_aggregations(df, cat_features, num_features, groupby_var):
    # Group by the specified variable and calculate the statistics
    df_cat_agg = df.groupby(groupby_var)[cat_features].agg(['count', 'first', 'last', 'nunique'])
    df_num_agg = df.groupby(groupby_var)[num_features].agg(['mean', 'std', 'max', 'min', 'first', 'last'])

    # Strings list for aggregated features names
    df_cat_agg.columns = ['_'.join(col) for col in df_cat_agg.columns]
    df_num_agg.columns = ['_'.join(col) for col in df_num_agg.columns]

    # Reset index
    # df_cat_agg = df_cat_agg.reset_index(inplace = True)
    # df_num_agg = df_num_agg.reset_index(inplace = True)

    # Concat the aggregated features
    df = pd.concat([df_cat_agg, df_num_agg], axis = 1)

    del df_cat_agg, df_num_agg
    return df


# In[6]: Feature engineering functions IV: Differences

# Function that creates new features with the difference between two observations of the same customer for a given variable

def feat_diff(data, features, lag: int): # features = numerical_features
    df_diffs = []
    customer_ids = []
    for customer_id, cid_data in data.groupby(['customer_ID']):
        # Get the differences between last observation and last - lag observation 
        diff_lag = cid_data[features].diff(lag).iloc[[-1]].values.astype(np.float32)
        # Append to lists
        df_diffs.append(diff_lag)
        customer_ids.append(customer_id)
    # Concatenate
    df_diffs = np.concatenate(df_diffs, axis=0)
    # Transform to dataframe
    df_diffs = pd.DataFrame(df_diffs, columns=[col + f'_diff{lag}' for col in cid_data[features].columns])
    # Add customer id
    df_diffs['customer_ID'] = customer_ids
    return df_diffs

# Note: Just taking the diff between last and lag, do I need other differences? (last-k and last-k-j, etc.)


# In[7]: Feature engineering functions V: Lagged features

# Function that creates new features with the lagged values of a given variable
# PROBLEMA: GENERA UNA NUEVA OBSERVACIÓN PARA CADA LAG
def feat_lag(data, lags: list): # [1, 2, 3, 6, 11]
    lag_variables = []
    customer_ids = []

    # Iterate through each customer's data
    for customer_id, cid_data in data.groupby(['customer_ID']):
        # Order by date
        sorted_data = cid_data.sort_values('S_2', ascending=False)
        # Number of observations of each customer
        num_observations = len(sorted_data)
        
        # Initialize lag_variable dictionary for current customer
        lag_variable = {'customer_ID': customer_id}

        # Iterate through each lag
        for lag in lags:
            # Check if lag < num_observations
            if lag < num_observations:
                # Get lag observation
                lag_observation = sorted_data.iloc[lag]
                # Iterate through each column of the data (except customer_ID and S_2)
                for column in cid_data.columns:
                    if column not in ['customer_ID', 'S_2']:
                        lag_variable['{}_lag{}'.format(column, lag)] = lag_observation[column]
            else:
                # Fill with NaNs
                for column in cid_data.columns:
                    if column not in ['customer_ID', 'S_2']:
                        lag_variable['{}_lag{}'.format(column, lag)] = np.nan

        lag_variables.append(lag_variable)
        customer_ids.append(customer_id)

    # Concatenate
    lag_variables = pd.DataFrame(lag_variables)
    # Add customer id
    lag_variables['customer_ID'] = customer_ids

    return lag_variables


# In[8]: Feature engineering functions VI: Period means

# Function that creates new features with the mean of a given variable for a given period
# depending on the number of observations of each customer

# ES ETERNO Y DA PROBLEMAS, REVISAR

# def feat_period_means(data, features): # features = numerical_features
#     period_means = []
#     customer_ids = []
#     column_names = []
#     for customer_id, cid_data in data.groupby(['customer_ID']):
#         # We are going to calculate the mean of different groups of observations. If a customer
#         # does not have enough observations, we will fill with NaNs.

#         # Get the mean of the last 6 observations and the mean of the 3 observations before the last 3 observations
#         if len(cid_data) >= 6:
#             mean_last_6M = cid_data[features].iloc[-6:].mean().values.astype(np.float32)
#             mean_before_3M = cid_data[features].iloc[-6:-3].mean().values.astype(np.float32)
#         else:
#             mean_last_6M = np.full(len(features), np.nan)
#             mean_before_3M = np.full(len(features), np.nan)
#         # Append to lists    
#         column_names.extend([col + '_mean_last_6M' for col in features])
#         column_names.extend([col + '_mean_before_3M' for col in features])
#         period_means.extend([mean_last_6M, mean_before_3M])
        
#         # Get the mean of the last 3 observations
#         if len(cid_data) >= 3:
#             mean_last_3M = cid_data[features].iloc[-3:].mean().values.astype(np.float32)
#         else:
#             mean_last_3M = np.full(len(features), np.nan)
#         # Append to lists   
#         column_names.extend([col + '_mean_last_3M' for col in features])
#         period_means.append(mean_last_3M)

#         # Get the mean of the 6 observations before the last 6 observations
#         if len(cid_data) >= 12:
#             mean_before_6M = cid_data[features].iloc[-12:-6].mean().values.astype(np.float32)
#         else:
#             mean_before_6M = np.full(len(features), np.nan)
#         # Append to lists
#         column_names.extend([col + '_mean_before_6M' for col in features])
#         period_means.append(mean_before_6M)

#         # Now we compute the relative increments of the means
#         # Relative semestral growth
#         inc_12m_6m = (mean_last_6M - mean_before_6M) / mean_before_6M
#         # Relative trimestral growth
#         inc_3m_3m = (mean_last_3M - mean_before_3M) / mean_before_3M

#         # Append column names
#         column_names.extend([col + '_inc_12m_6m' for col in features])
#         column_names.extend([col + '_inc_3m_3m' for col in features])
#         period_means.extend([inc_12m_6m, inc_3m_3m])
        
#         # Append customer id
#         customer_ids.append(customer_id)

#     # Concatenate
#     period_means = np.concatenate(period_means, axis=0) #.reshape(-1, len(column_names))
#     # Transform to dataframe
#     period_means = pd.DataFrame(period_means, columns=column_names)

#     # Add customer id
#     period_means['customer_ID'] = customer_ids

#     return period_means

def feat_period_means(data, features): # features = numerical_features
    period_means = []
    customer_ids = []
    column_names = []
    for customer_id, cid_data in data.groupby(['customer_ID']):
        # We are going to calculate the mean of different groups of observations. If a customer
        # does not have enough observations, we will fill with NaNs.

        # Get the mean of the last 6 observations and the mean of the 3 observations before the last 3 observations
        if len(cid_data) >= 6:
            mean_last_6M = cid_data[features].iloc[-6:].mean().values.astype(np.float32)
            mean_before_3M = cid_data[features].iloc[-6:-3].mean().values.astype(np.float32)
            # Append column names
            column_names.extend([col + '_mean_last_6M' for col in features])
            column_names.extend([col + '_mean_before_3M' for col in features])
        else:
            mean_last_6M = np.full(len(features), np.nan)
            mean_before_3M = np.full(len(features), np.nan)
            # Append column names
            column_names.extend([col + '_mean_last_6M' for col in features])
            column_names.extend([col + '_mean_before_3M' for col in features])
        # Append to lists
        period_means.append(mean_last_6M)
        period_means.append(mean_before_3M)

        # Get the mean of the last 3 observations
        if len(cid_data) >= 3:
            mean_last_3M = cid_data[features].iloc[-3:].mean().values.astype(np.float32)
            # Append column names
            column_names.extend([col + '_mean_last_3M' for col in features])
        else:
            mean_last_3M = np.full(len(features), np.nan)
            # Append column names
            column_names.extend([col + '_mean_last_3M' for col in features])
        # Append to lists
        period_means.append(mean_last_3M)

        # Get the mean of the 6 observations before the last 6 observations
        if len(cid_data) >= 12:
            mean_before_6M = cid_data[features].iloc[-12:-6].mean().values.astype(np.float32)
            # Append column names
            column_names.extend([col + '_mean_before_6M' for col in features])
        else:
            mean_before_6M = np.full(len(features), np.nan)
            # Append column names
            column_names.extend([col + '_mean_before_6M' for col in features])
        # Append to lists
        period_means.append(mean_before_6M)

        # Now we compute the relative increments of the means
        # Relative semestral growth
        inc_12m_6m = (mean_last_6M - mean_before_6M) / mean_before_6M
        # Relative trimestral growth
        inc_3m_3m = (mean_last_3M - mean_before_3M) / mean_before_3M

        # Append column names
        column_names.extend([col + '_inc_12m_6m' for col in features])
        column_names.extend([col + '_inc_3m_3m' for col in features])

        # Append to lists
        period_means.append(inc_12m_6m)
        period_means.append(inc_3m_3m)
        customer_ids.append(customer_id)

    # Concatenate
    period_means = np.concatenate(period_means, axis=0)
    # Transform to dataframe
    period_means = pd.DataFrame(period_means, columns=column_names)

    # Add customer id
    period_means['customer_ID'] = customer_ids

    return period_means


# In[9]: Feature engineering functions VII: First/last observations

# Function that creates new features with the first and last observations of a given variable
# 1. Last - First: The change since we first see the client to the last time we see the client.
# 2. Last / First: The fractional difference since we first see the client to the last time we see the client.

def feat_last_diffdiv(agg_features): # df_num_agg
    # Iterate through features
    for feature in agg_features:

        # Check if last and first are in the feature name and compute difference and division
        if 'last' in feature and feature.replace('last', 'first') in agg_features:
            agg_features[feature + '_lag_sub_first'] = agg_features[feature] - agg_features[feature.replace('last', 'first')]
            agg_features[feature + '_lag_div_first'] = agg_features[feature] / agg_features[feature.replace('last', 'first')]

        # Check if last and mean are in the feature name and compute difference and division
        elif 'last' in feature and feature.replace('last', 'mean') in agg_features:
            agg_features[feature + '_lag_sub_mean'] = agg_features[feature] - agg_features[feature.replace('last', 'mean')]
            agg_features[feature + '_lag_div_mean'] = agg_features[feature] / agg_features[feature.replace('last', 'mean')]

    return agg_features


# %%
# In[6]: data
train = pd.read_parquet('C:/Users/Jose/Documents/UNIVERSIDAD/TFG/amex-default-prediction/parquet_ds_integer_dtypes/train.parquet')
# test_data = pd.read_parquet('C:/Users/Jose/Documents/UNIVERSIDAD/TFG/amex-default-prediction/parquet_ds_integer_dtypes/test.parquet')

# In[7]: tests

not_used, cat_features, bin_features_1, bin_features_2, bin_features_3, num_features = feature_types(train)

# train_oh, test_oh, list_dummies_train, list_dummies_test = dummy_encoding(train, test_data, cat_features)

# %%

# train_agg = feat_aggregations(train_oh, list_dummies_train, num_features, 'customer_ID')
# train_agg
# %%


# Code for retrieving a list with the customers that have only one observation (only one record in S_2)
# train['customer_ID'].value_counts()[train['customer_ID'].value_counts() == 1].index.tolist()
# b8d4227c0d88180958483c0db718554819048f7af142edb30d84068a7fb5ee17

