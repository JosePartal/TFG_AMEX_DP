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
from tqdm import tqdm # Progress bar


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

def feat_aggregations(df, cat_features, num_features, groupby_var = 'customer_ID'):
    # Group by the specified variable and calculate the statistics
    df_cat_agg = df.groupby(groupby_var)[cat_features].agg(['count', 'first', 'last', 'nunique'])
    df_num_agg = df.groupby(groupby_var)[num_features].agg(['mean', 'std', 'max', 'min', 'first', 'last', 'median'])

    # Strings list for aggregated features names
    df_cat_agg.columns = ['_'.join(col) for col in df_cat_agg.columns]
    df_num_agg.columns = ['_'.join(col) for col in df_num_agg.columns]

    return df_cat_agg, df_num_agg


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
    # Set customer_ID as index
    df_diffs.set_index('customer_ID', inplace=True)

    return df_diffs

# Note: Just taking the diff between last and lag, do I need other differences? (last-k and last-k-j, etc.)


# In[7]: Feature engineering functions V: Lagged features

# Function that creates new features with the lagged values of a given variable
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
    # Set customer_ID as index
    lag_variables.set_index('customer_ID', inplace=True)

    return lag_variables


# In[8]: Feature engineering functions VI: Period means

# Function that creates new features with the mean of a given variable for a given period
# depending on the number of observations of each customer

# Code for computing the mean of the last 6 observations for all customers for the numerical features
# filling with NaNs the feature's values if the customer has less than 6 observations
# mean_test = train.groupby('customer_ID')[num_features].apply(lambda x: x.iloc[-6:].mean())

# Convert to NaN the mean values of the customers in mean_test with less than 6 observations in train dataframe
# mean_test_nan = mean_test.where(train['customer_ID'].value_counts() >= 6, np.nan)

""" The fastest way I have achieved to compute all these features is by 
    calculating the means for all the customers and then replace with NaNs the customers
    with less than N observations."""

def feat_period_means(data, features): # features = numerical_features
    # Compute the mean of the last 6 observations for all customers for the numerical features
    mean_last_6M = data.groupby('customer_ID')[features].apply(lambda x: x.iloc[-6:].mean())
    print('Mean last 6M computed')
    # Compute the mean of the last 3 observations for all customers for the numerical features
    mean_last_3M = data.groupby('customer_ID')[features].apply(lambda x: x.iloc[-3:].mean())
    print('Mean last 3M computed')
    # Compute the mean of the 6 observations before the last 6 observations
    mean_before_last_6M = data.groupby('customer_ID')[features].apply(lambda x: x.iloc[-12:-6].mean())
    print('Mean before last 6M computed')
    # Compute the mean of the 3 observations before the last 3 observations
    mean_before_last_3M = data.groupby('customer_ID')[features].apply(lambda x: x.iloc[-6:-3].mean())
    print('Mean before last 3M computed')
    # Compute the mean of the first 3 observations (only want customers with 13 observations)
    mean_first_3M = data.groupby('customer_ID')[features].apply(lambda x: x.iloc[:3].mean())
    print('Mean first 3M computed')


    # Convert to NaN the mean values of the customers with less than 6 observations in train dataframe
    mean_last_6M = mean_last_6M.where(data['customer_ID'].value_counts() >= 6, np.nan)
    print('Mean last 6M converted to NaNs for customers with less than 6 observations')
    # Convert to NaN the mean values of the customers with less than 3 observations in train dataframe
    mean_last_3M = mean_last_3M.where(data['customer_ID'].value_counts() >= 3, np.nan)
    print('Mean last 3M converted to NaNs for customers with less than 3 observations')
    # Convert to NaN the mean values of the customers with less than 6 observations in train dataframe
    mean_before_last_6M = mean_before_last_6M.where(data['customer_ID'].value_counts() >= 12, np.nan)
    print('Mean before last 6M converted to NaNs for customers with less than 12 observations')
    # Convert to NaN the mean values of the customers with less than 3 observations in train dataframe
    mean_before_last_3M = mean_before_last_3M.where(data['customer_ID'].value_counts() >= 6, np.nan)
    print('Mean before last 3M converted to NaNs for customers with less than 6 observations')
    # Convert to NaN the mean values of the customers with less than 13 observations in train dataframe
    mean_first_3M = mean_first_3M.where(data['customer_ID'].value_counts() == 13, np.nan)
    print('Mean first 3M converted to NaNs for customers with less than 13 observations')

    # Compute the relative difference between the mean of the last 6 observations and the mean of the 6 observations before
    # the last 6 observations
    inc_12m_6m = (mean_last_6M - mean_before_last_6M) / mean_before_last_6M
    # Convert to NaN if the denominator is 0 (need it to avoid inf values)
    inc_12m_6m = inc_12m_6m.where(mean_before_last_6M != 0, np.nan)
    print('Relative difference mean last 6M computed')
    # Compute the relative difference between the mean of the last 3 observations and the mean of the 3 observations before
    # the last 3 observations
    inc_6m_3m = (mean_last_3M - mean_before_last_3M) / mean_before_last_3M
    # Convert to NaN if the denominator is 0
    inc_6m_3m = inc_6m_3m.where(mean_before_last_3M != 0, np.nan)
    print('Relative difference mean last 3M computed')
    # Compute the relative difference between the mean of the last 3 observations and the mean of the first 3 observations
    inc_last3m_first3m = (mean_last_3M - mean_first_3M) / mean_first_3M
    # Convert to NaN if the denominator is 0
    inc_last3m_first3m = inc_last3m_first3m.where(mean_first_3M != 0, np.nan)
    print('Relative difference mean last 3M and mean first 3M computed')

    # Change column names
    mean_last_6M.columns = [col + '_mean_last_6M' for col in mean_last_6M.columns]
    mean_last_3M.columns = [col + '_mean_last_3M' for col in mean_last_3M.columns]
    mean_before_last_6M.columns = [col + '_mean_6M_before_last_6M' for col in mean_before_last_6M.columns]
    mean_before_last_3M.columns = [col + '_mean_3M_before_last_3M' for col in mean_before_last_3M.columns]
    inc_12m_6m.columns = [col + '_inc_12m_6m' for col in inc_12m_6m.columns]
    inc_6m_3m.columns = [col + '_inc_6m_3m' for col in inc_6m_3m.columns]
    inc_last3m_first3m.columns = [col + '_inc_last3m_first3m' for col in inc_last3m_first3m.columns]

    # Concatenate
    mean_df = pd.concat([mean_last_6M, mean_last_3M, mean_before_last_6M, mean_before_last_3M, 
                         inc_12m_6m, inc_6m_3m, inc_last3m_first3m], axis=1)

    return mean_df


# In[9]: Feature engineering functions VII: First/last observations

# Function that creates new features with the first and last observations of a given variable
# 1. Last - First: The change since we first see the client to the last time we see the client.
# 2. Last / First: The fractional difference since we first see the client to the last time we see the client.
# 3. Last - xxx: ['first','mean','std','median','min','max']
# 4. Last / xxx: ['first','mean','std','median','min','max']

def feat_last_diffdiv(df, cat_features, num_features, groupby_var = 'customer_ID'): 
    # Call the feat_aggregations function to get the aggregated features
    df_cat_agg, df_num_agg = feat_aggregations(df, cat_features, num_features, groupby_var)

    # Iterate through features
    for feature in df_num_agg:

        # Check if last and first are in the feature name and compute difference and division
        if 'last' in feature and feature.replace('last', 'first') in df_num_agg:
            df_num_agg[feature + '_last_sub_first'] = df_num_agg[feature] - df_num_agg[feature.replace('last', 'first')]
            df_num_agg[feature + '_last_div_first'] = df_num_agg[feature] / df_num_agg[feature.replace('last', 'first')]

        # Check if last and mean are in the feature name and compute difference and division
        elif 'last' in feature and feature.replace('last', 'mean') in df_num_agg:
            df_num_agg[feature + '_last_sub_mean'] = df_num_agg[feature] - df_num_agg[feature.replace('last', 'mean')]
            df_num_agg[feature + '_last_div_mean'] = df_num_agg[feature] / df_num_agg[feature.replace('last', 'mean')]

        # Check if last and std are in the feature name and compute difference and division
        elif 'last' in feature and feature.replace('last', 'std') in df_num_agg:
            df_num_agg[feature + '_last_sub_std'] = df_num_agg[feature] - df_num_agg[feature.replace('last', 'std')]
            df_num_agg[feature + '_last_div_std'] = df_num_agg[feature] / df_num_agg[feature.replace('last', 'std')]

        # Check if last and median are in the feature name and compute difference and division
        elif 'last' in feature and feature.replace('last', 'median') in df_num_agg:
            df_num_agg[feature + '_last_sub_median'] = df_num_agg[feature] - df_num_agg[feature.replace('last', 'median')]
            df_num_agg[feature + '_last_div_median'] = df_num_agg[feature] / df_num_agg[feature.replace('last', 'median')]

        # Check if last and min are in the feature name and compute difference and division
        elif 'last' in feature and feature.replace('last', 'min') in df_num_agg:
            df_num_agg[feature + '_last_sub_min'] = df_num_agg[feature] - df_num_agg[feature.replace('last', 'min')]
            df_num_agg[feature + '_last_div_min'] = df_num_agg[feature] / df_num_agg[feature.replace('last', 'min')]

        # Check if last and max are in the feature name and compute difference and division
        elif 'last' in feature and feature.replace('last', 'max') in df_num_agg:
            df_num_agg[feature + '_last_sub_max'] = df_num_agg[feature] - df_num_agg[feature.replace('last', 'max')]
            df_num_agg[feature + '_last_div_max'] = df_num_agg[feature] / df_num_agg[feature.replace('last', 'max')]

    # Concat the aggregated features
    df = pd.concat([df_cat_agg, df_num_agg], axis = 1)

    return df


# In[10]: Feature engineering functions VIII: Combined dataset

# Function that creates a combined dataset with all the features created
def feat_combined(data, groupby_var = 'customer_ID'):

    # Call the feature_types function to get the feature types
    not_used, cat_features, bin_features_1, bin_features_2, bin_features_3, num_features = feature_types(data)
    print('Feature types done')

    # Call the feat_diff function to get the difference features
    df_diff = feat_diff(data, num_features, 1)
    print('Difference features done')

    # Call the feat_lag function to get the lagged features
    df_lag = feat_lag(data, [1, 2, 3, 6, 11])
    print('Lagged features done')

    # Call the feat_period_means function to get the period means features
    df_means = feat_period_means(data, num_features)
    print('Period means features done')

    # Call the feat_last_diffdiv function to get the last difference and division features
    df_last_diffdiv = feat_last_diffdiv(data, cat_features, num_features, groupby_var)
    print('Last difference and division features done')

    return df_diff, df_lag, df_means, df_last_diffdiv

    # CREO QUE AQUÍ ME CREA UNA TUPLA Y NO SÉ POR QUÉ

    # # Merge on customer_ID (index)
    # df_merged = df_last_diffdiv.merge(df_diff, how='inner', on='customer_ID').merge(df_lag, how='inner', on='customer_ID').merge(df_means, how='inner', on='customer_ID')
    # df_merged = df_merged.reset_index(drop=False)
    # print('Merge done')

    # del not_used, bin_features_1, bin_features_2, bin_features_3, df_diff, df_lag, df_means, df_last_diffdiv

    # return df_merged, cat_features, num_features


# In[11]: Save the combined dataset in a parquet file

def save_combined(data, dataset_name: str): # dataset_name = 'train' or 'test'
    # Create a directory to store the output files
    results_path = Path('./DATASETS')
    results_path.mkdir(exist_ok=True)

    # Name experiment
    experiment_name ='combined_dataset'
    experiment_dir = results_path / experiment_name
    experiment_dir.mkdir(exist_ok=True)

    print(f'Saving combined dataset in {experiment_dir}')

    # Save the combined dataset in a parquet file
    if dataset_name == 'test':
        data.to_parquet(experiment_dir / 'test_combined_dataset.parquet')
    elif dataset_name == 'train':
        data.to_parquet(experiment_dir / 'train_combined_dataset.parquet')
    print('Combined dataset saved successfully')

# %%
# In[6]: data
# train = pd.read_parquet('C:/Users/Jose/Documents/UNIVERSIDAD/TFG/amex-default-prediction/parquet_ds_integer_dtypes/train.parquet')
test_data = pd.read_parquet('C:/Users/Jose/Documents/UNIVERSIDAD/TFG/amex-default-prediction/parquet_ds_integer_dtypes/test.parquet')

# In[7]: tests

# not_used, cat_features, bin_features_1, bin_features_2, bin_features_3, num_features = feature_types(train)

# train_oh, test_oh, list_dummies_train, list_dummies_test = dummy_encoding(train, test_data, cat_features)

# %%

# train_agg = feat_aggregations(train_oh, list_dummies_train, num_features, 'customer_ID')
# train_agg
# %%


# Code for retrieving a list with the customers that have only one observation (only one record in S_2)
# train['customer_ID'].value_counts()[train['customer_ID'].value_counts() == 1].index.tolist()
# b8d4227c0d88180958483c0db718554819048f7af142edb30d84068a7fb5ee17



# Store the means_df DataFrame in a parquet file and save it in the current directory
# means_df = feat_period_means_vectorized(train, num_features)
# means_df.to_parquet('means_df.parquet.gzip', compression='gzip')
# %%

# Code for computing the mean of the last 6 observations for all customers for the numerical features
# filling with NaNs the feature's values if the customer has less than 6 observations
# mean_test = train.groupby('customer_ID')[num_features].apply(lambda x: x.iloc[-6:].mean())

# Convert to NaN the mean values of the customers in mean_test with less than 6 observations in train dataframe
# mean_test_nan = mean_test.where(train['customer_ID'].value_counts() >= 6, np.nan)



# mean_test2 = train.groupby('customer_ID')[num_features].apply(lambda x: x.iloc[-6:].mean() if len(x) >= 6 else pd.Series([np.nan]))

# mean_test2

# Drop all lag variables
# train.drop(columns = [col for col in train.columns if 'lag' in col], inplace = True)
