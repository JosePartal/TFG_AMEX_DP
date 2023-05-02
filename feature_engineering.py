# Functions for feature engineering

# In[1]: Libaries

# Data manipulation
import pandas as pd 
import numpy as np

# Store and organize output files
from pathlib import Path


# In[2]: Feature engineering functions I: Feature types

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


# In[3]: Feature engineering functions II: Dummy encoding

"""Before creating new variables, we need to encode the categorical features.
If we dummy encode them we will have k-1 new features for each categorical feature instead of k,
so there will be less features in the final dataset and the dimensionality will be reduced.
"""

def dummy_encoding(train_df, test_df, cat_features):
    # Dummy encoding
    train_df_oh = pd.get_dummies(train_df, columns = cat_features, drop_first = True)
    test_df_oh = pd.get_dummies(test_df, columns = cat_features, drop_first = True)

    # List of the new dummy encoded features
    dummies_train = list(set(train_df_oh.columns) - set(train_df.columns))
    dummies_test = list(set(test_df_oh.columns) - set(test_df.columns))


    return train_df_oh, test_df_oh, dummies_train, dummies_test



# In[4]: Feature engineering functions III: Aggregations

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

    # Merge the aggregated features with the original dataset
    df = df.merge(df_cat_agg, on = groupby_var, how = 'inner')
    df = df.merge(df_num_agg, on = groupby_var, how = 'inner')

    del df_cat_agg, df_num_agg
    return df
    
# In[5]: data
train = pd.read_parquet('C:/Users/Jose/Documents/UNIVERSIDAD/TFG/amex-default-prediction/parquet_ds_integer_dtypes/train.parquet')
test_data = pd.read_parquet('C:/Users/Jose/Documents/UNIVERSIDAD/TFG/amex-default-prediction/parquet_ds_integer_dtypes/test.parquet')

# In[6]: tests

not_used, cat_features, bin_features_1, bin_features_2, bin_features_3, num_features = feature_types(train)

train_oh, test_oh, list_dummies_train, list_dummies_test = dummy_encoding(train, test_data, cat_features)

# %%

train_agg = feat_aggregations(train_oh, list_dummies_train, num_features, 'customer_ID')
train_agg
# %%
