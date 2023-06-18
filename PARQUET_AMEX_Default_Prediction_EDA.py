#!/usr/bin/env python
# coding: utf-8


# In[1]: Import libraries

import pandas as pd 
import numpy as np

import matplotlib as mpl  
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
# import plotly.graph_objects as go
from plotly.subplots import make_subplots

import imblearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold 
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# In[2]: Read data

# Train
train = pd.read_parquet('C:/Users/Jose/Documents/UNIVERSIDAD/TFG/amex-default-prediction/parquet_ds_integer_dtypes/train.parquet')

# Labels
train_labels = pd.read_csv('C:/Users/Jose/Documents/UNIVERSIDAD/TFG/amex-default-prediction/train_labels.csv', low_memory=False)

# Train + Labels
train_raw = train.merge(train_labels, left_on='customer_ID', right_on='customer_ID')
# train_raw = train_raw.drop(columns = ['customer_ID']), 'S_2'])

# Clear memory: train_labels
# del train_labels
# del train

# Test
test_data = pd.read_parquet('C:/Users/Jose/Documents/UNIVERSIDAD/TFG/amex-default-prediction/parquet_ds_integer_dtypes/test.parquet')
# test_data = test_data.drop(columns = ['customer_ID', 'S_2'])


# In[3]: Data overview

"""
The dataset contains aggregated profile features for each customer at each statement date. Features are anonymized and normalized, 
and fall into the following general categories:

D_* = Delinquency variables
S_* = Spend variables
P_* = Payment variables
B_* = Balance variables
R_* = Risk variables

with the following features being categorical:
    
    ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']

and 'S_2' being the date variable.
"""

# Categorical features
categorical_features = ['B_30', 'B_38', 'D_63', 'D_64', 'D_66', 'D_68', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126']

# Data shape (train)
print("We have %d records and %d features in train dataset." % (train.shape[0], train.shape[1]))

# Data shape (test)
print("We have %d records and %d features in test dataset." % (test_data.shape[0], test_data.shape[1]))

# No. of unique customers in train and test datasets
print(f'Those {train.shape[0]} records do not belong to a single client each, but there are multiple obsevations \n \
       for each client, one for each transaction date. In particular, we have {len(train["customer_ID"].unique())} \n \
       unique clients in train dataset and {test_data["customer_ID"].nunique()} clients in test dataset.')

# Date range for train and test datasets
print(f'The date range for the train dataset is from {train["S_2"].min()} to {train["S_2"].max()} \n \
         and for the test dataset is from {test_data["S_2"].min()} to {test_data["S_2"].max()}. \n \
         This means that the dates of train and test do not overlap.')


# In[4]: Exploratory data analysis (EDA)

# One of the first things we should do is have a look at the variables and their types.

train.info(max_cols = 200, show_counts=True)

"""
- We have 2 variables of type object (date - S_2 - and customer_id).

- There are many features with missing values (NaNs). Dropping them would result in a loss of information, 
so we will have to deal with them later. There are many decission-tree based models that can handle missing values,
so we won't have to worry about them too much for now.

- If we use regression models, we will have to deal with them.
"""


# In[5]: Exploratory data analysis (EDA) - Target

# Distribution of target variable 

target = train_raw.target.value_counts(normalize=True)
target.rename(index={1:'Default',0:'Paid'},inplace=True)
target

print(f'The target variable is highly imbalanced, with {round(target[0]*100,2)}% of the observations \n \
      being of clients that paid their credit card bill and {round(target[1]*100,2)}% from those who default.')

print('Furthermore, we are given that: "The good customers have been subsampled by a factor of 20; \n \
      this means that in reality there are 6.8 million good customers. 98 % of the customers are good; 2 % are bad"')

px.pie(target.index, values = target, names = target.index,  title='Target distribution') 


# In[5]: Exploratory data analysis (EDA) - Target distribution by date (2)
# Distribution of target variable by date

target_date = train_raw.groupby(['S_2'])['target'].value_counts(normalize=False)
target_date.rename(index={1:'Default',0:'Paid'},inplace=True)
target_date = target_date.reset_index(name='Count')
target_date

# Plot grouping by month

fig = px.bar(target_date, x="S_2", y="Count", color='target', barmode='group', title='Target distribution by date')
fig.show()

# We can also see that the monthly amount of default is more or less constant.

# In[7]: Exploratory data analysis (EDA) - Daily statements per customer in train dataset

# Line graph of number of statements issued daily

statements_per_customer = train_raw.groupby(['S_2'])['customer_ID'].nunique()
statements_per_customer = statements_per_customer.reset_index(name='count')

print('We can see that there is a weekly pattern in the number of statements issued. \n \
       Saturdays have the highest number of statements issued.')

px.line(statements_per_customer, x="S_2", y="count", title='Number of statements issued daily (Train)', 
        labels={'count':'Number of statements', 'S_2':'Statement Date'})


# In[6]: Exploratory data analysis (EDA) - Statements per customer in train dataset

# Pie chart of statements per customer

statements_per_customer = train_raw.groupby(['customer_ID'])['S_2'].nunique()
statements_per_customer = statements_per_customer.value_counts(normalize=False)
statements_per_customer = statements_per_customer.reset_index(name='count')

print(f'We see that 84.1% of customers  have 13 statements and the remaining 16% between \n \
      1 and 12 statements. It is worth noting if this is because of late entry of customers \n \
      in the dataset, specially for 12 months. \n \
      Furthermore, this may indicate that the customers get one credit card statement per month.')

px.pie(statements_per_customer, values = 'count', names = 'index', title='Statements per customer')

# Most statements occur in Saturday. See: https://www.kaggle.com/code/schopenhacker75/fancy-complete-eda
# Customers are issued monthly. Are these statements payments or just the monthly reports?


# In[8]: Exploratory data analysis (EDA) - Presence of customers in train dataset by target (1)

# Bar chart of number of months each customer has been present in the dataset

print('We compute the number of unique "customer_ID" and "target" combinations and then group by "target" \n \
      to get the percentage of customers that have been present for each month. \n \
      We can see that 86% of the customers have been present in the dataset for 13 months.')

presence_of_customers = train_raw.groupby(['customer_ID','target']).size().reset_index().rename(columns={0:'Presence'})

fig, ax = plt.subplots(1,1, figsize=(15,5))
sns.histplot(x='Presence', data=presence_of_customers, hue='target', stat='percent', multiple="dodge", bins=np.arange(0,14), ax=ax)
ax.bar_label(ax.containers[0], fmt='%.f%%')
ax.bar_label(ax.containers[1], fmt='%.f%%')
plt.show()


# In[9]: Exploratory data analysis (EDA) - Presence of customers in train dataset by target (2)

print('Lets zoom into the customers that have been present for less than 13 months. \n \
      We can see that people that are less than 13 months in the dataset are more likely to default. \n \
      However, we have to be careful with this here we also have late entry customers. \n \
      So we have customers that entered late and customers that dropped out early \n \
      or made not transactions anymore (see discussion in comments).') 

# https://www.kaggle.com/competitions/amex-default-prediction/discussion/327597

fig, ax = plt.subplots(1,1, figsize=(15,5))
sns.histplot(x='Presence', data=presence_of_customers, hue='target', stat='percent', multiple="dodge", bins=np.arange(0,14), ax=ax)
ax.bar_label(ax.containers[0], fmt='%.2f%%')
ax.bar_label(ax.containers[1], fmt='%.2f%%')
ax.set_xlim(0,12)
ax.set_ylim(0,1)
plt.show()


# In[10]: Exploratory data analysis (EDA) - Missing values

# We are going to explore the amount and percentage of missing values in each variable

pd_series_null_columns = train_raw.isnull().sum().sort_values(ascending=False)
# pd_series_null_rows = train_raw.isnull().sum(axis=1).sort_values(ascending=False)


pd_null_columnas = pd.DataFrame(pd_series_null_columns, columns=['nulos_columnas'])     
pd_null_columnas['porcentaje_columnas'] = pd_null_columnas['nulos_columnas']/train_raw.shape[0]
# pd_null_filas = pd.DataFrame(pd_series_null_rows, columns=['nulos_filas'])  
# pd_null_filas['target'] = train_raw['target'].copy()
# pd_null_filas['porcentaje_filas']= pd_null_filas['nulos_filas']/train_raw.shape[1]

#  Vector of features with null values

list_var_null_train = [x for x in list(pd_null_columnas.index) if pd_null_columnas.nulos_columnas[x] > 0]

tmp = train_raw.isna().sum().div(len(train_raw)).mul(100).sort_values(ascending=False)

print(f'There are {len(tmp[tmp > 0])} variables with missing values. \n \
      Some of these variables have a high percentage of missing values. \n \
      It is worth studying if this varibles also have such a high percentage of missing values \n \
      in the test set before deciding to drop them. \n \
      Furthermore, we can also study the structure of this missing data.')

plt.style.use('Solarize_Light2')
fig, ax = plt.subplots(2,1, figsize=(25,10))
sns.barplot(x=tmp[:100].index, y=tmp[:100].values, ax=ax[0])
sns.barplot(x=tmp[100:].index, y=tmp[100:].values, ax=ax[1])
ax[0].set_ylabel("Percentage [%]"), ax[1].set_ylabel("Percentage [%]")
ax[0].tick_params(axis='x', rotation=90); ax[1].tick_params(axis='x', rotation=90)
plt.suptitle("Amount of missing data")
plt.tight_layout()
plt.show()

# del tmp, fig, ax, pd_series_null_columns


# In[11]: Exploratory data analysis (EDA) - Missing values (2)

# List of variables with missing values in train dataset

# Missing values in test dataset

pd_series_null_columns_test = test_data.isnull().sum().sort_values(ascending=False)

pd_null_columnas_test = pd.DataFrame(pd_series_null_columns_test, columns=['nulos_columnas'])
pd_null_columnas_test['porcentaje_columnas'] = pd_null_columnas_test['nulos_columnas']/test_data.shape[0]

list_var_null_test = [x for x in list(pd_null_columnas_test.index) if pd_null_columnas_test.nulos_columnas[x] > 0]

tmp = test_data.isna().sum().div(len(test_data)).mul(100).sort_values(ascending=False)

plt.style.use('Solarize_Light2')
fig, ax = plt.subplots(2,1, figsize=(25,10))
sns.barplot(x=tmp[:100].index, y=tmp[:100].values, ax=ax[0])
sns.barplot(x=tmp[100:].index, y=tmp[100:].values, ax=ax[1])
ax[0].set_ylabel("Percentage [%]"), ax[1].set_ylabel("Percentage [%]")
ax[0].tick_params(axis='x', rotation=90); ax[1].tick_params(axis='x', rotation=90)
plt.suptitle("Amount of missing data")
plt.tight_layout()
plt.show()

del tmp, fig, ax, pd_series_null_columns, pd_null_columnas, pd_series_null_columns_test, pd_null_columnas_test



# In[12]: Exploratory data analysis (EDA) - Missing values (3)

"""
We have seen that the same variables have missing values in both the train and test datasets and in similar
proportions. Missing data in this dataset is "problematic" if we want to use simpler ML models that do not
handle missing data (eg. logistic regression).

However, in this post here https://www.kaggle.com/code/raddar/understanding-na-values-in-amex-competition/notebook
an analysis of the missing data is done and it is concluded that we have two types of missing data:

1. Systemic NA: missing data in the train and test datasets are missing for a reason. This missing data contains
information and we can't inpute them. For example, in the post before, the author discovers that
there is a good amount of missings that appear only on the first observation for each customer.
This probably represents fresh credit card accounts that haven't been used yet and with probably zero balance.

2. Random NA: missing data in the train and test datasets are missing at random. We may be able to impute
these missing values, although we are not really sure what amount of this data is random or systemic.
We should compare the models before and after inputing and if they improve, we can use the inputed data.

"""


# In[13]: Exploratory data analysis (EDA) - Variables types

# Categorical features
categorical_features = ['B_30', 'B_38', 'D_63', 'D_64', 'D_66', 'D_68', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126']
train_raw[categorical_features] = train_raw[categorical_features].astype("category")
test_data[categorical_features] = test_data[categorical_features].astype("category")

# Let's check for missing data in the categorical features
print(f'{train_raw[categorical_features].isna().sum().div(len(train_raw)).sort_values(ascending=False)}')

# Numerical features
features = train.drop(['customer_ID', 'S_2'], axis = 1).columns.to_list()
numerical_features = [col for col in features if col not in categorical_features]


# In[14]: Exploratory data analysis (EDA) - Feature distributions (1): Categorical features

# Let's plot the distribution of the categorical features making barplots and analysing the distribution 
# of the target variable in each category

print('There are at most 8 categories. One-hot encoder is feasible.')

plt.figure(figsize=(16, 16))
for i, f in tqdm(enumerate(categorical_features)):
    plt.subplot(4, 3, i+1)
    temp = pd.DataFrame(train_raw[f][train_raw.target == 0].value_counts(dropna=False, normalize=True).sort_index().rename('count'))
    temp.index.name = 'value'
    temp.reset_index(inplace=True)
    plt.bar(temp.index, temp['count'], alpha=0.5, label='Pagado')
    temp = pd.DataFrame(train_raw[f][train_raw.target == 1].value_counts(dropna=False, normalize=True).sort_index().rename('count'))
    temp.index.name = 'value'
    temp.reset_index(inplace=True)
    plt.bar(temp.index, temp['count'], alpha=0.5, label='Default')
    plt.xlabel(f)
    plt.ylabel('Frecuencia')
    plt.legend()
    plt.xticks(temp.index, temp.value)
plt.suptitle('Variables categóricas', fontsize=20, y=0.93)
plt.show() #savefig
del temp


# In[15]: Exploratory data analysis (EDA) - Feature distributions (2): Spend features

# Let's plot the distribution of the Spend features, making histograms and analysing the distribution
# of the target variable of each feature

# Spend features
spend_features = [col for col in features if col.startswith('S_') and col not in categorical_features] 

plt.figure(figsize=(16, 16))
for i, f in tqdm(enumerate(spend_features)):
    ax = plt.subplot(5, 5, i+1)
    sns.kdeplot(train_raw[f][train_raw.target == 0], label='Pagado')
    sns.kdeplot(train_raw[f][train_raw.target == 1], label='Default')
    plt.xlabel(f)
    plt.ylabel('Densidad')

# Crear una leyenda única en la esquina superior izquierda
handles, labels = ax.get_legend_handles_labels()
plt.figlegend(handles, labels, loc='upper left', bbox_to_anchor=(0.08, 0.97), bbox_transform=plt.gcf().transFigure)

plt.suptitle('Variables de gasto (S)', fontsize=20, y=0.965)
plt.tight_layout(rect=[0, 0.03, 0.98, 0.95])
plt.show() # savefig



# In[16]: Exploratory data analysis (EDA) - Feature distributions (3): Delincuquency features

# Delinquency features
delinquency_features = [col for col in features if col.startswith('D_') and col not in categorical_features]

plt.figure(figsize=(16, 54))
for i, f in tqdm(enumerate(delinquency_features)):
      plt.subplot(18, 5, i+1)
      sns.kdeplot(train_raw[f][train_raw.target == 0], label='Pagado')
      sns.kdeplot(train_raw[f][train_raw.target == 1], label='Default')
      plt.xlabel(f)
      plt.ylabel('Densidad')

# Crear una leyenda única en la esquina superior izquierda
handles, labels = ax.get_legend_handles_labels()
plt.figlegend(handles, labels, loc='upper left', bbox_to_anchor=(0.08, 0.97), bbox_transform=plt.gcf().transFigure)

plt.suptitle('Variables de delincuencia (D)', fontsize=20, y=0.965)
plt.tight_layout(rect=[0, 0.03, 0.98, 0.95])
plt.show() #savefig



# In[17]: Exploratory data analysis (EDA) - Feature distributions (4): Payment features

# Payment features
payment_features = [col for col in features if col.startswith('P_') and col not in categorical_features]

plt.figure(figsize=(16, 5))
for i, f in tqdm(enumerate(payment_features)):
      plt.subplot(1, 3, i+1)
      sns.kdeplot(train_raw[f][train_raw.target == 0], label='Pagado')
      sns.kdeplot(train_raw[f][train_raw.target == 1], label='Default')
      plt.xlabel(f)
      plt.ylabel('Densidad')

# Crear una leyenda única en la esquina superior izquierda
handles, labels = ax.get_legend_handles_labels()
plt.figlegend(handles, labels, loc='upper left', bbox_to_anchor=(0.08, 0.97), bbox_transform=plt.gcf().transFigure)

plt.suptitle('Variables de pago (P)', fontsize=20, y=0.965)
plt.tight_layout(rect=[0, 0.03, 0.98, 0.95])
plt.show() #savefig


# In[18]: Exploratory data analysis (EDA) - Feature distributions (5): Balance features

# Balance features
balance_features = [col for col in features if col.startswith('B_') and col not in categorical_features]

plt.figure(figsize=(16, 16))
for i, f in tqdm(enumerate(balance_features)):
      plt.subplot(8, 5, i+1)
      sns.kdeplot(train_raw[f][train_raw.target == 0], label='Pagado')
      sns.kdeplot(train_raw[f][train_raw.target == 1], label='Default')
      plt.xlabel(f)
      plt.ylabel('Densidad')

# Crear una leyenda única en la esquina superior izquierda
handles, labels = ax.get_legend_handles_labels()
plt.figlegend(handles, labels, loc='upper left', bbox_to_anchor=(0.08, 0.97), bbox_transform=plt.gcf().transFigure)

plt.suptitle('Variables de balance (B)', fontsize=20, y=0.965)
plt.tight_layout(rect=[0, 0.03, 0.98, 0.95])
plt.show() #savefig


# In[19]: Exploratory data analysis (EDA) - Feature distributions (6): Risk features

# Risk features
risk_features = [col for col in features if col.startswith('R_') and col not in categorical_features]

plt.figure(figsize=(16, 16))
for i, f in tqdm(enumerate(risk_features)):
      plt.subplot(6, 5, i+1)
      sns.kdeplot(train_raw[f][train_raw.target == 0], label='Pagado')
      sns.kdeplot(train_raw[f][train_raw.target == 1], label='Default')
      plt.xlabel(f)
      plt.ylabel('Densidad')

# Crear una leyenda única en la esquina superior izquierda
handles, labels = ax.get_legend_handles_labels()
plt.figlegend(handles, labels, loc='upper left', bbox_to_anchor=(0.08, 0.97), bbox_transform=plt.gcf().transFigure)

plt.suptitle('Variables de riesgo (R)', fontsize=20, y=0.965)
plt.tight_layout(rect=[0, 0.03, 0.98, 0.95])
plt.show() #savefig


# In[20]: Exploratory data analysis (EDA) - Correlations (1): With target variable

# Correlation with target variable using heatmap
plt.figure(figsize=(16, 16))
sns.heatmap(train[numerical_features + ['target']].corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation with target variable', fontsize=20)
plt.show() #savefig



# In[21]: Exploratory data analysis (EDA) - Correlations (2): Among Delinquency features


# In[22]: Exploratory data analysis (EDA) - Correlations (3): Among Spend features


# In[23]: Exploratory data analysis (EDA) - Correlations (4): Among Payment features


# In[24]: Exploratory data analysis (EDA) - Correlations (5): Among Balance features


# In[25]: Exploratory data analysis (EDA) - Correlations (6): Among Risk features


# In[]: Binary feaatures

# Code for identifying binary features (only takes value 0 or 1)
# binary_features = [col for col in features if train[col].nunique() == 2]
# print(f'{len(binary_features)} binary features: {binary_features}\n')
