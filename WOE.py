# AMEX default prediction Weight of Evidence (WOE)

# In[1]: Librerías

# Data manipulation
import pandas as pd 
import numpy as np
import gc

# Data visualization
import matplotlib.pyplot as plt

# Time management
import time

# Machine learning
from sklearn.model_selection import StratifiedKFold 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Feature engineering functions
import feature_engineering as fe

# Import optibinning
import optbinning

# In[2]: Lectura de datos

train_labels, train, test = fe.load_datasets(oh=False)


# In[3]: Variables categóricas

# cat_features = ['B_30', 'B_38', 'D_63', 'D_64', 'D_66', 'D_68', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126']

# Como se han agrupado los datos, las categóricas tendrán el mismo nombre que las variables de la lista anterior que contengan seguidos de "_last" o "_first"
# Por ejemplo, B_30_last, B_30_first, B_38_last, B_38_first, etc.

# Lista de variables categóricas
cat_features = ['B_30_last', 'B_30_first', 'B_38_last', 'B_38_first', 'D_63_last', 'D_63_first', 'D_64_last', 'D_64_first', 'D_66_last', 'D_66_first', 'D_68_last', 
                'D_68_first', 'D_114_last', 'D_114_first', 'D_116_last', 'D_116_first', 'D_117_last', 'D_117_first', 'D_120_last', 'D_120_first', 'D_126_last', 'D_126_first']

num_features = [col for col in train.columns if col not in cat_features and col != 'customer_ID']

# Lista de variables (exlucyendo 'customer_ID)
features = list(train.columns)
features.remove('customer_ID')
# features.remove('S_2')


# In[4]: Métrica

def amex_metric(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:

    def top_four_percent_captured(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        df = (pd.concat([y_true, y_pred], axis='columns')
              .sort_values('prediction', ascending=False))
        df['weight'] = df['target'].apply(lambda x: 20 if x==0 else 1)
        four_pct_cutoff = int(0.04 * df['weight'].sum())
        df['weight_cumsum'] = df['weight'].cumsum()
        df_cutoff = df.loc[df['weight_cumsum'] <= four_pct_cutoff]
        return (df_cutoff['target'] == 1).sum() / (df['target'] == 1).sum()
        
    def weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        df = (pd.concat([y_true, y_pred], axis='columns')
              .sort_values('prediction', ascending=False))
        df['weight'] = df['target'].apply(lambda x: 20 if x==0 else 1)
        df['random'] = (df['weight'] / df['weight'].sum()).cumsum()
        total_pos = (df['target'] * df['weight']).sum()
        df['cum_pos_found'] = (df['target'] * df['weight']).cumsum()
        df['lorentz'] = df['cum_pos_found'] / total_pos
        df['gini'] = (df['lorentz'] - df['random']) * df['weight']
        return df['gini'].sum()

    def normalized_weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        y_true_pred = y_true.rename(columns={'target': 'prediction'})
        return weighted_gini(y_true, y_pred) / weighted_gini(y_true, y_true_pred)

    g = normalized_weighted_gini(y_true, y_pred)
    d = top_four_percent_captured(y_true, y_pred)

    return 0.5 * (g + d)


# In[5]: Merge de train y train_labels

train_df = train.merge(train_labels, left_on='customer_ID', right_on='customer_ID')
del train, train_labels

# In[6]: Cálculo del WOE

# Función para calcular el WOE de una variable categórica
def woe_cat(df, cat_var, target):
    '''
    Calcula el WOE de una variable categórica
    '''
    # Tabla de frecuencias
    freq_table = pd.crosstab(df[cat_var], df[target], margins=True)
    freq_table = freq_table.reset_index()
    freq_table = freq_table.rename(columns={0: 'non_default', 1: 'default'})
    freq_table['total'] = freq_table['non_default'] + freq_table['default']
    freq_table['non_default_pct'] = freq_table['non_default'] / freq_table['total']
    freq_table['default_pct'] = freq_table['default'] / freq_table['total']
    freq_table['woe'] = np.log(freq_table['non_default_pct'] / freq_table['default_pct'])
    freq_table['iv'] = (freq_table['non_default_pct'] - freq_table['default_pct']) * freq_table['woe']
    freq_table['iv'] = freq_table['iv'].sum()
    freq_table['variable'] = cat_var
    freq_table = freq_table[['variable', cat_var, 'non_default', 'default', 'total', 'non_default_pct', 'default_pct', 'woe', 'iv']]
    return freq_table

# Función para calcular el WOE de una variable numérica
def woe_num(df, num_var, target):
    '''
    Calcula el WOE de una variable numérica
    '''
    # Tabla de frecuencias
    freq_table = pd.crosstab(df[num_var], df[target], margins=True)
    freq_table = freq_table.reset_index()
    freq_table = freq_table.rename(columns={0: 'non_default', 1: 'default'})
    freq_table['total'] = freq_table['non_default'] + freq_table['default']
    freq_table['non_default_pct'] = freq_table['non_default'] / freq_table['total']
    freq_table['default_pct'] = freq_table['default'] / freq_table['total']
    freq_table['woe'] = np.log(freq_table['non_default_pct'] / freq_table['default_pct'])
    freq_table['iv'] = (freq_table['non_default_pct'] - freq_table['default_pct']) * freq_table['woe']
    freq_table['iv'] = freq_table['iv'].sum()
    freq_table['variable'] = num_var
    freq_table = freq_table[['variable', num_var, 'non_default', 'default', 'total', 'non_default_pct', 'default_pct', 'woe', 'iv']]
    return freq_table


# In[7]: Cálculo del WOE de las variables

# Cálculo del WOE de las variables categóricas
woe_cat_df = pd.DataFrame()
for cat_var in cat_features:
    woe_cat_df = woe_cat_df.append(woe_cat(train_df, cat_var, 'target'))

# Cálculo del WOE de las variables numéricas
woe_num_df = pd.DataFrame()
for num_var in num_features:
    woe_num_df = woe_num_df.append(woe_num(train_df, num_var, 'target'))




# %%
