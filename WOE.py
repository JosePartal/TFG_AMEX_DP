# AMEX default prediction Weight of Evidence (WOE)

# In[1]: Librerías

# Data manipulation
# downgrade pandas to version 1.3.5
# pip install pandas==1.3.5
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

# Binning and WOE libraries
import optbinning
import monotonic_binning
from monotonic_binning import monotonic_woe_binning as mwb
import xverse
from xverse.transformer import MonotonicBinning
from xverse.transformer import WOE

# Progress bar
from tqdm import tqdm

# Error management
import traceback

# In[2]: Lectura de datos

train_labels, train, test = fe.load_datasets(oh=False)

# If oh=False
train = train.replace([np.inf, -np.inf], 0)

# Fill train NaN with np.nan
train = train.fillna(np.nan)


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

def amex_metric_np(preds: np.ndarray, target: np.ndarray) -> float:
    indices = np.argsort(preds)[::-1]
    preds, target = preds[indices], target[indices]

    weight = 20.0 - target * 19.0
    cum_norm_weight = (weight / weight.sum()).cumsum()
    four_pct_mask = cum_norm_weight <= 0.04
    d = np.sum(target[four_pct_mask]) / np.sum(target)

    weighted_target = target * weight
    lorentz = (weighted_target / weighted_target.sum()).cumsum()
    gini = ((lorentz - cum_norm_weight) * weight).sum()

    n_pos = np.sum(target)
    n_neg = target.shape[0] - n_pos
    gini_max = 10 * n_neg * (n_pos + 20 * n_neg - 19) / (n_pos + 20 * n_neg)

    g = gini / gini_max
    return 0.5 * (g + d)


# In[5]: Merge de train y train_labels

train_df = train.merge(train_labels, left_on='customer_ID', right_on='customer_ID')
del train, train_labels, test


# # Convertir las variables float32 a float64 para que funcione el binning

# for col in train_df.columns:
#     if train_df[col].dtype == 'float32':
#         train_df[col] = train_df[col].astype('float64')


# In[7]: Regresión logística usando variables Weight of Evidence (WOE)

# Vamos a hacer el binning y calcular las variables WOE usando la librería xverse

# In[8]: Binning

# # Creamos el objeto para el binning
# binning_fit = MonotonicBinning()

# # Entrenamos el objeto con los datos de entrenamientoç
# binning_fit.fit(train_df[features], train_df['target'])

# # Hacemos el binning de los datos de entrenamiento
# train_df_binned = binning_fit.transform(train_df[features])
# train_df_binned.head()

# # In[9]: WOE

# # Creamos el objeto para el WOE
# woe_fit = WOE()

# # Entrenamos el objeto con los datos de entrenamiento
# woe_fit.fit(train_df_binned, train_df['target'])

# # Hacemos el WOE de los datos de entrenamiento
# train_df_woe = woe_fit.transform(train_df_binned)
# train_df_woe.head()

# #downgrade pandas to version 1.3.5
# #pip install pandas==1.3.5



# In[10]: Binning

# Vamos a usar la librería optbinning para hacer el binning y el WOE
# Vamos a usar la función BinningProcess

# Creamos el objeto para el binning con special_codes identificando los NaN (special_codes must be a dit, list or numpy.ndarray)
binning_fit = optbinning.BinningProcess(variable_names=features,  categorical_variables= cat_features, special_codes=[np.nan], n_jobs=-1)

# Entrenamos el objeto con los datos de entrenamiento
binning_fit.fit(train_df[features], train_df['target'].to_numpy(), check_input=True)

# Hacemos el binning de los datos de entrenamiento
train_df_binned = binning_fit.transform(train_df[features])
train_df_binned.head()

# In[11]: Regresión logística usando variables Weight of Evidence (WOE)

# Calcular las variables WOE
# woe_values = binning_fit.transform(train_df[features], metric="woe")

# Event rate
# event_rate = binning_fit.transform(train_df[features], metric="event_rate")

# # Transformamos los índices
# transform_indices = binning_fit.transform(train_df[features], metric="indices")

# # Transformamos los bins
# transform_bins = binning_fit.transform(train_df[features], metric="bins")


# In[12]: WOE

# La tabla tran_df_binned contiene los valores WoE para cada bin de cada variable. 
# Vamos a renombrar las variables añadiendo "_woe" al final para diferenciarlas de las variables originales

# Renombramos las variables
train_df_binned.columns = [col + '_woe' for col in train_df_binned.columns]

# Añadimos la variable target
train_df_binned['target'] = train_df['target']


# In[13]: Regresión logística usando variables Weight of Evidence (WOE) --> HACER STEPWISE

"""Vamos a hacer la regresión logística usando las variables WOE.""" 

# En primer lugar, definimos X e y
X = train_df_binned.drop('target', axis=1)
y = train_df_binned['target']

# Separamos los datos en entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Definimos el modelo
model = LogisticRegression(random_state=42, max_iter=1000)

# Entrenamos el modelo
model.fit(X_train, y_train)

# Predecimos los datos de validación
y_pred = model.predict_proba(X_val)[:, 1]

# pred_train = 1/(1+np.exp(-model.predict(X_train)))
# pred_train_df = pd.DataFrame(pred_train, columns=['prediction'])

# y_val_df = pd.DataFrame(y_val, columns=['target'])

# Calculamos la métrica
metric = amex_metric_np(y_pred, y_val.to_numpy())

print(f'AMEX metric: {metric}')


# In[14]: Regresión logística usando variables Weight of Evidence (WOE) II

# Vamos a ver el resultado de la regresión logística, comprobando los coeficientes de cada variable

# Creamos un dataframe con los coeficientes
# coef_df = pd.DataFrame({'feature': X_train.columns, 'coef': model.coef_[0]})
# coef_df


# In[15]: Regresión logística usando variables Weight of Evidence (WOE) III: Test predictions


# %%
