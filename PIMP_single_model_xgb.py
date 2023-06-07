# AMEX default prediction XGBoost (agg) - PIMP

# In[1]: Librerías

# Manipulación de datos
import pandas as pd 
import numpy as np
import gc

# Visualización
import matplotlib.pyplot as plt
import seaborn as sns

# Tiempo
import time

# Machine learning
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold 

# Funciones ingeniería de variables
import feature_engineering as fe

# Librería para monitorizar bucles
from tqdm import tqdm


# In[2]: Lectura de datos
oh = True

train_labels, train, test = fe.load_datasets(oh)

# In[3]: Variables categóricas

# cat_features = ['B_30', 'B_38', 'D_63', 'D_64', 'D_66', 'D_68', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126']

# Como se han agrupado los datos, las categóricas tendrán el mismo nombre que las variables de la lista anterior que contengan seguidos de "_last" o "_first"
# Por ejemplo, B_30_last, B_30_first, B_38_last, B_38_first, etc.

# Lista de variables categóricas
cat_features = ['B_30_last', 'B_30_first', 'B_38_last', 'B_38_first', 'D_63_last', 'D_63_first', 'D_64_last', 'D_64_first', 'D_66_last', 'D_66_first', 'D_68_last', 
                'D_68_first', 'D_114_last', 'D_114_first', 'D_116_last', 'D_116_first', 'D_117_last', 'D_117_first', 'D_120_last', 'D_120_first', 'D_126_last', 'D_126_first']

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

# https://www.kaggle.com/kyakovlev
# https://www.kaggle.com/competitions/amex-default-prediction/discussion/327534
def amex_metric_mod(y_true, y_pred):

    labels     = np.transpose(np.array([y_true, y_pred]))
    labels     = labels[labels[:, 1].argsort()[::-1]]
    weights    = np.where(labels[:,0]==0, 20, 1)
    cut_vals   = labels[np.cumsum(weights) <= int(0.04 * np.sum(weights))]
    top_four   = np.sum(cut_vals[:,0]) / np.sum(labels[:,0])

    gini = [0,0]
    for i in [1,0]:
        labels         = np.transpose(np.array([y_true, y_pred]))
        labels         = labels[labels[:, i].argsort()[::-1]]
        weight         = np.where(labels[:,0]==0, 20, 1)
        weight_random  = np.cumsum(weight / np.sum(weight))
        total_pos      = np.sum(labels[:, 0] *  weight)
        cum_pos_found  = np.cumsum(labels[:, 0] * weight)
        lorentz        = cum_pos_found / total_pos
        gini[i]        = np.sum((lorentz - weight_random) * weight)

    return 0.5 * (gini[1]/gini[0] + top_four)


# In[5]: Codificación de las variables

# Dummy encoding de las variables categóricas (ya tengo los dataframes finales, omitir)
if oh == False:
    train_df_oh, test_df_oh, dummies_train, dummies_test = fe.dummy_encoding(train, test, cat_features)
    del train, test, dummies_test, dummies_train, oh
elif oh == True:
    train_df_oh, test_df_oh = train, test
    del train, test, oh
gc.collect()

# In[6]: Separamos los datos 

# Primero añadimos la variable target a train_df_oh
train_df_oh_raw = train_df_oh.merge(train_labels, left_on='customer_ID', right_on='customer_ID')

# # # Transform train_df_oh_raw inf values to zero
# train_df_oh_raw = train_df_oh_raw.replace([np.inf, -np.inf], 0)

# # Transform test_df_oh inf values to nan
# test_df_oh = test_df_oh.replace([np.inf, -np.inf], np.nan)

# Definimos X e y
X = train_df_oh_raw.drop(columns = ['target', 'customer_ID']) 
y = train_df_oh_raw['target']

del train_df_oh, test_df_oh, train_df_oh_raw
gc.collect()


# In[7]: Parámetros XGBoost

# XGB MODEL PARAMETERS
xgb_parms = { 
    'max_depth':4, 
    'learning_rate':0.05, 
    'subsample':0.8,
    'colsample_bytree':0.6, 
    'eval_metric':'logloss',
    'objective':'binary:logistic',
    'tree_method':'gpu_hist',
    'predictor':'gpu_predictor',
    'random_state':42
}


# In[8]: XGBoost con Stratified K-Fold Cross Validation

# Diccionario para guardar las importancias de cada fold
importances = {} 
# Diccionario para guardar los scores de cada fold
scores = {'AMEX': []} 
# Necesitamos el tiempo para generar la carpeta donde guardar los modelos
current_time = time.strftime('%Y%m%d_%H%M%S')

"""Como ya tenemos los modelos calculados y estamos empleando siempre la misma semilla, se van a generar las mismas particiones en cada fold. 
Por tanto, como necesitamos las particiones para calcular el permutation feature importance, vamos a guardarlas en un excel para poder cargarlas en el futuro.
al ser un algoritmo computacionalmente muy costoso (8 horas por cada fold), vamos a lanzarlo poco a poco, 1 fold cada vez, en lugar de iterar sobre los 5
al mismo tiempo, e iremos guardando los resultados"""

def skf_func(X_input, y_input, folds):

    # Vamos a hacer un stratified k-fold cross validation con 5 folds
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    split = skf.split(X_input, y)

    # Creamos un diccionario para guardar las particiones
    split_dict = {}

    # Creamos el bucle para hacer el cross validation
    for fold, (train_index, valid_index) in enumerate(split):

        # Mensajes informativos
        print('-'*50)
        print('Fold:',fold+1)
        print( 'Train size:', len(train_index), 'Validation size:', len(valid_index))
        print('-'*50)

        # Separamos los datos en entrenamiento y validación
        X_train, X_valid = X_input.iloc[train_index], X_input.iloc[valid_index]
        y_train, y_valid = y_input.iloc[train_index], y_input.iloc[valid_index]

        # Guardamos las X_train, X_valid, y_train, y_valid en el diccionario
        split_dict['X_train_fold_'+str(fold+1)] = X_train
        split_dict['X_valid_fold_'+str(fold+1)] = X_valid
        split_dict['y_train_fold_'+str(fold+1)] = y_train
        split_dict['y_valid_fold_'+str(fold+1)] = y_valid

    return split_dict

skf_func(X, y, 5)