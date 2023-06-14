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
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold 

# Funciones ingeniería de variables
import feature_engineering as fe

# Librería para monitorizar bucles
from tqdm import tqdm


# In[2]: Lectura de datos
# oh = True

# train_labels, train, test = fe.load_datasets(oh)

# # In[3]: Variables categóricas

# # cat_features = ['B_30', 'B_38', 'D_63', 'D_64', 'D_66', 'D_68', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126']

# # Como se han agrupado los datos, las categóricas tendrán el mismo nombre que las variables de la lista anterior que contengan seguidos de "_last" o "_first"
# # Por ejemplo, B_30_last, B_30_first, B_38_last, B_38_first, etc.

# # Lista de variables categóricas
# cat_features = ['B_30_last', 'B_30_first', 'B_38_last', 'B_38_first', 'D_63_last', 'D_63_first', 'D_64_last', 'D_64_first', 'D_66_last', 'D_66_first', 'D_68_last', 
#                 'D_68_first', 'D_114_last', 'D_114_first', 'D_116_last', 'D_116_first', 'D_117_last', 'D_117_first', 'D_120_last', 'D_120_first', 'D_126_last', 'D_126_first']

# # Lista de variables (exlucyendo 'customer_ID)
# features = list(train.columns)
# features.remove('customer_ID')
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


# # In[5]: Codificación de las variables

# # Dummy encoding de las variables categóricas (ya tengo los dataframes finales, omitir)
# if oh == False:
#     train_df_oh, test_df_oh, dummies_train, dummies_test = fe.dummy_encoding(train, test, cat_features)
#     del train, test, dummies_test, dummies_train, oh
# elif oh == True:
#     train_df_oh, test_df_oh = train, test
#     del train, test, oh
# gc.collect()

# # In[6]: Separamos los datos 

# # Primero añadimos la variable target a train_df_oh
# train_df_oh_raw = train_df_oh.merge(train_labels, left_on='customer_ID', right_on='customer_ID')

# # # # Transform train_df_oh_raw inf values to zero
# # train_df_oh_raw = train_df_oh_raw.replace([np.inf, -np.inf], 0)

# # # Transform test_df_oh inf values to nan
# # test_df_oh = test_df_oh.replace([np.inf, -np.inf], np.nan)

# # Definimos X e y
# X = train_df_oh_raw.drop(columns = ['target', 'customer_ID']) 
# y = train_df_oh_raw['target']

# del train_df_oh, test_df_oh, train_df_oh_raw
# gc.collect()


# In[7]: Parámetros LightGBM

# LightGBM MODEL PARAMETERS
LGBM_PARAMS = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting': 'dart',
    'max_depth' : 6,
    'num_leaves' : 64,
    'learning_rate' : 0.3,
    'bagging_freq': 5,
    'bagging_fraction' : 0.75,
    'feature_fraction' : 0.1,
    'min_data_in_leaf': 256,
    'max_bin': 50,
    'min_data_in_bin': 256,
    # 'min_sum_heassian_in_leaf': 10,
    'tree_learner': 'voting',
    'boost_from_average': 'false',
    'lambda_l1' : 0.1,
    'lambda_l2' : 30,
    'num_threads': -1,
    'force_row_wise' : True,
    'verbosity' : 0,
    'device' : 'gpu',
    'min_gain_to_split' : 0.001,
    'early_stopping_round' : 100,
    }


# In[8]: LightGBM con Stratified K-Fold Cross Validation

# Diccionario para guardar las importancias de cada fold
importances = {} 
# Diccionario para guardar los scores de cada fold
scores = {'AMEX': []} 

"""Como ya tenemos los modelos calculados y estamos empleando siempre la misma semilla, se van a generar las mismas particiones en cada fold. 
Por tanto, como necesitamos las particiones para calcular el permutation feature importance, vamos a guardarlas en un excel para poder cargarlas en el futuro.
al ser un algoritmo computacionalmente muy costoso, vamos a lanzarlo poco a poco, 1 fold cada vez, en lugar de iterar sobre los 5
al mismo tiempo, e iremos guardando los resultados"""

def skf_func(X_input, y_input, folds):

    # Vamos a hacer un stratified k-fold cross validation con 5 folds
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    split = skf.split(X_input, y_input)

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

# split_dict = skf_func(X, y, 5)


# In[9]: Guardamos las particiones en un excel

# Vamos a generar un archivo para cada key del diccionario y guardaremos datos en una carpeta para cada fold
# # En primer lugar crearemos una carpeta llamada PARTICIONES y dentro subcarpetas para cada fold
# import os
# os.mkdir('PARTICIONES')
# for i in range(1,6):
#     os.mkdir('PARTICIONES/FOLD_'+str(i))

# # Creamos un bucle para guardar cada X (son dataframes) en formato parquet
# for key in split_dict.keys():
#     if 'X' in key:
#         split_dict[key].to_parquet('PARTICIONES/FOLD_'+str(key[-1])+'/'+str(key)+'.parquet')
#     elif 'y' in key:
#         # Son pd.Series, no se pueden guardar en parquet, necesitamos transformarlas en dataframes primero
#         split_dict[key].to_frame().to_parquet('PARTICIONES/FOLD_'+str(key[-1])+'/'+str(key)+'.parquet')


# In[10]: Cargamos las particiones

# Función para cargar las particiones según el fold
def load_split(fold):
    # Cargamos las particiones
    # X_train = pd.read_parquet('PARTICIONES/FOLD_'+str(fold)+'/X_train_fold_'+str(fold)+'.parquet')
    X_valid = pd.read_parquet('PARTICIONES/FOLD_'+str(fold)+'/X_valid_fold_'+str(fold)+'.parquet')
    # y_train = pd.read_parquet('PARTICIONES/FOLD_'+str(fold)+'/y_train_fold_'+str(fold)+'.parquet')
    y_valid = pd.read_parquet('PARTICIONES/FOLD_'+str(fold)+'/y_valid_fold_'+str(fold)+'.parquet')

    # Transformamos las y en pd.Series
    # y_train = y_train.iloc[:,0]
    y_valid = y_valid.iloc[:,0]
    return X_valid, y_valid #, X_train, y_train

# X_valid, y_valid = load_split(1)

# In[11]: Calculamos el permutation feature importance

# Creamos una función para calcular el permutation feature importance de cada fold
def pimp_func(fold, X_valid, y_valid, current_time = '20230518_151655'):
    # Diccionario para guardar los scores de cada fold
    scores = {'AMEX': []} 

    # Cargamos el modelo
    lgbm_model = lgb.Booster(model_file='MODELOS/LGBM_' + current_time + '/' + 'LGBM_model_' + str(fold) + '.json')
    print('Modelo cargado')

    # Predecimos sobre el conjunto de validación
    y_pred = lgbm_model.predict(X_valid)

    # Calculamos el score con la métrica modificada
    AMEX_score = amex_metric_mod(y_valid.values, y_pred) 
    print(f'Métrica de Kaggle para el fold {fold}:', AMEX_score)
    scores['AMEX'].append(AMEX_score)

    # Calculamos el permutation feature importance
    print('-'*50)
    print('Calculando permutation feature importance...')

    # Creamos un diccionario para guardar los valores de la métrica
    perm_scores = {}

    # Creamos un bucle para calcular el valor de la métrica tras predecir habiendo permutado cada variable
    for col in tqdm(X_valid.columns):
        # Guardamos la variable original
        temp = X_valid.loc[:, col].copy()
        # Permutamos la columna actual
        X_valid.loc[:, col] = np.random.permutation(X_valid[col])
        # Predecimos sobre el conjunto de validación
        y_pred = lgbm_model.predict(X_valid)
        # Calculamos el score para el fold actual con la métrica customizada
        perm_scores[col] = amex_metric_mod(y_valid.values, y_pred)
        # Restauramos la columna original
        X_valid.loc[:, col] = temp

    # Creamos un dataframe con los scores
    perm_scores_df = pd.DataFrame.from_dict(perm_scores, orient='index').reset_index()
    # Calculamos la diferencia entre el score original y el score permutado
    perm_scores_df['score_diff'] = perm_scores_df[0] - AMEX_score
    # Ordenamos los scores por la diferencia
    perm_scores_df = perm_scores_df.sort_values('score_diff', ascending=False).reset_index(drop=True)
    # Guardamos el dataframe en un excel
    perm_scores_df.to_excel(f'C:/Users/Jose/Documents/UNIVERSIDAD/TFG/MATEMATICAS/PYTHON/MODELOS/LGBM_{current_time}/PIMP/permutation_feature_importance_{fold}.xlsx', index=True)

    # Plot Permutation Feature Importance: Top 100
    plt.figure(figsize=(10, 30))
    sns.barplot(x='score_diff', y='index', data=perm_scores_df[:100])
    plt.title('XGB Permutation Feature Importance: Top 100')
    plt.savefig(f'C:/Users/Jose/Documents/UNIVERSIDAD/TFG/MATEMATICAS/PYTHON/MODELOS/LGBM_{current_time}/PIMP/permutation_feature_importance_{fold}.png')
    plt.show()

    # Liberamos memoria
    del lgbm_model, y_pred, AMEX_score, perm_scores, perm_scores_df
    gc.collect()

# pimp_func(0, X_valid, y_valid)

# In[12]: Cálculo para cada fold

# # Fold 1
# X_valid, y_valid = load_split(1)
# pimp_func(0, X_valid, y_valid)

# # Fold 2
# X_valid, y_valid = load_split(2)
# pimp_func(1, X_valid, y_valid)

# # Fold 3
# X_valid, y_valid = load_split(3)
# pimp_func(2, X_valid, y_valid)

# # Fold 4
# X_valid, y_valid = load_split(4)
# pimp_func(3, X_valid, y_valid)

# # Fold 5
# X_valid, y_valid = load_split(5)
# pimp_func(4, X_valid, y_valid)