# AMEX default prediction XGBoost

# In[1]: Librerías

# Store and organize output files
from pathlib import Path

# Data manipulation
import pandas as pd 
import numpy as np
import time

# Data visualization
import matplotlib as mpl  
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import gc

# Time management
import time

# Machine learning
import imblearn
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold 

# Saving models
import pickle

# Feature engineering functions
import feature_engineering as fe


# In[2]: Lectura de datos

# Train
train = pd.read_parquet('C:/Users/Jose/Documents/UNIVERSIDAD/TFG/amex-default-prediction/parquet_ds_integer_dtypes/train.parquet')
# Labels
train_labels = pd.read_csv('C:/Users/Jose/Documents/UNIVERSIDAD/TFG/amex-default-prediction/train_labels.csv', low_memory=False)
# Train + Labels
# train_raw = train.merge(train_labels, left_on='customer_ID', right_on='customer_ID')
# train_raw = train_raw.drop(columns = ['customer_ID', 'S_2'])
# Test
test_data = pd.read_parquet('C:/Users/Jose/Documents/UNIVERSIDAD/TFG/amex-default-prediction/parquet_ds_integer_dtypes/test.parquet')
# test_data = test_data.drop(columns = ['customer_ID', 'S_2'])


# In[3]: Tipos de variables

not_used, cat_features, bin_features_1, bin_features_2, bin_features_3, num_features = fe.feature_types(train)


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

# Dummy encoding

train_df_oh, test_df_oh, dummies_train, dummies_test = fe.dummy_encoding(train, test_data, cat_features)

# Limpiamos la memoria
del train, test_data, dummies_train, dummies_test, not_used, bin_features_1, bin_features_2, bin_features_3, num_features
gc.collect()


# In[6]: Separamos los datos en entrenamiento y test

# Primero añadimos la variable target a train_df_oh
train_df_oh_raw = train_df_oh.merge(train_labels, left_on='customer_ID', right_on='customer_ID')
train_df_oh_raw = train_df_oh_raw.groupby('customer_ID').tail(1).set_index('customer_ID') # Última observación

# Definimos X e y
X = train_df_oh_raw.drop(columns = ['S_2', 'target'])
y = train_df_oh_raw['target']

# # # Separamos los datos en entrenamiento y test
# X_train, X_valid, y_train, y_valid = train_test_split(X, y, stratify = y, test_size = .20, random_state = 42, shuffle=True)

# print('Datos entrenamiento: ', X_train.shape)
# print('Datos test: ', X_valid.shape)


# In[7]: Parámetros XGBoost

# Parámetros LGBM (usando dart)

LGBM_params = {
                  'objective' : 'binary',
                  'metric' : 'binary_logloss',
                  'boosting': 'dart',
                  'max_depth' : -1,
                  'num_leaves' : 64,
                  'learning_rate' : 0.3,
                  'bagging_freq': 5,
                  'bagging_fraction' : 0.75,
                  'feature_fraction' : 0.05,
                  'min_data_in_leaf': 256,
                  'max_bin': 63,
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
                  'min_gain_to_split': 0.001,
                  'early_stopping_rounds': 100,
    }


# In[8]: LGBM con StratifiedKFold

# Definimos los folds
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
split = skf.split(X, y)

# Diccionario para guardar los scores de cada fold
scores = {'AMEX': []}

# Definimos las listas para guardar las predicciones
y_pred_train = []
y_pred_valid = []

# Definimos las listas para guardar los modelos
models = []

# Generamos la fecha para guardar los outpus en el mismo directorio
current_time = time.strftime('%Y%m%d_%H%M%S')

# Creamos el bucle para hacer el cross validation
for fold, (train_index, valid_index) in enumerate(split):

    # Mensajes informativos
    print('-'*50)
    print('Fold:',fold+1)
    print( 'Train size:', len(train_index), 'Validation size:', len(valid_index))
    print('-'*50)

    # Separamos los datos en entrenamiento y validación
    X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

    # Creamos el dataset de entrenamiento y validación incluyendo las variables categóricas
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_valid = lgb.Dataset(X_valid, y_valid)

    # Medimos el tiempo de entrenamiento para cada fold: start
    start = time.time()

    # Entrenamos el modelo para el fold actual
    model_lgb = lgb.train(params=LGBM_params, train_set=lgb_train, num_boost_round=3000, valid_sets=[lgb_train, lgb_valid],
                            callbacks=[lgb.log_evaluation(period=100), lgb.early_stopping(100)])
    
    # Medimos el tiempo de entrenamiento para cada fold: end
    end = time.time()
    print(f'Tiempo de entrenamiento para el fold {fold}:', end - start)
    
    # Guardamos el modelo
    models.append(model_lgb)
    fe.save_model_fe('LGBM', model_lgb, fold, current_time)

    # Predecimos sobre el conjunto de validación
    y_pred = model_lgb.predict(X_valid)

    # Guardamos las predicciones
    y_pred_train.append(model_lgb.predict(X_train))
    
    # Calculamos el score para el fold actual con la métrica customizada
    AMEX_score = amex_metric_mod(y_valid.values, y_pred) # DA ERROR LA ORIGINAL
    print(f'Métrica de Kaggle para el fold {fold}:', AMEX_score)
    scores['AMEX'].append(AMEX_score)

    # # Predicciones sobre el conjunto de test
    # test_df_oh = test_df_oh.groupby('customer_ID').tail(1).set_index('customer_ID') # Última observación
    # test_df_oh = test_df_oh.reset_index()
    # X_test = test_df_oh.drop(columns=['customer_ID', 'S_2'])
    # dtest = xgb.DMatrix(X_test, feature_names=X_test.columns, nthread=-1, enable_categorical=True)
    # y_pred_test = xgb_model.predict(dtest)
    # print(f'Prediction for fold {fold} done')
    # # Creamos un dataframe con las predicciones
    # submission = pd.DataFrame({'customer_ID': test_df_oh['customer_ID'], 'prediction': y_pred_test})
    # # Guardamos el dataframe en un csv
    # submission.to_csv(f'C:/Users/Jose/Documents/UNIVERSIDAD/TFG/MATEMATICAS/PYTHON/MODELOS/XGBoost_{current_time}/submission_lastob_xgb_{fold}.csv', index=False)
    # print(f'Submission for fold {fold} done')

    # Liberamos memoria
    del X_train, X_valid, y_train, y_valid, lgb_train, lgb_valid
    gc.collect()


# Mostramos los resultados
print('-'*50)
print('Valor medio de la métrica de Kaggle para todos los folds:', np.mean(scores['AMEX']))


# # In[10]: Save model

# fe.save_model('XGB_model', xgb_model)


# # In[10]: Métrica

# # Calculamos la métrica
# y_pred1=pd.DataFrame(data={'prediction':y_pred})
# y_true1=pd.DataFrame(data={'target':y_valid.reset_index(drop=True)})

# metric_score = amex_metric(y_true1, y_pred1)
# print('Gini: ', metric_score)


# # In[11]: Curva ROC

# # Curva ROC de cada fold
# from sklearn.metrics import roc_curve, auc
# fpr, tpr, thresholds = roc_curve(y_valid, y_pred)
# roc_auc = auc(fpr, tpr)

# plt.figure()
# lw = 2
# plt.plot(fpr, tpr, color='darkorange',
#             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC')
# plt.legend(loc="lower right")
# plt.show()


# # In[12]: Matriz de confusión

# from sklearn.metrics import confusion_matrix
# import seaborn as sns

# cm = confusion_matrix(y_valid, y_pred.round())
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

# %%
