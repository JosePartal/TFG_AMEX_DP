# AMEX default prediction

# TEST ÁRBOLES DE DECISIÓN (imput missings)

# In[1]: Librerías

# Data manipulation
import pandas as pd 
import numpy as np
import gc

# Visualización
import matplotlib.pyplot as plt
import seaborn as sns

# Time management
import time

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold 

# Feature engineering functions
import feature_engineering as fe


# In[2]: Lectura de datos

train_labels, train, test = fe.load_datasets(oh=True)


# In[3]: Tipos de variables

# cat_features = ['B_30', 'B_38', 'D_63', 'D_64', 'D_66', 'D_68', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126']

# Como se han agrupado los datos, las categóricas tendrán el mismo nombre que las variables de la lista anterior que contengan seguidos de "_last" o "_first"
# Por ejemplo, B_30_last, B_30_first, B_38_last, B_38_first, etc.

# Lista de variables categóricas
cat_features = ['B_30_last', 'B_30_first', 'B_38_last', 'B_38_first', 'D_63_last', 'D_63_first', 'D_64_last', 'D_64_first', 'D_66_last', 'D_66_first', 'D_68_last', 
                'D_68_first', 'D_114_last', 'D_114_first', 'D_116_last', 'D_116_first', 'D_117_last', 'D_117_first', 'D_120_last', 'D_120_first', 'D_126_last', 'D_126_first']

# Lista de variables (exlucyendo 'customer_ID)
features = list(train.columns)
features.remove('customer_ID')


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


# In[5]: Separamos datos 

# Primero añadimos la variable target a train_df_oh
train_df_oh = train.merge(train_labels, left_on='customer_ID', right_on='customer_ID')

# # Selección de variables basada en PIMP
train_df_oh = fe.select_model_features(train_df_oh, -0.000001, 'lgbm')

# Definimos X e y
X = train_df_oh.drop(columns = ['target', 'customer_ID']) 
y = train_df_oh['target']

del train_df_oh, test
gc.collect()


# In[6]: Parámetros LGBM

# Parámetros LGBM (usando dart)

# LGBM_params = {
#                   'objective' : 'binary',
#                   'metric' : 'binary_logloss',
#                   'boosting': 'dart',
#                   'max_depth' : -1,
#                   'num_leaves' : 64,
#                   'learning_rate' : 0.3,
#                   'bagging_freq': 5,
#                   'bagging_fraction' : 0.75,
#                   'feature_fraction' : 0.05,
#                   'min_data_in_leaf': 256,
#                   'max_bin': 63,
#                   'min_data_in_bin': 256,
#                   # 'min_sum_heassian_in_leaf': 10,
#                   'tree_learner': 'voting',
#                   'boost_from_average': 'false',
#                   'lambda_l1' : 0.1,
#                   'lambda_l2' : 30,
#                   'num_threads': -1,
#                   'force_row_wise' : True,
#                   'verbosity' : 0,
#                   'device' : 'gpu',
#     }
# # https://www.kaggle.com/code/ragnar123/amex-lgbm-dart-cv-0-7977#Training-&-Inference
# LGBM_params2 = {
#         'objective': 'binary',
#         'metric': 'binary_logloss',
#         'boosting': 'dart',
#         'seed': 42,
#         'num_leaves': 100,
#         'learning_rate': 0.01,
#         'feature_fraction': 0.20,
#         'bagging_freq': 10,
#         'bagging_fraction': 0.50,
#         'n_jobs': -1,
#         'lambda_l2': 2,
#         'min_data_in_leaf': 40,
#         }

# lgb_params = {
#     'boosting_type': 'dart',
#     'objective': 'cross_entropy', 
#     'metric': ['AUC'],
#     'subsample': 0.8,  
#     'subsample_freq': 1,
#     'learning_rate': 0.01, 
#     'num_leaves': 2 ** 6, 
#     'min_data_in_leaf': 2 ** 11, 
#     'feature_fraction': 0.2, 
#     'feature_fraction_bynode':0.3,
#     'first_metric_only': True,
#     'n_estimators': 17001,  # -> 5000 for gbdt 
#     'boost_from_average': False,
#     'early_stopping_rounds': 300,
#     'verbose': -1,
#     'num_threads': -1,
#     'seed': SEED,
# }

# LGBM_params = {
#     'objective': 'binary',
#     'metric': 'binary_logloss',
#     'boosting': 'dart',
#     'max_depth' : 6,
#     'num_leaves' : 64,
#     'learning_rate' : 0.3,
#     'bagging_freq': 5,
#     'bagging_fraction' : 0.75,
#     'feature_fraction' : 0.1,
#     'min_data_in_leaf': 256,
#     'max_bin': 50,
#     'min_data_in_bin': 256,
#     # 'min_sum_heassian_in_leaf': 10,
#     'tree_learner': 'voting',
#     'boost_from_average': 'false',
#     'lambda_l1' : 0.1,
#     'lambda_l2' : 30,
#     'num_threads': -1,
#     'force_row_wise' : True,
#     'verbosity' : 0,
#     'device' : 'gpu',
#     'min_gain_to_split' : 0.001,
#     'early_stopping_round' : 100,
#     }

LGBM_params = {'boosting_type': 'gbdt',
            'n_estimators': 2500,
            'num_leaves': 50,
            'learning_rate': 0.05,
            'colsample_bytree': 0.7,
            'min_data_in_leaf': 1500,
            'max_bins': 63,
            # 'reg_alpha': 2,
            'objective': 'binary',
            'metric': 'binary_logloss',
            'force_col_wise': True,
            'random_state': 42}


# In[7]: Hyperparameter tuning


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

# Definimos las listas para guardar las importancias de las variables
feature_importances = pd.DataFrame()
feature_importances['feature'] = list(X.columns)

# Generamos la fecha para guardar los outpus en el mismo directorio
current_time = time.strftime('%Y%m%d_%H%M%S')

# Creamos el bucle para hacer cross validation
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

    # Medimos el tiempo de entrenamiento de cada fold (start)
    start = time.time()

    # Entrenamos el modelo para el fold actual
    model_lgb = lgb.train(params=LGBM_params, train_set=lgb_train, num_boost_round=2500, valid_sets=[lgb_train, lgb_valid],
                            callbacks=[lgb.log_evaluation(period=100), lgb.early_stopping(100)])
    
    # Medimos el tiempo de entrenamiento de cada fold (end)
    end = time.time()
    print('Tiempo de entrenamiento:', end - start)
    
    # Guardamos el modelo
    models.append(model_lgb)
    fe.save_model_fe('LGBM', model_lgb, fold, current_time)
    
    # Importancia de las variables
    feature_importances[f'fold_{fold + 1}'] = model_lgb.feature_importance(importance_type='split')

    # Predecimos sobre el conjunto de validación
    y_pred = model_lgb.predict(X_valid)

    # Guardamos las predicciones
    y_pred_train.append(model_lgb.predict(X_train))

    # Evaluamos las predicciones con la métrica customizada
    AMEX_score = amex_metric_mod(y_valid.values, y_pred)
    print(f'Métrica de Kaggle para el fold {fold}:', AMEX_score)
    scores['AMEX'].append(AMEX_score)

    # # Predicciones sobre el conjunto de test
    # X_test = test.drop(columns=['customer_ID'])
    # y_pred_test = model_lgb.predict(X_test)
    # print(f'Prediction for fold {fold} done')

    # # Creamos un dataframe con las predicciones
    # submission = pd.DataFrame({'customer_ID': test['customer_ID'], 'prediction': y_pred_test})

    # # Guardamos el dataframe en un csv
    # submission.to_csv(f'C:/Users/Jose/Documents/UNIVERSIDAD/TFG/MATEMATICAS/PYTHON/MODELOS/LGBM_{current_time}/submission_lastob_lgbm_{fold}.csv', index=False)
    # print(f'Submission for fold {fold} done')

    # Libera memoria
    del X_train, X_valid, y_train, y_valid, lgb_train, lgb_valid
    gc.collect()

# Mostramos los resultados
print('-'*50)
print('Valor medio de la métrica de Kaggle para todos los folds:', np.mean(scores['AMEX']))


# In[9]: Feature importance

# Plot de las 150 variables más importantes
feature_importances['average'] = feature_importances[[f'fold_{fold + 1}' for fold in range(skf.n_splits)]].mean(axis=1)
feature_importances.sort_values(by='average', ascending=False, inplace=True)
feature_importances.reset_index(drop=True, inplace=True)

plt.figure(figsize=(10, 30))
sns.barplot(x='average', y='feature', data=feature_importances.head(150))
plt.title('150 variables más importantes')
plt.tight_layout()
plt.savefig(f'./MODELOS/LGBM_{current_time}/feature_importances_{current_time}.png')
plt.show()

# Guardamos el dataframe de importancia de variables en un archivo excel
feature_importances.to_excel(f'./MODELOS/LGBM_{current_time}/feature_importances_{current_time}.xlsx', index=False)


# In[13]: Predicciones

def test_predictions(model_name, threshold, test_df, nfolds=5):
    # Cargamos datos de test si no están cargados
    if test_df is None:
        test_df = pd.read_parquet('./DATASETS/combined_dataset/test_df_oh.parquet')
        print('Test data loaded')

    # Seleccionamos las variables del modelo
    if threshold is not None:
        test_df = fe.select_model_features(test_df, threshold, 'lgbm')
        print(f'Test data features selected based on PIMP.')
    else:
        test_df = test_df.drop(columns=['customer_ID'])
        print(f'Test data features not selected based on PIMP.')

    X_test = test_df.drop(columns=['customer_ID'])

    # Iteramos sobre cada fold para calucular las predicciones de cada modelo
    for fold in range(nfolds):
        # Cargamos el modelo
        model_lgb = lgb.Booster(model_file=f'./MODELOS/LGBM_{model_name}/LGBM_model_{fold}.json')
        print(f'Model for fold {fold} loaded')
        # Predecimos sobre test
        y_pred_test =  model_lgb.predict(X_test)
        print(f'Prediction for fold {fold} done')
        # Creamos un dataframe con las predicciones
        submission = pd.DataFrame({'customer_ID': test_df['customer_ID'], 'prediction': y_pred_test})
        # Guardamos el dataframe en un csv
        submission.to_csv(f'./MODELOS/LGBM_{model_name}/submission_{model_name}_{fold}.csv', index=False)
        print(f'Submission for fold {fold} done')
        # Liberamos memoria
        del y_pred_test, submission
        gc.collect()

# test_predictions('20230620_111308', 0, None)


# # In[16]: Curva ROC de cada fold

# # Curva ROC de cada fold
# from sklearn.metrics import roc_curve, auc
# fpr, tpr, thresholds = roc_curve(y_test, y_pred)
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

# # In[17]: Matriz de confusión

# from sklearn.metrics import confusion_matrix
# import seaborn as sns

# cm = confusion_matrix(y_test, y_pred.round())
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

# # In[18]: Feature importance

# # Feature importance top 50
# lgb.plot_importance(model_train, max_num_features=50, figsize=(10,10))


# # In[19]: Metrica Gini

# y_pred1=pd.DataFrame(data={'prediction':y_pred})
# y_true1=pd.DataFrame(data={'target':y_test.reset_index(drop=True)})

# metric_score = amex_metric(y_true1, y_pred1)
# print('Gini: ', metric_score)
# # %%
# %%
