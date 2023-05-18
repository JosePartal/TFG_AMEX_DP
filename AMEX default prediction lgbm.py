# AMEX default prediction

# TEST ÁRBOLES DE DECISIÓN (imput missings)

# In[1]: Librerías

# Data manipulation
import pandas as pd 
import numpy as np
import gc

# Time management
import time

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score  
from sklearn.metrics import precision_score                         
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# Feature engineering functions
import feature_engineering as fe


# In[2]: Lectura de datos

train_labels, train, test = fe.load_datasets(oh=True)


# In[3]: Tipos de variables

# Lista de variables categóricas
# cat_features = ['B_30_last', 'B_30_first', 'B_38_last', 'B_38_first', 'D_63_last', 'D_63_first', 'D_64_last', 'D_64_first', 'D_66_last', 'D_66_first', 'D_68_last', 
#                 'D_68_first', 'D_114_last', 'D_114_first', 'D_116_last', 'D_116_first', 'D_117_last', 'D_117_first', 'D_120_last', 'D_120_first', 'D_126_last', 'D_126_first']

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

# Definimos X e y
X = train_df_oh.drop(columns = ['target', 'customer_ID']) 
y = train_df_oh['target']

del train_df_oh, test
gc.collect()


# In[6]: Parámetros LGBM

# Parámetros LGBM (usando dart)

LGBM_params = {
                  'objective' : 'binary',
                  'metric' : 'binary_logloss',
                  'boosting': 'dart',
                  'max_depth' : -1,
                  'num_leaves' : 64,
                  'learning_rate' : 0.1,
                  'bagging_freq': 5,
                  'bagging_fraction' : 0.75,
                  'feature_fraction' : 0.05,
                  'min_data_in_leaf': 256,
                  'max_bin': 63,
                  'min_data_in_bin': 256,
                  # 'min_sum_heassian_in_leaf': 10,
                  'tree_learner': 'serial',
                  'boost_from_average': 'false',
                  'lambda_l1' : 0.1,
                  'lambda_l2' : 30,
                  'num_threads': -1,
                  'force_row_wise' : True,
                  'verbosity' : 1,
    }
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

# In[7]: Hyperparameter tuning


# In[8]: LGBM con StratifiedKFold

# Definimos los folds
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
split = skf.split(X, y)

# Diccionario para guardar los scores de cada fold
scores = {'AMEX': [], 'Accuracy': [], 'Recall': [], 'Precision': [], 'F1': []}

# Definimos las listas para guardar las predicciones
y_pred_train = []
y_pred_valid = []

# Definimos las listas para guardar los modelos
models = []

# Definimos las listas para guardar las importancias de las variables
feature_importances = pd.DataFrame()
feature_importances['feature'] = features

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

    # Creamos el dataset de entrenamiento incluyendo las variables categóricas
    lgb_train = lgb.Dataset(X_train, y_train)

    # Creamos el dataset de validación incluyendo las variables categóricas
    lgb_valid = lgb.Dataset(X_valid, y_valid)

    # Entrenamos el modelo
    model_lgb = lgb.train(params=LGBM_params, train_set=lgb_train, num_boost_round=6000, valid_sets=[lgb_train, lgb_valid],
                            callbacks=[lgb.log_evaluation(period=100)])
    
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
    print('Métrica de Kaggle para el fold {fold}:', AMEX_score)
    scores['AMEX'].append(AMEX_score)

    # Libera memoria
    del X_train, X_valid, y_train, y_valid, lgb_train, lgb_valid
    gc.collect()

# Mostramos los resultados
print('-'*50)
print('Métrica de Kaggle:', np.mean(scores['AMEX']))


# In[9]: Feature importance

# Plot de las 150 variables más importantes
feature_importances['average'] = feature_importances[[f'fold_{fold + 1}' for fold in range(skf.n_splits)]].mean(axis=1)
feature_importances.sort_values(by='average', ascending=False, inplace=True)
feature_importances.reset_index(drop=True, inplace=True)

plt.figure(figsize=(10, 30))
sns.barplot(x='average', y='feature', data=feature_importances.head(150))
plt.title('150 variables más importantes')
plt.tight_layout()
plt.savefig(f'feature_importances_{current_time}.png')
plt.show()

# Guardamos el dataframe de importancia de variables en un archivo excel
feature_importances.to_excel(f'feature_importances_{current_time}.xlsx', index=False)


# In[13]: Predicciones



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