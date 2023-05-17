# AMEX default prediction XGBoost (agg)

# In[1]: Librerías

# Data manipulation
import pandas as pd 
import numpy as np
import time

# Data visualization
import gc

# Time management
import time

# Machine learning
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score  
from sklearn.metrics import precision_score                         
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# Librerías árboles de decisión
from sklearn.metrics import accuracy_score

# Feature engineering functions
import feature_engineering as fe


# In[2]: Lectura de datos

# Train
train = pd.read_parquet('C:/Users/Jose/Documents/UNIVERSIDAD/TFG/MATEMATICAS/PYTHON/DATASETS/combined_dataset/train_combined_dataset.parquet')
# Labels
train_labels = pd.read_csv('C:/Users/Jose/Documents/UNIVERSIDAD/TFG/amex-default-prediction/train_labels.csv', low_memory=False)
# Train + Labels
# train_raw = train.merge(train_labels, left_on='customer_ID', right_on='customer_ID')
# train_raw = train_raw.drop(columns = ['customer_ID', 'S_2'])
# Test
test = pd.read_parquet('C:/Users/Jose/Documents/UNIVERSIDAD/TFG/MATEMATICAS/PYTHON/DATASETS/combined_dataset/test_combined_dataset.parquet')
# test_data = test_data.drop(columns = ['customer_ID', 'S_2'])

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

# Dummy encoding de las variables categóricas

train_df_oh, test_df_oh, dummies_train, dummies_test = fe.dummy_encoding(train, test, cat_features)

del train, test, dummies_test, dummies_train
# In[6]: Separamos los datos en entrenamiento y test

# Primero añadimos la variable target a train_df_oh
train_df_oh_raw = train_df_oh.merge(train_labels, left_on='customer_ID', right_on='customer_ID')

# Definimos X e y
X = train_df_oh_raw.drop(columns = ['target']) # creo que no hace falta quitar S_2
y = train_df_oh_raw['target']


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

# Vamos a hacer un stratified k-fold cross validation con 5 folds
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
importances = [] # Lista para guardar las importancias de las variables de cada fold
scores = {'AMEX': [], 'Accuracy': [], 'Recall': [], 'Precision': [], 'F1': []} # Diccionario para guardar los scores de cada fold
split = skf.split(X, y)

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

    # Creamos el dataset de entrenamiento indicando las variables categóricas
    # (Cambiar a DeviceQuantileDMatrix, es mucho más rápido)

    dtrain = xgb.DeviceQuantileDMatrix(X_train, label=y_train, feature_names=X_train.columns, nthread=-1, enable_categorical=True)
    dvalid = xgb.DMatrix(X_valid, label=y_valid, feature_names=X_valid.columns, nthread=-1, enable_categorical=True)

    # Entrenamos el modelo para el fold actual

    xgb_model = xgb.train(xgb_parms, dtrain, num_boost_round=2500, evals=[(dtrain,'train'),(dvalid,'test')],
                            early_stopping_rounds=50, verbose_eval=50) # feval ver custom metric https://www.kaggle.com/code/jiweiliu/rapids-cudf-feature-engineering-xgb
    
    # Guardamos el modelo
    fe.save_model_fe('XGBoost', xgb_model, fold, current_time) # Crea una carpeta para cada minuto, cambiar para que lo guarde todo en la misma

    # Feature importance para el fold actual
    importances.append(xgb_model.get_score(importance_type='weight')) # ‘weight’ - the number of times a feature is used to split the data across all trees.

    # Predecimos sobre el conjunto de validación
    y_pred = xgb_model.predict(dvalid)
    
    # Calculamos el score para el fold actual con la métrica customizada
    AMEX_score = amex_metric_mod(y_valid.values, y_pred) # DA ERROR LA ORIGINAL
    print('Métrica de Kaggle para el fold {fold}:', AMEX_score)
    scores['AMEX'].append(AMEX_score)

    # Liberamos memoria
    # del X_train, X_valid, y_train, y_valid, dtrain, dvalid
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


# In[13]: Feature importance

# Plot top 50 features using plotly express
import plotly.express as px
fig = px.bar(pd.DataFrame(importances).T.sort_values(by=0, ascending=False).head(50), orientation='h')
fig.show()
# %%
