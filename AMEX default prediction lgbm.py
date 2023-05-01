# AMEX default prediction

# TEST ÁRBOLES DE DECISIÓN (imput missings)

# In[1]: Librerías

# Store and organize output files
from pathlib import Path

# Data manipulation
import pandas as pd 
import numpy as np

# Data visualization
import matplotlib as mpl  
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# Machine learning
import imblearn
import lightgbm as lgb
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold 
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler

# Librerías árboles de decisión
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import accuracy_score

# Librerías Random Forest
from sklearn.ensemble import RandomForestClassifier

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
train_raw = train.merge(train_labels, left_on='customer_ID', right_on='customer_ID')
train_raw = train_raw.drop(columns = ['customer_ID', 'S_2'])
# Test
test_data = pd.read_parquet('C:/Users/Jose/Documents/UNIVERSIDAD/TFG/amex-default-prediction/parquet_ds_integer_dtypes/test.parquet')
test_data = test_data.drop(columns = ['customer_ID', 'S_2'])


# In[3]: Tipos de variables

# Recordemos, en primer lugar, que las siguientes variables eran categóricas:

# `['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']`

# Y `[S_2]` es una variable temporal.

# Variables categóricas
categorical_features = ['B_30', 'B_38', 'D_63', 'D_64', 'D_66', 'D_68', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126']
train_raw[categorical_features] = train_raw[categorical_features].astype("category")
test_data[categorical_features] = test_data[categorical_features].astype("category")

features = train.drop(['customer_ID', 'S_2'], axis = 1).columns.to_list()
# Numerical features
numerical_features = [col for col in features if col not in categorical_features]


# In[4]: Detección de missings

# Veamos la cantidad y porcentaje de datos faltantes tenemos en cada variable
pd_series_null_columns = train_raw.isnull().sum().sort_values(ascending=False)
pd_series_null_rows = train_raw.isnull().sum(axis=1).sort_values(ascending=False)


pd_null_columnas = pd.DataFrame(pd_series_null_columns, columns=['nulos_columnas'])     
pd_null_filas = pd.DataFrame(pd_series_null_rows, columns=['nulos_filas'])  
pd_null_filas['target'] = train_raw['target'].copy()
pd_null_columnas['porcentaje_columnas'] = pd_null_columnas['nulos_columnas']/train_raw.shape[0]
pd_null_filas['porcentaje_filas']= pd_null_filas['nulos_filas']/train_raw.shape[1]

pd_null_columnas

#Creemos un vector de variables con datos faltantes

threshold = 0
list_vars_not_null = list(pd_null_columnas[pd_null_columnas['porcentaje_columnas'] == threshold].index)
list_var_null = list(pd_null_columnas[pd_null_columnas['porcentaje_columnas'] > threshold].index)
train_data = train_raw.loc[:, list_vars_not_null]
list_var_null

tmp = train_raw.isna().sum().div(len(train_raw)).mul(100).sort_values(ascending=False)

plt.style.use('Solarize_Light2')
fig, ax = plt.subplots(2,1, figsize=(25,10))
sns.barplot(x=tmp[:100].index, y=tmp[:100].values, ax=ax[0])
sns.barplot(x=tmp[100:].index, y=tmp[100:].values, ax=ax[1])
ax[0].set_ylabel("Percentage [%]"), ax[1].set_ylabel("Percentage [%]")
ax[0].tick_params(axis='x', rotation=90); ax[1].tick_params(axis='x', rotation=90)
plt.suptitle("Amount of missing data")
plt.tight_layout()
plt.show()

del tmp, fig, ax


# In[5]: Métrica

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

# In[6]: Semillas aleatorias

import random  
import os
seed = 42
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# In[7]: Codificación de las variables

# Creamos una copia de train_data para trabajar con ella
train_data_2 = train_raw.copy()

# Codificación de las variables
# Meto un label encoder por tener menos variables, pero habría que usar onehot porque no sé si existe una relación de orden entre las categorías

# enc = LabelEncoder()
# for cat_feat in categorical_features:
#     train_data_2[cat_feat] = enc.fit_transform(train_data_2[cat_feat])
#     test_data[cat_feat] = enc.transform(test_data[cat_feat])

# Error con label encoder porque hay categorías nuevas en test. Tal vez al agrupar por id pueda solucionarse. Paso a onehot

# Onehot encoding uising sklearn

enc = OneHotEncoder(handle_unknown='ignore')
# Ajustar y transformar los datos de entrenamiento
train_oh = enc.fit_transform(train_data_2[categorical_features])
# Transformar los datos de test
test_oh = enc.transform(test_data[categorical_features])

# Convertir los datos codificados a un DataFrame y añadir los nombres de las columnas
train_oh = pd.DataFrame(train_oh.toarray(), columns=enc.get_feature_names_out(categorical_features))
test_oh = pd.DataFrame(test_oh.toarray(), columns=enc.get_feature_names_out(categorical_features))

# Unir los datos codificados con los datos numéricos
train_encoded = train_data_2.join(train_oh)
test_encoded = test_data.join(test_oh)

del train_data_2

# In[8]: Separamos los datos

X = train_encoded.drop(['target'],axis=1)
y = train_encoded['target']

# Dividimos los datos en entrenamiento y test (80 training, 20 test)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size = .20, random_state = seed, shuffle=True)

print('Datos entrenamiento: ', X_train.shape)
print('Datos test: ', X_test.shape)


# In[9]: Parámetros LGBM

# Parámetros LGBM

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

from lightgbm import callback
callbacks = [callback.early_stopping(patience=100)]

# No es necesario escalar en árboles de decisión

# # In[9]: Escalamos (con los datos de train)
# scaler = preprocessing.StandardScaler().fit(X_train)

# X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns = X_train.columns, index = X_train.index)
# X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns = X_test.columns, index = X_test.index)

# In[10]: Imputación de missings

# # Para la imputación de missings, vamos a usar KNNImputer. Daremos más peso a los vecinos más próximos.
# from sklearn.impute import KNNImputer
# imputer = KNNImputer(n_neighbors=5, weights="distance")
# X_train_nomis = imputer.fit_transform(X_train_scaled)
# X_test = imputer.fit_transform(X_test_scaled)

# In[11]: Undersampling

# from imblearn.under_sampling import RandomUnderSampler

# under_sampler = RandomUnderSampler(random_state = seed)
# X_train_res, y_train_res = under_sampler.fit_resample(X_train_scaled, y_train)


# In[12]: Hyperparameter tuning


# In[13]: LGBM

# Tiempo de inicio
import time
start_time = time.time()

# Creamos el dataset de entrenamiento
lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=categorical_features)

# Creamos el dataset de test
lgb_test = lgb.Dataset(X_test, y_test, categorical_feature=categorical_features)

# Entrenamos el modelo
model_train = lgb.train(params=LGBM_params, train_set=lgb_train, num_boost_round=2000, valid_sets=[lgb_train, lgb_test], 
                        verbose_eval=100, callbacks=callbacks)

# Tiempo de ejecución
print('Tiempo de ejecución: ', time.time() - start_time)


# In[13]: Predicciones

start_time = time.time()

# Predicciones
y_pred = model_train.predict(X_test)

print('Tiempo de ejecución: ', time.time() - start_time)


# In[14]: Evaluación usando la métrica

# Evaluación usando la métrica
# print('Gini: ', amex_metric(y_test, y_pred))

# In[15]: Save model

# Save model
import pickle
pickle.dump(model_train, open('model_train.pkl', 'wb'))

# In[16]: Curva ROC de cada fold

# Curva ROC de cada fold
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
            lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="lower right")
plt.show()

# In[17]: Matriz de confusión

from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test, y_pred.round())
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

# In[18]: Feature importance

# Feature importance top 50
lgb.plot_importance(model_train, max_num_features=50, figsize=(10,10))


# In[19]: Metrica Gini

y_pred1=pd.DataFrame(data={'prediction':y_pred})
y_true1=pd.DataFrame(data={'target':y_test.reset_index(drop=True)})

metric_score = amex_metric(y_true1, y_pred1)
print('Gini: ', metric_score)
# %%
