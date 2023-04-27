# AMEX default prediction XGBoost

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

# Time management
import time

# Machine learning
import imblearn
import lightgbm as lgb
import xgboost as xgb
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


# In[2]: Workspace

# Create a directory to store the output files

results_path = Path('./MODELOS')
results_path.mkdir(exist_ok=True)

# Name experiment XGBoost + current time

experiment_name = 'XGBoost' + '_' + time.strftime('%Y%m%d_%H%M%S')
experiment_dir = results_path / experiment_name
experiment_dir.mkdir(exist_ok=True)

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


# In[5]: Codificación de las variables

# Creamos una copia de train_data para trabajar con ella
train_data_2 = train_raw.copy()

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


# In[6]: Separamos los datos

X = train_encoded.drop(['target'],axis=1)
y = train_encoded['target']

# Dividimos los datos en entrenamiento y test (80 training, 20 test)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size = .20, random_state = 42, shuffle=True)

print('Datos entrenamiento: ', X_train.shape)
print('Datos test: ', X_test.shape)


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


# In[8]: XGBoost

# Creamos el dataset de entrenamiento indicando las variables categóricas
# (Cambiar a DeviceQuantileDMatrix, es mucho más rápido)

dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=X_train.columns, nthread=-1, enable_categorical=True)
dtest = xgb.DMatrix(X_test, label=y_test, feature_names=X_test.columns, nthread=-1, enable_categorical=True)

# Entrenamos el modelo

xgb_model = xgb.train(xgb_parms, dtrain, num_boost_round=1000, evals=[(dtrain,'train'),(dtest,'test')], 
                      early_stopping_rounds=50, verbose_eval=50)


# In[9]: Evaluación del modelo
# Predecimos sobre el conjunto de test

y_pred = xgb_model.predict(dtest)


# In[10]: Métrica

# Calculamos la métrica
y_pred1=pd.DataFrame(data={'prediction':y_pred})
y_true1=pd.DataFrame(data={'target':y_test.reset_index(drop=True)})

metric_score = amex_metric(y_true1, y_pred1)
print('Gini: ', metric_score)


# In[11]: Curva ROC

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


# In[12]: Matriz de confusión

from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test, y_pred.round())
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')


# In[13]: Feature importance

# Feature importance
xgb.plot_importance(xgb_model, max_num_features=20, height=0.5)

# %%
