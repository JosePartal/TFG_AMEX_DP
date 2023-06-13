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
import optuna

# Feature engineering functions
import feature_engineering as fe

# Binning and WOE
import optbinning

# Progress bar
from tqdm import tqdm


# In[2]: Lectura de datos
oh=False

train_labels, train, test = fe.load_datasets(oh)

# if oh is False:
#     train = train.replace([np.inf, -np.inf], 0)
#     test = test.replace([np.inf, -np.inf], 0)

# # Fill train NaN with np.nan
# train = train.fillna(np.nan)
# test = test.fillna(np.nan)


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

# def amex_metric(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:

#     def top_four_percent_captured(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
#         df = (pd.concat([y_true, y_pred], axis='columns')
#               .sort_values('prediction', ascending=False))
#         df['weight'] = df['target'].apply(lambda x: 20 if x==0 else 1)
#         four_pct_cutoff = int(0.04 * df['weight'].sum())
#         df['weight_cumsum'] = df['weight'].cumsum()
#         df_cutoff = df.loc[df['weight_cumsum'] <= four_pct_cutoff]
#         return (df_cutoff['target'] == 1).sum() / (df['target'] == 1).sum()
        
#     def weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
#         df = (pd.concat([y_true, y_pred], axis='columns')
#               .sort_values('prediction', ascending=False))
#         df['weight'] = df['target'].apply(lambda x: 20 if x==0 else 1)
#         df['random'] = (df['weight'] / df['weight'].sum()).cumsum()
#         total_pos = (df['target'] * df['weight']).sum()
#         df['cum_pos_found'] = (df['target'] * df['weight']).cumsum()
#         df['lorentz'] = df['cum_pos_found'] / total_pos
#         df['gini'] = (df['lorentz'] - df['random']) * df['weight']
#         return df['gini'].sum()

#     def normalized_weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
#         y_true_pred = y_true.rename(columns={'target': 'prediction'})
#         return weighted_gini(y_true, y_pred) / weighted_gini(y_true, y_true_pred)

#     g = normalized_weighted_gini(y_true, y_pred)
#     d = top_four_percent_captured(y_true, y_pred)

#     return 0.5 * (g + d)

# https://www.kaggle.com/code/rohanrao/amex-competition-metric-implementations
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


# In[6]: Binning y WOE

# Vamos a usar la librería optbinning para hacer el binning y el WOE. Usaremos la función BinningProcess

# Creamos el objeto para el binning con special_codes identificando los NaN (special_codes must be a dit, list or numpy.ndarray)
# binning_fit = optbinning.BinningProcess(variable_names=features, categorical_variables= cat_features, special_codes=[np.nan], n_jobs=-1)

# # Save binning process --> ¿Debería ir después del fit?
# binning_fit.save('binning_process.pkl')

# Load binning process
binning_fit = optbinning.BinningProcess.load('DATASETS/binning_process.pkl')

# # Entrenamos el objeto con los datos de entrenamiento
binning_fit.fit(train_df[features], train_df['target'].to_numpy(), check_input=True)

# Hacemos el binning de los datos de entrenamiento
train_df_binned = binning_fit.transform(train_df[features])

# La tabla tran_df_binned contiene los valores WoE para cada bin de cada variable. 
# Vamos a renombrar las variables añadiendo "_woe" al final para diferenciarlas de las variables originales

# Renombramos las variables
train_df_binned.columns = [col + '_woe' for col in train_df_binned.columns]

# Añadimos la variable target
train_df_binned['target'] = train_df['target']

print('WoE calculado')


# In[7]: Binning y WOE II: Selección de variables basada en el IV

"""Eliminamos las variables con IV < 0.02"""

# Definimos el diccionario donde guardaremos los IV de cada variable
iv_score_dict = {}

# Calculamos el IV de cada variable. Tenemos que usar la función OptimalBinning para poder obtener el IV de cada variable
for col in tqdm(features):
    if col in cat_features:
        optb = optbinning.OptimalBinning(dtype='categorical')
        optb.fit(train_df[col], train_df['target'])
    else:
        optb = optbinning.OptimalBinning(dtype='numerical')
        optb.fit(train_df[col], train_df['target'])
    binning_table = optb.binning_table
    binning_table.build()
    iv_score_dict[col] = binning_table.iv

# Convertimos el diccionario en un DataFrame y lo ordenamos de mayor a menor
iv_score_df = pd.DataFrame.from_dict(iv_score_dict, orient='index', columns=['IV'])
iv_score_df = iv_score_df.sort_values(by='IV', ascending=False)

print('IV calculado')

# Nos quedamos con las variables con IV >= 0.02
iv_score_df = iv_score_df[iv_score_df['IV'] >= 0.02]

# Lista de variables con IV >= 0.02
selected_features = list(iv_score_df.index)

# Variables con IV <= 0.02
dropped_features = list(set(features) - set(selected_features))

print('Hemos eliminado', len(dropped_features), 'variables con IV <= 0.02')

# Calcular las variables WOE (default)
# woe_values = binning_fit.transform(train_df[features], metric="woe")

# Event rate
# event_rate = binning_fit.transform(train_df[features], metric="event_rate")

# # Transformamos los índices
# transform_indices = binning_fit.transform(train_df[features], metric="indices")

# # Transformamos los bins
# transform_bins = binning_fit.transform(train_df[features], metric="bins")


# # In[8]: PRUEBA I - Regresión logística usando variables Weight of Evidence (WOE)

# """Baseline para hacer la regresión logística usando las variables WOE.""" 

# # En primer lugar, definimos X e y
# X = train_df_binned.drop('target', axis=1)
# y = train_df_binned['target']

# # Separamos los datos en entrenamiento y validación
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# # Definimos el modelo
# model = LogisticRegression(random_state=42, max_iter=1000)

# # Entrenamos el modelo
# model.fit(X_train, y_train)

# # Predecimos los datos de validación
# y_pred = model.predict_proba(X_val)[:, 1]

# # pred_train = 1/(1+np.exp(-model.predict(X_train)))
# # pred_train_df = pd.DataFrame(pred_train, columns=['prediction'])

# # y_val_df = pd.DataFrame(y_val, columns=['target'])

# # Calculamos la métrica
# metric = amex_metric_np(y_pred, y_val.to_numpy())

# print(f'AMEX metric: {metric}')


# # In[9]: PRUEBA II - Regresión logística usando variables Weight of Evidence (WOE) 

# # Vamos a calcular la regresión logística usando las variables WoE. No emplearemos las variables con IV <= 0.02

# # En primer lugar, definimos X e y. Tenemos que añadir primero '_woe' a las variables seleccionadas
# selected_features = [col + '_woe' for col in selected_features]
# X = train_df_binned[selected_features]
# y = train_df_binned['target']

# # Separamos los datos en entrenamiento y validación
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# # Definimos el modelo
# model = LogisticRegression(random_state=42)

# # Entrenamos el modelo
# model.fit(X_train, y_train)

# # Predecimos los datos de validación
# y_pred = model.predict_proba(X_val)[:, 1]

# # Calculamos la métrica
# metric = amex_metric_np(y_pred, y_val.to_numpy())

# print(f'AMEX metric: {metric}')


# In[10]: Regresión logística usando variables Weight of Evidence (WOE) I: Separación de datos

# Vamos a calcular la regresión logística usando las variables WoE. No emplearemos las variables con IV <= 0.02

# En primer lugar, definimos X e y. Tenemos que añadir primero '_woe' a las variables seleccionadas
selected_features = [col + '_woe' for col in selected_features]
X = train_df_binned[selected_features]
y = train_df_binned['target']


# In[11]: Regresión logística usando variables Weight of Evidence (WOE) II: Cross-validation 

"""Lo tengo para una regresión L2 (Ridge) ahora mismo, si quiero una L1 (Lasso) tengo que cambiar el penalty y el solver a 'saga'."""

# Creamos un diccionario para guardar los scores de cada fold
scores = {'AMEX': []} 
# Creamos un diccionario para guardar los coeficientes de cada variable para cada fold
coefficients = {}

# Necesitamos el tiempo para generar la carpeta donde guardar los modelos
current_time = time.strftime('%Y%m%d_%H%M%S')

# Definimos una función para calcular la regresión logística usando CV
def logistic_regression_func(X_input, y_input, folds, current_time, intercept: bool, hyperopt: bool):

    # Stratifed K-Fold
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    split = skf.split(X_input, y_input)

    # Creamos el bucle para hacer el cross validation
    for fold, (train_index, valid_index) in enumerate(split):
        # Mensajes informativos
        print('-'*50)
        print('Fold:',fold+1)
        print( 'Train size:', len(train_index), 'Validation size:', len(valid_index))
        print('-'*50)

        # Separamos los datos en entrenamiento y validación
        X_train, X_val = X_input.iloc[train_index], X_input.iloc[valid_index]
        y_train, y_val = y_input.iloc[train_index], y_input.iloc[valid_index]

        if hyperopt is True:
            print(f'Optimización de hiperparámetros para el fold {fold+1}')
            # Optimización de hiperparámetros con optuna: C
            def objective(trial):
                # Definimos los hiperparámetros
                C = trial.suggest_loguniform('C', 1e-10, 1e10)

                # Definimos el modelo
                model = LogisticRegression(random_state=42, penalty='l2', max_iter=1000, fit_intercept= intercept, 
                                        C=C, solver='lbfgs', n_jobs=-1)

                # Entrenamos el modelo
                model.fit(X_train, y_train)

                # Predecimos los datos de validación
                y_pred = model.predict_proba(X_val)[:, 1]

                # Calculamos la métrica
                metric = amex_metric_np(y_pred, y_val.to_numpy())

                return metric
            
            # Ejecutamos optuna
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=100)

            # Obtenemos los mejores hiperparámetros
            best_params = study.best_params

            # Definimos el modelo
            log_model = LogisticRegression(random_state=42, penalty='l2', max_iter=1000, fit_intercept = intercept,
                                        C=best_params['C'], solver='lbfgs', n_jobs=-1)
            
        else:
            # Definimos el modelo
            print('No se ha realizado la optimización de hiperparámetros')
            log_model = LogisticRegression(random_state=42, penalty='l2', max_iter=1000, fit_intercept = intercept, 
                                           solver='lbfgs', n_jobs=-1)

        # Entrenamos el modelo
        print('Entrenamiento del modelo')
        log_model.fit(X_train, y_train)

        # Predecimos los datos de validación
        y_pred = log_model.predict_proba(X_val)[:, 1]

        # Guardamos el modelo
        fe.save_model_fe('LogReg', log_model, fold, current_time)

        # Calculamos la métrica
        AMEX_score = amex_metric_np(y_pred, y_val.to_numpy())
        print(f'Métrica de Kaggle para el fold {fold}:', AMEX_score)
        scores['AMEX'].append(AMEX_score)

        # Tabla con cada variable y su coeficiente del fold actual
        coefficients[fold] = pd.DataFrame({'feature': X_train.columns, 'coef': log_model.coef_[0]})

        # Liberamos memoria
        del X_train, X_val, y_train, y_val, log_model, y_pred
        gc.collect()

    # Mostramos los resultados
    print('-'*50)
    print('Valor medio de la métrica de Kaggle para todos los folds:', np.mean(scores['AMEX']))

logistic_regression_func(X, y, 5, current_time, intercept=True, hyperopt=False)


# %%: Regresión logística usando variables Weight of Evidence (WOE) III: Test SAS
# SAS_features = [
#     'B_1_last_woe',
#     'B_2_mean_last_6M_woe',
#     'B_3_last_last_sub_first_woe',
#     'B_4_last_last_sub_first_woe',
#     'B_7_mean_last_3M_woe',
#     'D_127_mean_woe',
#     'D_39_last_woe',
#     'D_41_last_last_sub_first_woe',
#     'D_42_min_woe',
#     'D_44_mean_last_6M_woe',
#     'P_2_last_woe',
#     'P_2_mean_woe',
#     'R_1_last_woe',
#     'R_1_mean_last_3M_woe',
#     'R_2_last_woe',
#     'R_3_mean_woe',
#     'S_3_max_woe'
# ]

# X = train_df_binned[SAS_features]
# y = train_df_binned['target']

# # Separamos los datos en entrenamiento y validación
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# # Definimos el modelo
# model = LogisticRegression(random_state=42, max_iter=1000)

# # Entrenamos el modelo
# model.fit(X_train, y_train)

# # Predecimos los datos de validación
# y_pred = model.predict_proba(X_val)[:, 1]

# # Calculamos la métrica
# metric = amex_metric_np(y_pred, y_val.to_numpy())

# # Tabla con cada variable y su coeficiente
# coef_df = pd.DataFrame({'feature': X_train.columns, 'coef': model.coef_[0]})

# print(f'AMEX metric: {metric}')
