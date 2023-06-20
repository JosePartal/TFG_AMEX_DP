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
# import cuml

# Feature engineering functions
import feature_engineering as fe

# Binning and WOE
import optbinning

# Progress bar
from tqdm import tqdm

# Save models
import pickle


# In[2]: Lectura de datos

# Train
train = pd.read_parquet('C:/Users/Jose/Documents/UNIVERSIDAD/TFG/amex-default-prediction/parquet_ds_integer_dtypes/train.parquet')
# Labels
train_labels = pd.read_csv('C:/Users/Jose/Documents/UNIVERSIDAD/TFG/amex-default-prediction/train_labels.csv', low_memory=False)


# In[3]: Variables categóricas

# Lista de variables categóricas
cat_features = ['B_30', 'B_38', 'D_63', 'D_64', 'D_66', 'D_68', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126']

num_features = [col for col in train.columns if col not in cat_features and col != 'customer_ID']

# Lista de variables (exlucyendo 'customer_ID)
features = list(train.columns)
features.remove('customer_ID')
features.remove('S_2')


# In[4]: Métrica

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

train_df = train.groupby('customer_ID').tail(1).set_index('customer_ID') # Última observación
train_df = train_df.merge(train_labels, left_on='customer_ID', right_on='customer_ID')
del train, train_labels#, test
gc.collect()


# In[6]: Binning y WOE

# Vamos a usar la librería optbinning para hacer el binning y el WOE. Usaremos la función BinningProcess

# Creamos el objeto para el binning con special_codes identificando los NaN (special_codes must be a dit, list or numpy.ndarray)
binning_fit = optbinning.BinningProcess(variable_names=features, categorical_variables= cat_features, special_codes=[np.nan], n_jobs=-1)

# # Save binning process --> ¿Debería ir después del fit?
# binning_fit.save('binning_process.pkl')

# Load binning process
# binning_fit = optbinning.BinningProcess.load('DATASETS/binning_process.pkl')

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


# In[10]: Regresión logística usando variables Weight of Evidence (WOE) I: Separación de datos

# Vamos a calcular la regresión logística usando las variables WoE. No emplearemos las variables con IV <= 0.02

# En primer lugar, definimos X e y. Tenemos que añadir primero '_woe' a las variables seleccionadas
selected_features = [col + '_woe' for col in selected_features]
X = train_df_binned[selected_features]
y = train_df_binned['target']

del binning_fit, binning_table, optb, train_df_binned

# In[11]: Regresión logística usando variables Weight of Evidence (WOE) II: Cross-validation 

"""Lo tengo para una regresión L2 (Ridge) ahora mismo, si quiero una L1 (Lasso) tengo que cambiar el penalty y el solver a 'saga'."""

# Creamos un diccionario para guardar los scores de cada fold
scores = {'AMEX': []} 
# Creamos un diccionario para guardar los coeficientes de cada variable para cada fold
coefficients = {}

# Necesitamos el tiempo para generar la carpeta donde guardar los modelos
current_time = time.strftime('%Y%m%d_%H%M%S')

# Definimos una función para calcular la regresión logística usando CV
def logistic_regression_func(X_input, y_input, folds, current_time, intercept: bool, hyperopt: bool, max_iter: int, verbose, solver):
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
        print('Datos separados en entrenamiento y validación')

        if hyperopt is True:
            print(f'Optimización de hiperparámetros para el fold {fold+1}')
            # Optimización de hiperparámetros con optuna: C
            def objective(trial):
                # Definimos los hiperparámetros
                C = trial.suggest_loguniform('C', 1e-10, 1e10)

                # Definimos el modelo
                model = LogisticRegression(random_state=42, penalty='l2', max_iter=max_iter, fit_intercept= intercept, 
                                        C=C, solver=solver, verbose=verbose)

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
            log_model = LogisticRegression(random_state=42, penalty='l2', max_iter=max_iter, fit_intercept = intercept,
                                        C=best_params['C'], solver=solver,verbose=verbose)
            
        else:
            # Definimos el modelo
            print('No se ha realizado la optimización de hiperparámetros')
            log_model = LogisticRegression(random_state=42, penalty='l2', max_iter=max_iter, fit_intercept = intercept, 
                                           solver=solver, verbose=verbose)
            
        # Medimos el tiempo de entrenamiento de cada fold (start)
        start = time.time()

        # Entrenamos el modelo
        print('Entrenamiento del modelo')
        log_model.fit(X_train, y_train)
        print('Entrenamiento finalizado')

        # Medimos el tiempo de entrenamiento de cada fold (end)
        end = time.time()
        print('Tiempo de entrenamiento:', end - start)

        # Predecimos los datos de validación
        print('Predicción de los datos de validación')
        y_pred = log_model.predict_proba(X_val)[:, 1]
        print('Predicción finalizada')

        # Guardamos el modelo
        fe.save_model_fe('LogReg', log_model, fold, current_time)

        # Calculamos la métrica
        AMEX_score = amex_metric_np(y_pred, y_val.to_numpy())
        print(f'Métrica de Kaggle para el fold {fold}:', AMEX_score)
        scores['AMEX'].append(AMEX_score)

        # Tabla con cada variable y su coeficiente del fold actual
        coefficients[fold] = pd.DataFrame({'feature': X_train.columns, 'coef': log_model.coef_[0]})
        print('Tabla con los coeficientes de cada variable del fold creada')

        # Liberamos memoria
        del X_train, X_val, y_train, y_val, log_model, y_pred
        gc.collect()

    # Mostramos los resultados
    print('-'*50)
    print('Valor medio de la métrica de Kaggle para todos los folds:', np.mean(scores['AMEX']))

logistic_regression_func(X, y, 5, current_time, intercept=True, hyperopt=False, max_iter=2000, verbose=1, solver='lbfgs') # 'lbfgs'


# In[12]: Regresión logística usando variables Weight of Evidence (WOE) III: Test prediction

def test_predictions(model_name, nfolds=5):

    # Cargamos los datos de test
    test = pd.read_parquet('C:/Users/Jose/Documents/UNIVERSIDAD/TFG/amex-default-prediction/parquet_ds_integer_dtypes/test.parquet')
    test = test.groupby('customer_ID').tail(1).set_index('customer_ID') # Última observación
    test = test.reset_index()

    # Binning y WOE
    # Hacemos el binning de los datos de test
    test_binned = binning_fit.transform(test[features])

    # Renombramos las variables
    test_binned.columns = [col + '_woe' for col in test_binned.columns]

    # Creamos un bucle para hacer las predicciones de cada fold
    for fold in range(nfolds):
        # Cargamos el modelo
        logreg_model = pickle.load(open(f'MODELOS/LogReg_{model_name}/LogReg_model_{fold}.pkl', 'rb'))
        print(f'Modelo {fold} cargado')

        # Hacemos logreg_model predicciones
        y_pred = logreg_model.predict_proba(test_binned)[:, 1]
        print(f'Predicciones del modelo {fold} realizadas')

        # Creamos un dataframe con las predicciones
        submission = pd.DataFrame({'customer_ID': test['customer_ID'], 'prediction': y_pred})

        # Guardamos el dataframe en un csv
        submission.to_csv(f'MODELOS/LogReg_{model_name}/LogReg_lastobs_{model_name}_{fold}.csv', index=False)
        print(f'Predicciones del modelo {fold} guardadas')

        # Liberamos memoria
        del logreg_model, y_pred, submission

test_predictions('20230620_172139', nfolds=5)

# %%
