# AMEX default prediction XGBoost (agg)

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

# Optimización de hiperparámetros
import optuna


# In[2]: Lectura de datos
oh = True

train_labels, train, test = fe.load_datasets(oh)

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

# train_df_oh_raw = fe.select_model_features(train_df_oh_raw, 0, 'xgb')

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

def xgb_model_func(X_input, y_input, folds, FEAT_IMPORTANCE: bool):

    # Vamos a hacer un stratified k-fold cross validation con 5 folds
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
        X_train, X_valid = X_input.iloc[train_index], X_input.iloc[valid_index]
        y_train, y_valid = y_input.iloc[train_index], y_input.iloc[valid_index]

        # Creamos el dataset de entrenamiento indicando las variables categóricas
        # (Cambiar a DeviceQuantileDMatrix, es mucho más rápido)

        dtrain = xgb.QuantileDMatrix(X_train, label=y_train, feature_names=X_train.columns, nthread=-1, enable_categorical=True)
        dvalid = xgb.DMatrix(X_valid, label=y_valid, feature_names=X_valid.columns, nthread=-1, enable_categorical=True)

        # Medimos el tiempo de entrenamiento de cada fold (start)
        start = time.time()

        # Entrenamos el modelo para el fold actual

        xgb_model = xgb.train(xgb_parms, dtrain, num_boost_round=2500, evals=[(dtrain,'train'),(dvalid,'test')],
                                early_stopping_rounds=50, verbose_eval=50) # feval ver custom metric https://www.kaggle.com/code/jiweiliu/rapids-cudf-feature-engineering-xgb
        
        # Medimos el tiempo de entrenamiento de cada fold (end)
        end = time.time()
        print(f'Tiempo de entrenamiento del fold {fold}:', end-start)
        
        # Guardamos el modelo
        fe.save_model_fe('XGBoost', xgb_model, fold, current_time) 

        # Feature importance para el fold actual
        importances[fold] = xgb_model.get_score(importance_type='weight') # ‘weight’ - the number of times a feature is used to split the data across all trees.

        # Predecimos sobre el conjunto de validación
        y_pred = xgb_model.predict(dvalid)
        
        # Calculamos el score para el fold actual con la métrica customizada
        AMEX_score = amex_metric_mod(y_valid.values, y_pred) # DA ERROR LA ORIGINAL
        print(f'Métrica de Kaggle para el fold {fold}:', AMEX_score)
        scores['AMEX'].append(AMEX_score)

        # Calculamos el permutation feature importance
        if FEAT_IMPORTANCE is True:
            print('-'*50)
            print('Calculando permutation feature importance...')

            # Creamos un diccionario para guardar los valores de la métrica
            perm_scores = {}

            # Creamos un bucle para calcular el valor de la métrica tras predecir habiendo permutado cada variable
            for col in tqdm(X_valid.columns):
                print(f'Calculando permutation feature importance para la variable {col}')
                # Guardamos la variable original
                temp = X_valid.loc[:, col].copy()
                # Permutamos la columna actual
                X_valid.loc[:, col] = np.random.permutation(X_valid[col])
                # Validamos el modelo con la columna permutada
                dvalid = xgb.DMatrix(X_valid, label=y_valid)
                # Predecimos sobre el conjunto de validación
                y_pred = xgb_model.predict(dvalid)
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
            perm_scores_df.to_excel(f'C:/Users/Jose/Documents/UNIVERSIDAD/TFG/MATEMATICAS/PYTHON/MODELOS/XGBoost_{current_time}/permutation_feature_importance_{fold}.xlsx', index=True)

            # Plot Permutation Feature Importance: Top 100
            plt.figure(figsize=(10, 30))
            sns.barplot(x='score_diff', y='index', data=perm_scores_df[:100])
            plt.title('XGB Permutation Feature Importance: Top 100')
            plt.savefig(f'C:/Users/Jose/Documents/UNIVERSIDAD/TFG/MATEMATICAS/PYTHON/MODELOS/XGBoost_{current_time}/permutation_feature_importance_{fold}.png')
            plt.show()

        # Liberamos memoria
        del X_train, X_valid, y_train, y_valid, dtrain, dvalid
        gc.collect()

    # Mostramos los resultados
    print('-'*50)
    print('Valor medio de la métrica de Kaggle para todos los folds:', np.mean(scores['AMEX']))

xgb_model_func(X, y, 5, False)


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


# In[13]: Feature importance I:

def fi_func(importances, current_time, X_input):
    # Importancia media de cada variable
    importances_df = pd.DataFrame(importances).T
    mean_importances = importances_df.mean(axis=0).sort_values(ascending=False)

    # Plot top 150 variables
    plt.figure(figsize=(10, 30))
    sns.barplot(x=mean_importances.values[:150], y=mean_importances.index[:150])
    plt.title('Feature Importances over {} folds (top 150)'.format(len(importances)))
    plt.savefig(f'C:/Users/Jose/Documents/UNIVERSIDAD/TFG/MATEMATICAS/PYTHON/MODELOS/XGBoost_{current_time}/feature_importance_150.png')
    plt.show()

    # Plot top 50 variables
    plt.figure(figsize=(10, 30))
    sns.barplot(x=mean_importances.values[:50], y=mean_importances.index[:50])
    plt.title('Feature Importances over {} folds (top 50)'.format(len(importances)))
    plt.savefig(f'C:/Users/Jose/Documents/UNIVERSIDAD/TFG/MATEMATICAS/PYTHON/MODELOS/XGBoost_{current_time}/feature_importance_50.png')
    plt.show()

    # Variables con importancia 0
    zero_importance_features = [feature for feature in X_input.columns if feature not in mean_importances.index]
    print('There are {} features with 0 importance'.format(len(zero_importance_features)))

    # Añadimos la media al dataframe
    importances_df = importances_df.T
    importances_df['mean_importance'] = mean_importances

    # Añadimos las variables con importancia 0 al dataframe (rellenamos con NaN para consistencia)
    importances_df = importances_df.append(pd.DataFrame(index=zero_importance_features))
    importances_df['mean_importance'].fillna(0, inplace=True)

    # Guardamos el dataframe en un excel
    importances_df.to_excel(f'C:/Users/Jose/Documents/UNIVERSIDAD/TFG/MATEMATICAS/PYTHON/MODELOS/XGBoost_{current_time}/feature_importance.xlsx', index=True)

    return importances_df, zero_importance_features

importances_df, zero_importance_features = fi_func(importances, current_time, X)


# In[14]: Remodelado eliminando las variables con importancia 0

# Diccionario para guardar las importancias de cada fold
importances = {} 
# Diccionario para guardar los scores de cada fold
scores = {'AMEX': []} 
# Necesitamos el tiempo para generar la carpeta donde guardar los modelos
current_time = time.strftime('%Y%m%d_%H%M%S')

# Eliminamos las variables con importancia 0 de X
X_0_out = X.drop(columns=zero_importance_features)

# Hacemos de nuevo un stratified k-fold cross validation con 5 folds bajo la misma semilla para comparar resultados
xgb_model_func(X_0_out, y, 5)

importances_df, zero_importance_features = fi_func(importances, current_time, X_0_out)


# In[15]: Test predictions (pruebas) --> Agregar modelos para hacer el ensemble y evitar overfitting

# Función para recorrer los distintos modelos y hacer predicciones sobre test
def test_predictions(model_name, threshold, test_df, nfolds=5):
    # Cargamos datos de test si no están cargados
    if test_df is None:
        test_df = pd.read_parquet('C:/Users/Jose/Documents/UNIVERSIDAD/TFG/MATEMATICAS/PYTHON/DATASETS/combined_dataset/test_df_oh.parquet')
        print('Test data loaded')

    # Seleccionamos las variables del modelo
    if threshold is not None:
        test_df = fe.select_model_features(test_df, threshold, 'xgb')
        print(f'Test data features selected based on PIMP. Test data shape is now: {test_df.shape}')

    # Iteramos sobre cada fold para calucular las predicciones de cada modelo
    for fold in range(nfolds):
        xgb_model = xgb.Booster()
        xgb_model.load_model(f'C:/Users/Jose/Documents/UNIVERSIDAD/TFG/MATEMATICAS/PYTHON/MODELOS/XGBoost_{model_name}/XGBoost_model_{fold}.json')
        print(f'Model for fold {fold} loaded')
        # Predecimos sobre test
        X_test = test_df.drop(columns=['customer_ID'])
        dtest = xgb.DMatrix(X_test, feature_names=X_test.columns, nthread=-1, enable_categorical=True)
        y_pred_test = xgb_model.predict(dtest)
        print(f'Prediction for fold {fold} done')
        # Creamos un dataframe con las predicciones
        submission = pd.DataFrame({'customer_ID': test_df['customer_ID'], 'prediction': y_pred_test})
        # Guardamos el dataframe en un csv
        submission.to_csv(f'C:/Users/Jose/Documents/UNIVERSIDAD/TFG/MATEMATICAS/PYTHON/MODELOS/XGBoost_{model_name}/submission_{model_name}_{fold}.csv', index=False)
        print(f'Submission for fold {fold} done')
        # Liberamos memoria
        del xgb_model, X_test, dtest, y_pred_test, submission
        gc.collect()

# test_predictions('20230531_190457', None, None)
test_predictions('20230610_004736', 0, None)


# In[16]: Optimización de hiperparámetros con Optuna

# Diccionario para guardar los scores de cada fold
scores = {'AMEX': []} 

# Función para optimizar los hiperparámetros de XGBoost con Optuna
def objective(trial):
    # Definimos los hiperparámetros a optimizar
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.5),
        'subsample': trial.suggest_discrete_uniform('subsample', 0.6, 1, 0.1),
        'colsample_bytree': trial.suggest_discrete_uniform('colsample_bytree', 0.6, 1, 0.1),
        'eval_metric':'logloss',
        'objective':'binary:logistic',
        'tree_method':'gpu_hist',
        'predictor':'gpu_predictor',
        'random_state':42 # default n_threads = -1 (max available)
    }

    # Vamos a hacer un stratified k-fold cross validation con 5 folds
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    split = skf.split(X, y)

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
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=X_train.columns, nthread=-1, enable_categorical=True)
        dvalid = xgb.DMatrix(X_valid, label=y_valid, feature_names=X_valid.columns, nthread=-1, enable_categorical=True)

        # Entrenamos el modelo para el fold actual
        xgb_model = xgb.train(params, dtrain, num_boost_round=2500, evals=[(dtrain,'train'),(dvalid,'test')],
                                early_stopping_rounds=50, verbose_eval=50)
        
        # Predecimos sobre el conjunto de validación
        y_pred = xgb_model.predict(dvalid)

        # Calculamos el score para el fold actual con la métrica customizada
        AMEX_score = amex_metric_mod(y_valid.values, y_pred)
        print(f'Métrica de Kaggle para el fold {fold}:', AMEX_score)
        scores['AMEX'].append(AMEX_score)

        # Liberamos memoria
        del X_train, X_valid, y_train, y_valid, dtrain, dvalid
        gc.collect()

    # Mostramos los resultados
    print('-'*50)
    print('Valor medio de la métrica de Kaggle para todos los folds:', np.mean(scores['AMEX']))

    return np.mean(scores['AMEX'])

# Optimizamos los hiperparámetros
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50, show_progress_bar=True, gc_after_trial=True)

# Mostramos los resultados
print('Valor óptimo de la métrica:', study.best_value)
print('Mejores hiperparámetros:', study.best_params)

# Guardamos los resultados en un excel
results = study.trials_dataframe()
results.to_excel('C:/Users/Jose/Documents/UNIVERSIDAD/TFG/MATEMATICAS/PYTHON/MODELOS/XGBoost_optuna/results.xlsx', index=True)

