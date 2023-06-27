import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gc,os,random
import time,datetime
from tqdm import tqdm

# Ensemble de las mejores predicciones
p0 = pd.read_csv('./MODELOS/LGBM_20230620_111308/submission_20230620_111308_0.csv')
p1 = pd.read_csv('./MODELOS/XGBoost_20230610_004736/submission_20230610_004736_4.csv')
p2 = pd.read_csv('./MODELOS/LogReg_20230620_224831/LogReg_lastobs_20230620_224831_3.csv')

p0['prediction'] = p0['prediction']*0.4 + p1['prediction']*0.4 + p2['prediction']*0.2

p0.to_csv('submission_ensemble.csv', index=False)