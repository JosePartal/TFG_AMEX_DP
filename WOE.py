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

# Feature engineering functions
import feature_engineering as fe

# Import optibinning
import optbinning

# %%
