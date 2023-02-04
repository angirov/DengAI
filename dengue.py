# %%
import numpy as np
import pandas as pd
from datetime import datetime as dt
import plotly.express as px

# %%
from scripts.preproc import load_data, split_df, get_xxyy
from scripts.train import train_grid, predict_val
from scripts.submit import get_submitable
train_cols = ['precipitation_amt_mm', 'weekofyear_sin', 'weekofyear_cos']
target_col = 'total_cases'
test_size = 0

# %% [markdown]
# # 1.Preprocessing
# 
# ## 1.1 Train features and labes are merged, both train and test DFs are devided by cities

# %%
train_sj, train_iq, test_sj, test_iq = load_data()

# %% [markdown]
# ## 1.2. Visualisation

# %%
# fig = px.line(train_sj, x='date', y=["precipitation_amt_mm", 'station_avg_temp_c', 'total_cases', 'weekofyear_sin', 'weekofyear_cos'], title='San Juan')
# fig.show()
# fig = px.line(train_iq, x='date', y=["precipitation_amt_mm", 'station_avg_temp_c', 'total_cases', 'weekofyear_sin', 'weekofyear_cos'], title='Iquitos')
# fig.show()

# %% [markdown]
# ## 1.3. Spliting into training and validation sets
# 
# I suggest that at the end of preprocessing for either city we get **one** dictionary with **four** dataframes: X_train, X_val, y_train, y_val

# %%
xxyy_sj = get_xxyy(train_sj, test_size, train_cols, target_col)
xxyy_iq = get_xxyy(train_iq, test_size, train_cols, target_col)

# %% [markdown]
# # 2. Training

# %% [markdown]
# At the end of training we need:
# 1. a model compatible with scikit-learn api
# 2. (maybe???) an encoder

# %%
from sklearn.model_selection import TimeSeriesSplit

cv = TimeSeriesSplit(5)
scoring = 'neg_mean_absolute_error'

# %%
from sklearn.ensemble import RandomForestRegressor
estimator = RandomForestRegressor(verbose=True)
params = dict(
	n_estimators = [3000, 2500, 2000],
	max_depth = [3, 4, 5],
	# eta = [0.3, 0.1, 0.01],
	# subsample = [0.3, 0.5, 0.8, 1],
	# colsample_bytree = 1,
)

# %%
model_sj = train_grid(xxyy_sj, estimator, params, scoring)
model_iq = train_grid(xxyy_iq, estimator, params, scoring)

# %%
pd.DataFrame(model_sj.cv_results_).columns

# %%
print('San Jose')
print(pd.DataFrame(model_sj.cv_results_)
	.sort_values(by='rank_test_score')[[
		'param_max_depth', 'param_n_estimators', 'mean_train_score']])
print()
print('Iquitos')
print(pd.DataFrame(model_iq.cv_results_)
	.sort_values(by='rank_test_score')[[
		'param_max_depth', 'param_n_estimators', 'mean_train_score']])

# %% [markdown]
# # 3. Predict for test features and format for submition
# 
# For that we need:
# 1. [from precrocessing] - (list of) test DFs
# 2. [from training]      - (list of) models

# %%
get_submitable([test_sj, test_iq], [model_sj, model_iq], train_cols)


