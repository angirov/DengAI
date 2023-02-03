import numpy as np
import pandas as pd
from datetime import datetime as dt

# https://www.kaggle.com/code/avanwyk/encoding-cyclical-features-for-deep-learning
def encode(data, col, max_val):
	data[col + '_sin'] = np.sin(2 * np.pi * data[col]/max_val)
	data[col + '_cos'] = np.cos(2 * np.pi * data[col]/max_val)
	return data

def load_data():
	# Loading
	df = pd.read_csv('data/dengue_features_train.csv')
	target = pd.read_csv('data/dengue_labels_train.csv')
	test_df = pd.read_csv('data/dengue_features_test.csv')
	# Merging features and labels
	train_df = pd.merge(df, target,  how='inner', left_on=['city', 'year', 'weekofyear'], right_on = ['city', 'year', 'weekofyear'])
	# Transforming to datetime and sorting 
	train_df['date'] = pd.to_datetime(train_df['week_start_date'])
	test_df['date']  = pd.to_datetime(test_df['week_start_date'])
	train_df.sort_values(by= ['city', 'date'], ascending=[False, True])
	test_df.sort_values(by= ['city', 'date'], ascending=[False, True])
	# Missing values
	train_df = train_df.bfill()
	test_df = test_df.bfill()
	# Cycic encoding
	train_df = encode(train_df, 'weekofyear', 51)
	test_df = encode(test_df, 'weekofyear', 51)
	# Spliting by cities
	train_sj = train_df[train_df['city'] == 'sj']
	train_iq = train_df[train_df['city'] == 'iq']
	test_sj  = test_df[test_df['city'] == 'sj']
	test_iq  = test_df[test_df['city'] == 'iq']

	return train_sj, train_iq, test_sj, test_iq

def split_df(df, test_size):
	N = int(len(df) * (1 - test_size))
	train = df[: N]
	val  = df[N:]
	print('Train size: {}, validation size: {}'.format(len(train), len(val)))
	return train, val

def get_xxyy(df, test_size, train_cols, target_col):
	train, val = split_df(df, test_size)
	return dict(
			X_train = train[train_cols],
			X_val = val[train_cols],
			y_train = train[target_col],
			y_val = val[target_col]
		)
