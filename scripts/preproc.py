import numpy as np
import pandas as pd
from datetime import datetime as dt

def load_data():
	df = pd.read_csv('data/dengue_features_train.csv')
	target = pd.read_csv('data/dengue_labels_train.csv')
	test_df = pd.read_csv('data/dengue_features_test.csv')
	full_df = pd.merge(df, target,  how='inner', left_on=['city', 'year', 'weekofyear'], right_on = ['city', 'year', 'weekofyear'])
	full_df['date'] = pd.to_datetime(full_df['week_start_date'])
	test_df['date'] = pd.to_datetime(test_df['week_start_date'])
	full_df.sort_values(by= ['city', 'date'], ascending=[False, True])
	test_df.sort_values(by= ['city', 'date'], ascending=[False, True])
	full_df = full_df.bfill()
	test_df = test_df.bfill()
	full_sj = full_df[full_df['city'] == 'sj']
	full_iq = full_df[full_df['city'] == 'iq']
	test_sj = test_df[test_df['city'] == 'sj']
	test_iq = test_df[test_df['city'] == 'iq']

	return full_sj, full_iq, test_sj, test_iq

def split_df(full_df, test_size):
	N = int(len(full_df) * (1 - test_size))
	train = full_df[: N]
	val  = full_df[N:]
	print('Train size: {}, validation size: {}'.format(len(train), len(val)))
	return train, val

def get_xy(df, test_size, train_cols, target_col):
	train, val = split_df(df, test_size)
	X_train, X_val, y_train, y_val =	train[train_cols], \
										val[train_cols], \
										train[target_col], \
										val[target_col]
	return X_train, X_val, y_train, y_val