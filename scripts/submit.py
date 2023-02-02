import pandas as pd
from datetime import datetime as dt

def get_submitable(test_lst, model_lst, train_cols):
	for test, model in zip(test_lst, model_lst):
		X_test = test[train_cols]
		y_test = model.predict(X_test)
		test['total_cases'] = y_test.astype(int)
	tosubmit = pd.concat(test_lst)[['city','year','weekofyear','total_cases']] 
	file_name = dt.now().strftime('%Y%m%d%H%M') + '.csv'
	tosubmit.to_csv(file_name, index=False)