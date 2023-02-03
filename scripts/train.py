from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV

def train_rfr(	X_train,
				y_train,
				n_estimators=1000):
	rfr = RandomForestRegressor(n_estimators)
	rfr.fit(X_train, y_train)
	return rfr

def predict_val(model, X_val, y_val):
	y_pred = model.predict(X_val)
	mae = mean_absolute_error(y_val, y_pred)
	print(mae)
	return y_pred, mae

def train_grid(xxyy_dict, estimator, params, scoring, n_jobs=6, refit=True, ret=True):
	grid = GridSearchCV(
		estimator=estimator,
		param_grid=params,
		scoring=scoring,
		n_jobs=n_jobs,
		refit=refit,
		cv=None,
		return_train_score=ret
	)
	return grid.fit(xxyy_dict['X_train'], xxyy_dict['y_train'])
