from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

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