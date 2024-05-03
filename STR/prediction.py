import joblib
def predict(data):
	clf = joblib.load("model.sav")
	return clf.predict(data)