data = pd.read_csv("data/ai4i2020.csv")
data = data.drop(columns=['Product ID', 'UDI', 'Type'])
data.rename(columns={'Air temperature [K]': 'air_temperature', 'Process temperature [K]': 'process_temperature', 'Rotational speed [rpm]':'rotational_speed', 'Torque [Nm]': 'torque',                            	'Tool wear [min]': 'tool_wear'}, inplace = True)
data.drop(['TWF', 'HDF', 'PWF', 'OSF', 'RNF'], axis = 1, inplace = True)
X= data.drop("Machine failure",axis=1) 
Y= data["Machine failure"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
model = LogisticRegression()
model.fit(X_train, Y_train)


y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")