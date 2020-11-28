import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
data = np.load("data.npz")

y_train = data['arr_1']
y_test = data['arr_3']

X_train = data['arr_0']
X_test =  data['arr_2']

label_encoder = LabelEncoder()
y_train_int = label_encoder.fit_transform(y_train)
y_test_int = label_encoder.fit_transform(y_test)

logistic = LogisticRegression(random_state=0)

logistic.fit(X_train, y_train_int)

y_pred = logistic.predict(X_test)

print('Real Values',y_test_int)
print('Predicted Values', y_pred)
print("accuracy : ", accuracy_score(y_test_int,y_pred)*100 ,"%")