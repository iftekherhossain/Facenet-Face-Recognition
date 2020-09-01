import numpy as np
from sklearn.svm import SVC
import sklearn as sk
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
data = np.load("data.npz")

y_train = data['arr_1']
y_test = data['arr_3']

X_train = data['arr_0']
X_test =  data['arr_2']
print("kuku",X_test[-1])
label_encoder = LabelEncoder()
y_train_int = label_encoder.fit_transform(y_train)
y_test_int = label_encoder.fit_transform(y_test)
print("uloi",X_test.shape)
clf = make_pipeline(sk.svm.LinearSVC(C=0.1))
clf.fit(X_train,y_train_int)
y_pred = clf.predict(X_test)

print('Real Values',y_test_int)
print('Predicted Values', y_pred)
print("accuracy : ", accuracy_score(y_test_int,y_pred)*100 ,"%")

print("New")
a = X_test[-1]
a = np.reshape(a,(1,128))
print(a.shape)
print(clf.predict(a))