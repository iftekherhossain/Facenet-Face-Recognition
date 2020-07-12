import numpy as np
from sklearn.preprocessing import LabelEncoder
data = np.load("D:\\Facenet-Face_recognition\\data.npz")

X = data['arr_0']
y= data['arr_1']

lb = LabelEncoder()

#int_label is integer form of labels
int_label = lb.fit_transform(y)

#concating two embaddings and labels
train_set = tuple(zip(X,int_label))

#initializing a list of vectors
X_emds = [np.zeros(128,) for i in range(6)]
print(len(X_emds))
#for getting summation of each face classes
for (X_prime,y_prime) in train_set:
    X_emds[y_prime] = np.add(X_emds[y_prime],X_prime)

#X_emb is summation of same label faces embadding
#print(X_emds[0])

#counting the occurance of each classes in the training set

unique, counts = np.unique(int_label, return_counts=True)
dic = dict(zip(unique,counts))

h=0
final_mean_embedding = []
for X_emd in X_emds:
    a = np.divide(X_emd,dic[h])
    final_mean_embedding.append(a)
    h+=1
    
#print(final_mean_embedding[0])

X_test_embeddings = data['arr_2']
y_test_labels = data['arr_3']
pic=25
i = 5
quki = list(set(y))
#print("testing on ",y_test_labels[pic])
#print("distance from the image",quki[i])
#temp = np.linalg.norm(X_test_embeddings[pic]-final_mean_embedding[i])

#print(temp)

for w in X_test_embeddings:
    temp = np.linalg.norm(w-final_mean_embedding[i])
    print(temp)