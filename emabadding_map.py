import numpy as np
from sklearn.preprocessing import LabelEncoder

#load the embedding data
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

#for getting summation of each face classes
for (X_prime,y_prime) in train_set:
    X_emds[y_prime] = np.add(X_emds[y_prime],X_prime)

#X_emb is summation of same label faces embadding

#counting the occurance of each classes in the training set
unique, counts = np.unique(int_label, return_counts=True)
dic = dict(zip(unique,counts))
#------------------------------------------------------------#
h=0
final_mean_embedding = []
for X_emd in X_emds:
    a = np.divide(X_emd,dic[h])
    final_mean_embedding.append(a)
    h+=1

labels = list(set(int_label))
#Save the mean embeddings to the file "mean_embeddings.npz"    
np.savez_compressed("mean_embeddings.npz",final_mean_embedding,labels)