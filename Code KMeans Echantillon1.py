import os
import csv
import math
import numpy as np
print("Detection de spam dans les reseaux sociaux")
import pandas 
df = pandas.read_csv('Echantillon1.csv')
df = pandas.read_csv('Echantillon1.csv', sep = ';')
(N,m)=df.shape
#print(df.shape)
#print(df.head())
df=np.array(df,dtype=object)
Ye= pandas.read_csv('Ye1.csv')
Ye= pandas.read_csv('Ye1.csv', sep = '\t', header = None)
#print(Y.shape)
#print(Y.head())
Y = Ye.replace(-1, 0)
Y=np.array(Y,dtype=int)
Y=Y.ravel()
#print(Y)

def formdata(df):
    X = np.zeros(shape=(N,5))
       
    for k in range(N):
      X[k,0]=df[k,1]
      X[k,1]=df[k,2]
      X[k,2]=df[k,3]
      X[k,3]=df[k,4]
      X[k,4]=df[k,4]/df[k,3]

    return X
X=formdata(df)

from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.metrics import roc_curve, auc,silhouette_samples

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
# Data Normalization

from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(X)
from sklearn.decomposition import PCA as sklearnPCA
sklearn_pca = sklearnPCA(n_components = 2)
X_PCA= sklearn_pca.fit_transform(X_std)
#plt.scatter(X_PCA[:,0],X_PCA[:,1], label='True Position')
#plt.show()
kmeans = KMeans(n_clusters=2, init='k-means++',
    n_init=10, max_iter=400, 
    tol=1e-04, random_state=0)
kmeans.fit(X_PCA)
ycluster = kmeans.fit_predict(X_PCA)
#print(ycluster)

print(classification_report(Y, ycluster))
print(confusion_matrix(Y, ycluster))
print("Accuracy:",accuracy_score(Y, ycluster))
auc= metrics.roc_auc_score(Y, ycluster)
print("AUC:" ,auc)
 # Get silhouette samples
#print( metrics.silhouette_samples(Y, ycluster))

print("complit")
print("adjusted_rand_score:",metrics.adjusted_rand_score(Y, ycluster))
print("silhouette_score:",metrics.silhouette_score(X, ycluster,
                                      metric='euclidean',
                                      sample_size=200))

print("completeteness:",metrics.completeness_score(Y, ycluster))
print("homogeneity:", metrics.homogeneity_score(Y, ycluster))
print("v_measure:",metrics.v_measure_score(Y, ycluster))
#print(kmeans.cluster_centers_)
#print(kmeans.labels_)
plt.scatter(X_PCA[:,0],X_PCA[:,1], c=kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:,0] ,kmeans.cluster_centers_[:,1], color='black')
plt.legend()
plt.ylabel('x2')
plt.xlabel('x1')
for i in range(0, X_PCA.shape[0]):
    if kmeans.labels_[i] == 1:
       c1 = plt.scatter(X_PCA[i,0],X_PCA[i,1],c='r',
    marker='o')
    elif kmeans.labels_[i] == 0:
       c2 = plt.scatter(X_PCA[i,0],X_PCA[i,1],c='y',
    marker='o')
plt.legend([c1, c2],['Spams', 'Non-Spams'
    ])       
plt.title("Clusters formés par K-Means après ACP (Echantillon 1)")
plt.show()
