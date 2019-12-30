import os
import csv
import math
import numpy as np
print("Detection de spam dans les reseaux sociaux")
import pandas 
df = pandas.read_csv('Echantillon2.csv')
df = pandas.read_csv('Echantillon2.csv', sep = '\t', header = None)
#print(df.shape)
#print(df.head())
us= pandas.read_csv('userdata.csv')
us= pandas.read_csv('userdata.csv', sep = '\t', header = None)
#print(us.shape)
#print(us.head())
us = us.replace("F", 1)
us = us.replace("M", 0)
u=np.array(us,dtype=object)
#-------------------------------------------------------------------------------
# function pour avoir les valeurs uniques 
def unik(list1): 
    x = np.array(list1) 
    #print(np.unique(x)) 
    return np.unique(x)
#--------------------------------------------------------------------------------
#mise en forme des données n  nombre de relations dans l'échantillon	
x=np.array(df,int)
(n,m)=x.shape
print("Number of Relations =")
print((n,m))
#--------------------------------------------------------------------------------
#extraction des utilisateurs source dans l'échantillon
#print(x[:,2])
Yu= unik(x[:,2])
print("===Liste des utilisateurs source===")
#print(Yu)
#N nombre d'utilisateurs source dans l'échantillon
N=Yu.shape[0]
print("Nombre d'utilisateurs sources  de l'échantillon =" )
print(N)

#---------------------------------------------------------------------------------
#calcule du vecteur X caracteristiques de l'utilisateeur
def formdata(x,Yu,u):
    X = np.zeros(shape=(N,7))
    Y=np.zeros(shape=(N,1))
    TY=np.zeros(shape=(N,1))
      
    for k in range(N):
      y=Yu[k]
      ty=0
      for i in range(n):                           
          if x[i,2]==y:
               ty=ty+1
               for j in [1,2,3,4,5,6]:
                  if x[i,4]==j:
                     X[k,j]=X[k,j]+1    
               
      TY[k,0]=ty
      for z in [1,2,3,4,5,6]:         
        X[k,z]=X[k,z]/TY[k,0]
      X[k,0]=u[k,2]

    return X
X=formdata(x,Yu,u)
#-------------------------------------------------------------------------------
#classification réel
userFichier=open('Ye2.csv','r')
usercsv=csv.reader(userFichier,delimiter=',')
U = list(usercsv)
#print(U)
Y=np.array(U,float)
Y=Y.ravel()  

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
plt.title("Clusters formés par K-Means après ACP (Echantillon 2)")
plt.show()
