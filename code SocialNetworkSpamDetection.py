import os
import csv
import math
import numpy as np
print("Detection de spam dans les reseaux sociaux")
import pandas 
df = pandas.read_csv('Echantillon.csv')
df = pandas.read_csv('Echantillon.csv', sep = '\t', header = None)
#print(df.shape)
#print(df.head())
#us= pandas.read_csv('usersdata.csv')
#us= pandas.read_csv('usersdata.csv', sep = '\t', header = None)
#print(us.shape)
#print(us.head())
#u=np.array(us,dtype=object)
#-------------------------------------------------------------------------------
# function pour avoir les valeurs uniques 
def unik(list1): 
    x = np.array(list1) 
   # print(np.unique(x)) 
    return np.unique(x)
#--------------------------------------------------------------------------------
#mise en forme des données n  nombre de relations dans l'échantillon	
x=np.array(df,int)
(n,m)=x.shape
print("Number of Relations =")
print((n,m))
#--------------------------------------------------------------------------------
#extration des utilisateurs source dans l'échantillon
#print(x[:,2])
Yu= unik(x[:,2])
print("===Liste des utilisateurs source===")
#print(Yu)
#N nombre d'utilisateurs source dans l'échantillon
N=Yu.shape[0]
print("Nombre d'utilisateurs sources  de l'échantillon =" )
print(N)

#------------------------------------------------------------------------------
#Extraction des données de userdata
#X=np.zeros(shape=(N,1))
#Y=np.zeros(shape=(N,1))
#for i in range(N):
# b=Yu[i]
# Y[i]=u[b,4]
# X[i]=u[b,2]
# print(i+1,b,u[b,4])
#with open('Ye.csv', 'w', newline='') as f: 
   # writer = csv.writer(f)
   # writer.writerows(Y)
#with open('Xtimes.csv', 'w', newline='') as g:
  # writer = csv.writer(g)
  # writer.writerows(X)
#-------------------------------------------------------------------------------
#composante  timePassedValidation du vecteur  caracteristique
V= pandas.read_csv('Xtimes.csv')
V= pandas.read_csv('Xtimes.csv', sep = '\t', header = None)
#print(V)
Xt=np.array(V,dtype=object)
#---------------------------------------------------------------------------------
#calcule du vecteur X caracteristiques de l'utilisateeur
#pour l'utilisateur k ,nk=TY[k,0],r1k=X[k,0],r2k=X[k,1],tk=X[k,2],utk=X[k,3]
def formdata(x,Yu,Xt):
  X = np.zeros(shape=(N,4))
  TY=np.zeros(shape=(N,1))
  #input_string= input("Entrer les elements de la liste R1 separé de l'espace ")
  #R1 = input_string.split()
  R1=[3,4,5]
 # input_string= input("Entrer les elements de la liste R2 separé de l'espace ")
  #R2 = input_string.split()
  R2=[1]
  for k in range(N):
      y=Yu[k]
      ty=0
      Td=0
      Te=0
      Tf=0
      for i in range(n):                           
          if x[i,2]==y:
               ty=ty+1
               if ty==1 :
                Td=x[i,1]
                Dd=x[i,0]
               elif ty!=1 and x[i,0]!=Dd:
                 Te=Te+x[i,1]-Td
                 Td=x[i,1]
               if x[i,4]in R1:
                 X[k,0]=X[k,0]+1
               elif x[i,4] in R2:
                 X[k,1]=X[k,1]+1
          elif x[i,2]!=y and x[i-1,2]==y  :
               Tf=x[i,1]
  TY[k,0]=ty     
  X[k,0]=X[k,0]/TY[k,0]
  X[k,1]=X[k,1]/TY[k,0]
  X[k,2]=Tf-Td+Te
  X[k,2]=X[k,2]/TY[k,0]
  X[k,3]=Xt[k]

  return X
X=formdata(x,Yu,Xt)
#-------------------------------------------------------------------------------
#classification réel
userFichier=open('Ye.csv','r')
usercsv=csv.reader(userFichier,delimiter=',')
U = list(usercsv)
#print(U)
Y=np.array(U,float)
Y=Y.ravel()  
#-------------------------------------------------------------------------------------
print("===  Resultats Apprentissage supervisé  ===")
#Tree Decision classification
print("===  Tree Decision Classification  ===")
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
start_time = time.time()
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2,stratify=Y)
dtc = DecisionTreeClassifier()
dtc = dtc.fit(X_train, y_train)
pred =dtc.predict(X)
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y, pred)))
auc= metrics.roc_auc_score(Y, pred)
print("AUC:", auc)
print("Temps d execution Tree Decision: %s secondes ---" % (time.time() - start_time))
#-------------------------------------------------------------------------------------
#Random forest classification
print("===  Random forest  Classification  ===")
start_time = time.time()
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=20, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X)
print("=== Root Mean Squared Error  ===")
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y, y_pred)))
auc= metrics.roc_auc_score(Y, y_pred)
print("AUC:", auc)
print("Temps d execution Random Forest : %s secondes ---" % (time.time() - start_time))
#-------------------------------------------------------------------------------------
#Naive Bayes  classification
print("===  Naive Bayes Classification  ===")
start_time = time.time()
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
mnb.fit(X_train, y_train)
y_pred = mnb.predict(X)
print(confusion_matrix(Y, y_pred))
print(classification_report(Y, y_pred))
print("Accuracy:",metrics.accuracy_score(Y, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y, y_pred)))
auc= metrics.roc_auc_score(Y, y_pred)
print("AUC:", auc)
print("Temps d execution Naive Bayes : %s secondes ---" % (time.time() - start_time))
#-------------------------------------------------------------------------------------
# Logistic regression classification
print("===  Logistic regression Classification  ===")
start_time = time.time()
logmodel = LogisticRegression( solver='lbfgs',multi_class='multinomial')
logmodel.fit(X_train, y_train)
predictions = logmodel.predict(X)
#print(predictions)
print(classification_report(Y, predictions))
print(confusion_matrix(Y, predictions))
print("=== Root Mean Squared Error  ===")
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y, predictions)))
print("Accuracy:",accuracy_score(Y, predictions))
auc= metrics.roc_auc_score(Y, predictions)
print("AUC:" ,auc)
print("Temps d execution Regression logistique : %s secondes ---" % (time.time() - start_time))
#-------------------------------------------------------------------------------------
#KNN  classification
print("===  KNN Classification  ===")
start_time = time.time()
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=3, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X)
print(confusion_matrix(Y, y_pred))
print(classification_report(Y, y_pred))
print("Accuracy:",metrics.accuracy_score(Y, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y, y_pred)))
auc= metrics.roc_auc_score(Y, y_pred)
print("AUC:", auc)
print("Temps d execution KNN : %s secondes ---" % (time.time() - start_time))
#----------------------------------------------------------------------------------------
#SVM classification
print("===  SVM Classification  ===")
start_time = time.time()
from sklearn.svm import SVC
svclassifier = SVC(kernel='rbf', gamma=100, C=10000)
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
prediction = svclassifier.predict(X)
print(classification_report(Y, prediction))
print(confusion_matrix(Y, prediction))
print("=== Root Mean Squared Error  ===")
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y, prediction)))
print("Accuracy:",accuracy_score(Y, prediction))
auc= metrics.roc_auc_score(Y, prediction)
print("AUC:", auc)
print("Temps d execution SVM : %s secondes ---" % (time.time() - start_time))

#-------------------------------------------------------------------------------------
#Deep Learning classification
print("===  Deep Learning Classification  ===")
start_time = time.time()
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(activation='logistic', alpha=1e-05, batch_size='auto',
              beta_1=0.9, beta_2=0.999, early_stopping=False,
              epsilon=1e-08, hidden_layer_sizes=(50,),
              learning_rate='constant', learning_rate_init=0.001,
              max_iter=1000, momentum=0.9, n_iter_no_change=10,
              nesterovs_momentum=True, power_t=0.5,
              shuffle=True, solver='lbfgs', tol=0.0000001,
              validation_fraction=0.1, verbose=False, warm_start=False)

clf.fit(X_train, y_train)
y_pred = clf.predict(X)
print(classification_report(Y, y_pred))
print( confusion_matrix(Y, y_pred))
print("=== Root Mean Squared Error  ===")
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y, y_pred)))
auc= metrics.roc_auc_score(Y, y_pred)
print("AUC:", auc)
print("Accuracy:",metrics.accuracy_score(Y, y_pred))
print("Temps d execution Deep Learning : %s secondes ---" % (time.time() - start_time))

os.system("pause")
