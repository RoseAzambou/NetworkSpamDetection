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
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state= 42,stratify=Y)
dtc = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=50,
                       max_features='sqrt', max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=4, min_samples_split=5,
                       min_weight_fraction_leaf=0.0, presort=False,
                       random_state=None, splitter='best')
dtc = dtc.fit(X_train, y_train)
pred =dtc.predict(X_test)

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, pred)))
auc= metrics.roc_auc_score(y_test, pred)
print("AUC:", auc)
fpr, tpr, thresholds = roc_curve(y_test, pred)
plt.plot(fpr, tpr, 'b', label='DecisionTree') 
plt.plot([0, 1], [0, 1], 'k--') # random predictions curve 
plt.xlim([0.0, 1.0]) 
plt.ylim([0.0, 1.0]) 
plt.xlabel('False Positive Rate ') 
plt.ylabel('True Positive Rate' ) 
plt.title('Receiver Operating Characteristic')
plt.show()
print("Temps d execution Tree Decision: %s secondes ---" % (time.time() - start_time))
#-------------------------------------------------------------------------------------
#Random forest classification
print("===  Random forest  Classification  ===")
start_time = time.time()
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=50,
                      max_features='sqrt', max_leaf_nodes=None,
                      min_impurity_decrease=0.0, min_impurity_split=None,
                      min_samples_leaf=1, min_samples_split=5,
                      min_weight_fraction_leaf=0.0, n_estimators=65,
                      n_jobs=None, oob_score=False, random_state=None,
                      verbose=0, warm_start=False)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print("=== Root Mean Squared Error  ===")
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
auc= metrics.roc_auc_score(y_test, y_pred)
print("AUC:", auc)
print("Temps d execution Random Forest : %s secondes ---" % (time.time() - start_time))
#-------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------
# Logistic regression classification
print("===  Logistic regression Classification  ===")
start_time = time.time()
logmodel = LogisticRegression(C=10.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='warn', n_jobs=None, penalty='l2',
                   random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                   warm_start=False)
logmodel.fit(X_train, y_train)
predictions = logmodel.predict(X_test)
#print(predictions)
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print("=== Root Mean Squared Error  ===")
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
print("Accuracy:",accuracy_score(y_test, predictions))
auc= metrics.roc_auc_score(y_test, predictions)
print("AUC:" ,auc)
print("Temps d execution Regression logistique : %s secondes ---" % (time.time() - start_time))
#-------------------------------------------------------------------------------------
#KNN  classification
print("===  KNN Classification  ===")
start_time = time.time()
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                    metric_params=None, n_jobs=None, n_neighbors=10, p=2,
                    weights='uniform')
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
auc= metrics.roc_auc_score(y_test, y_pred)
print("AUC:", auc)
print("Temps d execution KNN : %s secondes ---" % (time.time() - start_time))
#----------------------------------------------------------------------------------------
#SVM classification
print("===  SVM Classification  ===")
start_time = time.time()
from sklearn.svm import SVC
svclassifier = SVC(C=10, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=0.001, kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
prediction = svclassifier.predict(X_test)
print(classification_report(y_test, prediction))
print(confusion_matrix(y_test, prediction))
print("=== Root Mean Squared Error  ===")
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))
print("Accuracy:",accuracy_score(y_test, prediction))
auc= metrics.roc_auc_score(y_test, prediction)
print("AUC:", auc)
print("Temps d execution SVM : %s secondes ---" % (time.time() - start_time))

#-------------------------------------------------------------------------------------
#Deep Learning classification
print("===  Deep Learning Classification  ===")
start_time = time.time()
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(activation='logistic', alpha=0.0001, batch_size='auto',
              beta_1=0.9, beta_2=0.999, early_stopping=False, epsilon=1e-08,
              hidden_layer_sizes=(350, 3), learning_rate='constant',
              learning_rate_init=0.001, max_iter=1500, momentum=0.9,
              n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
              random_state=None, shuffle=True, solver='adam', tol=0.0001,
              validation_fraction=0.1, verbose=False, warm_start=False)
    
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
print( confusion_matrix(y_test, y_pred))
print("=== Root Mean Squared Error  ===")
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
auc= metrics.roc_auc_score(y_test, y_pred)
print("AUC:", auc)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Temps d execution Deep Learning : %s secondes ---" % (time.time() - start_time))

os.system("pause")
