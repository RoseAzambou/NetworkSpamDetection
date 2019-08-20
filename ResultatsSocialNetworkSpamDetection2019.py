Python 3.7.4 (tags/v3.7.4:e09359112e, Jul  8 2019, 20:34:20) [MSC v.1916 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> 
 RESTART: E:/cours Rose IG/TFE ROSE 2019/RAPPORT FINAL TFE 2019/codeTfeSocialNetworkSpamDetection2019.py 
Detection de spam dans les reseaux sociaux
Number of Relations =
(122641, 5)
===Liste des utilisateurs source===
Nombre d'utilisateurs sources  de l'échantillon =
574
===  Resultats Apprentissage supervisé  ===
===  Tree Decision Classification  ===
Root Mean Squared Error: 0.21688321172034594
AUC: 0.662045594577942
Temps d execution Tree Decision: 0.16999006271362305 secondes ---
===  Random forest  Classification  ===
=== Root Mean Squared Error  ===
Root Mean Squared Error: 0.1978266766497661
AUC: 0.8753710860919733
Temps d execution Random Forest : 0.22998476028442383 secondes ---
===  Naive Bayes Classification  ===
[[172 369]
 [  4  29]]
              precision    recall  f1-score   support

         0.0       0.98      0.32      0.48       541
         1.0       0.07      0.88      0.13        33

    accuracy                           0.35       574
   macro avg       0.53      0.60      0.31       574
weighted avg       0.93      0.35      0.46       574

Accuracy: 0.3501742160278746
Root Mean Squared Error: 0.8061177233953645
AUC: 0.5983588192460652
Temps d execution Naive Bayes : 0.09299349784851074 secondes ---
===  Logistic regression Classification  ===
              precision    recall  f1-score   support

         0.0       0.95      1.00      0.97       541
         1.0       0.67      0.06      0.11        33

    accuracy                           0.94       574
   macro avg       0.81      0.53      0.54       574
weighted avg       0.93      0.94      0.92       574

[[540   1]
 [ 31   2]]
=== Root Mean Squared Error  ===
Root Mean Squared Error: 0.2361125344403821
Accuracy: 0.9442508710801394
AUC: 0.5293788158852853
Temps d execution Regression logistique : 0.17798829078674316 secondes ---
===  KNN Classification  ===
[[541   0]
 [ 32   1]]
              precision    recall  f1-score   support

         0.0       0.94      1.00      0.97       541
         1.0       1.00      0.03      0.06        33

    accuracy                           0.94       574
   macro avg       0.97      0.52      0.52       574
weighted avg       0.95      0.94      0.92       574

Accuracy: 0.9442508710801394
Root Mean Squared Error: 0.2361125344403821
AUC: 0.5151515151515151
Temps d execution KNN : 0.24498438835144043 secondes ---
===  SVM Classification  ===
              precision    recall  f1-score   support

         0.0       0.96      0.99      0.98       541
         1.0       0.81      0.39      0.53        33

    accuracy                           0.96       574
   macro avg       0.89      0.69      0.75       574
weighted avg       0.96      0.96      0.95       574

[[538   3]
 [ 20  13]]
=== Root Mean Squared Error  ===
Root Mean Squared Error: 0.20017414021583763
Accuracy: 0.9599303135888502
AUC: 0.6941970537164622
Temps d execution SVM : 0.3409759998321533 secondes ---
===  Deep Learning Classification  ===
              precision    recall  f1-score   support

         0.0       0.95      1.00      0.97       541
         1.0       1.00      0.06      0.11        33

    accuracy                           0.95       574
   macro avg       0.97      0.53      0.54       574
weighted avg       0.95      0.95      0.92       574

[[541   0]
 [ 31   2]]
=== Root Mean Squared Error  ===
Root Mean Squared Error: 0.23239399441705672
AUC: 0.5303030303030303
Accuracy: 0.945993031358885
Temps d execution Deep Learning : 6.184587717056274 secondes ---
>>> 
