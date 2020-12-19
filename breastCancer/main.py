import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold, RandomizedSearchCV
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn import svm
from sklearn.linear_model import LogisticRegression
import pickle

# load the data
data_set = load_breast_cancer()
# creat the dataframe
data_df = pd.DataFrame(np.c_[data_set['data'], data_set['target']], columns=np.append(data_set['feature_names'], ['target']))
# get the data
X = data_df.drop(['target'], 1)
# get the label
y = data_df['target']
# spilt the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# do a scale for the data
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)
# try support vector machine algo
# we train the model without the scaler data
clf = svm.SVC(kernel='rbf', C=10, gamma='scale')
clf.fit(X_train, y_train)
pre = clf.predict(X_test)
score = accuracy_score(y_test, pre)
print(f'the score data without scale is {score}')
# now we will train our model with the scaler data
clf_sc = svm.SVC(kernel='rbf', C=10, gamma='scale')
clf_sc.fit(X_train_sc, y_train)
pre_sc = clf_sc.predict(X_test_sc)
score_sc = accuracy_score(y_test, pre_sc)
print(f'the score with scaler data is {score_sc}')
# get the f1 score for the svm
f1 = f1_score(y_test,  pre_sc, labels=[''])
class_repo = classification_report(y_test, pre_sc)
print(f'the class report using svm is {class_repo}')
# use logistic reg with scaler data
cls_log = LogisticRegression(random_state=51).fit(X_train_sc, y_train)
pre_log = cls_log.predict(X_test_sc)
score_log = accuracy_score(y_test, pre_log)
print(f'the score for the logistic re without the scale data is {score_log}')
# get the class repo for logistic reg
class_repo_log = classification_report(y_test, pre_log)
print(class_repo_log)
parameters = {'kernel': ['rbf'], 'C': [0.1, 1, 10, 100, 1000], 'gamma': [0.1, 1, 0.01, 0.001], 'epsilon': [1, 0.1, 2, 0.01, 0.0001]}
# save our model
pickle.dump(clf, open('breast_cancer_detector.pickle', 'wb'))
# load the model
breast_cancer_model = pickle.load(open('breast_cancer_detector.pickle', 'rb'))
# predict the out put
y_pre = breast_cancer_model.predict(X_test)
# get the score for the loaded model
a_score = accuracy_score(y_test, y_pre)
print(f'the score for the final model is {a_score}')








