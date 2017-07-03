import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split,StratifiedKFold, cross_val_score,learning_curve, validation_curve, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, make_scorer, roc_curve, auc
from scipy import interp


#Import breast cancer data
print '\n\n'
print "Loading Wisconsin breast cancer dataset"
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data',header=None)

X = df.loc[:,2:].values
y = df.loc[:,1].values

#Transform string representation to integers
le = LabelEncoder()
y = le.fit_transform(y)

#Divide Our dataset in training and test dataset (80,20)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)

#Use a Pipeline to combine estiamtors and transformers
#Inicializo pipeline
pipe_lr = Pipeline([('scl',StandardScaler()),
	 	            ('pca',PCA(n_components=2)),
	 	            ('clf',LogisticRegression(random_state=1))])

pipe_lr.fit(X_train,y_train)

print '\n\n'
print "Logistic Regression With Pipeline:"
print('Test Accuracy: %.3f' % pipe_lr.score(X_test,y_test))


#Now I will use K-Fold cross validation to tune hyperparameters
kfold = StratifiedKFold(n_splits=10,random_state=1)

scores = []

print '\n\n'
print "Stratified KFold Validation with n_splits = 10: "

for k,(train,test) in enumerate(kfold.split(X_train, y_train)): #Notar que para ver el peso de cada clase se usa solo y
	pipe_lr.fit(X_train[train],y_train[train]) #Tomo los elementos que me dice el kfold
	score = pipe_lr.score(X_train[test],y_train[test])
	scores.append(score)
	print('Fold: %s, Class Dist: %s, Acc.: %.3f' % (k+1, np.bincount(y_train[train]),score))

print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores),np.std(scores)))

#Now use cross val score

print '\n\n'
print 'Cross Val Score Example'

scores = cross_val_score(estimator=pipe_lr,X=X_train,y=y_train,cv=10,n_jobs=-1) #n_jobs = -1 use all available threads
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores),np.std(scores)))

#Now will use learning and validation curves to evaluate if an algorithm has high variance or high bias

print '\n\n'
print 'Learning and Validation Curves'
#Initialize new pipeline
pipe_lr = Pipeline([('scl',StandardScaler()),
	 	            ('clf',LogisticRegression(penalty='l2',random_state=0))])

train_sizes, train_scores, test_scores = learning_curve(
											estimator =pipe_lr,
											X=X_train,
											y=y_train,
											train_sizes=np.linspace(0.1,1.0,10), #0,1 0,2 0,3 0,4....1.0
											cv=10,
											n_jobs=-1)

train_mean = np.mean(train_scores,axis=1)
train_std = np.std(train_scores,axis=1)
test_mean = np.mean(test_scores,axis=1)
test_std = np.std(test_scores,axis=1)

#Plot learning curves
plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
plt.fill_between(train_sizes, train_mean+train_std, train_mean-train_std,alpha=0.15,color='blue')

plt.plot(train_sizes, test_mean, color='green',linestyle='--' , marker='s', markersize=5, label='validation accuracy')
plt.fill_between(train_sizes, test_mean+test_std, test_mean-test_std,alpha=0.15,color='green')

plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.8,1.0])
plt.show()


#Validation curve we vary a parameter, in this case the inverse regularization parameter C
param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

train_scores, test_scores = validation_curve(
											estimator =pipe_lr,
											X=X_train,
											y=y_train,
											param_name='clf__C',
											param_range=param_range,
											cv=10)

train_mean = np.mean(train_scores,axis=1)
train_std = np.std(train_scores,axis=1)
test_mean = np.mean(test_scores,axis=1)
test_std = np.std(test_scores,axis=1)


#Plot Validation curves
plt.plot(param_range, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
plt.fill_between(param_range, train_mean+train_std, train_mean-train_std,alpha=0.15,color='blue')

plt.plot(param_range, test_mean, color='green',linestyle='--' , marker='s', markersize=5, label='validation accuracy')
plt.fill_between(param_range, test_mean+test_std, test_mean-test_std,alpha=0.15,color='green')

plt.grid()
plt.xscale('log')
plt.xlabel('Parameter C')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.8,1.0])
plt.show()

#Now we are tuning hyperparemeters via grid search
pipe_svc = Pipeline([('scl',StandardScaler()),
	 	            ('clf',SVC(random_state=1))])

param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

#Defino una grid de parametros
param_grid = [{'clf__C' : param_range, 'clf__kernel' : ['linear']},
              {'clf__C' : param_range, 'clf__gamma' : param_range, 'clf__kernel' : ['rbf']}]

# Initialize grid search
gs = GridSearchCV(	estimator=pipe_svc,
					param_grid=param_grid,
					scoring='accuracy',
					cv=10,
					n_jobs=-1)

print '\n\n'
print 'Fitting Grid Search CV'
gs = gs.fit(X_train,y_train)

print gs.best_score_
print gs.best_params_

#Now we select the best model and test its performance on test dataset
clf = gs.best_estimator_
clf.fit(X_train,y_train)
print('Test Accuracy: %.3f' % clf.score(X_test,y_test))

#Finally were checking some other performance metrics
#For skewed classes we have precision and recall

#We are using our SVC as model to see how metrics work
pipe_svc.fit(X_train,y_train)

y_pred = pipe_svc.predict(X_test)


print '\n\n'
print('Precision for SVC: %.3f' % precision_score(y_true=y_test, y_pred=y_pred))
print('Recall for SVC: %.3f' % recall_score(y_true=y_test, y_pred=y_pred))

#If I want to change the scoring so as the positive value is 0 I can create a new scorer
scorer = make_scorer(f1_score, pos_label=0)
'''
gs = GridSearchCV(	estimator = pipe_svc,
					param_grid = param_grid,
					scoring = scorer,
					cv=10)
'''



