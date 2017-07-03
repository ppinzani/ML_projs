
import pandas as pd
#from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
#from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
#import matplotlib.pyplot as plt


def get_el_tuple(x):
	try:
		ret = x[0]
	except:
		ret = 0

	return ret

train_df = pd.read_csv('train.csv')

predictors = train_df.columns[1:]

#SGD Classifier
#alg = SGDClassifier(average=True)

#Get the scores for 3 cross validation folds
#scores = cross_validation.cross_val_score(alg, train_df[predictors], train_df["label"], cv=3)

#print "Average Score SGDClassifier= " + str(scores.mean())

#Now try with neural network
hidden_layers = (len(predictors),len(predictors)) #2 Hidden Layer same elements as input
alg = MLPClassifier(solver='adam', alpha=0.000001 ,hidden_layer_sizes=hidden_layers, random_state=1, verbose=True)

''' 
J_cv = []
J_train = []
m = []


for data_size in range(42000,43000,4200):
	crnt_df = train_df.sample(data_size)

	X_train, X_test, y_train, y_test = train_test_split(crnt_df[predictors], crnt_df['label'], test_size=0.4, random_state=0)

	mlb = MultiLabelBinarizer()
	y_train = mlb.fit_transform(y_train.apply(lambda x: [x]))
	y_test = mlb.fit_transform(y_test.apply(lambda x: [x]))


	alg.fit(X_train,y_train)

	score = alg.score(X_test,y_test)


	print "Neural network train. Data Size: " + str(data_size)
	print "Loss: " + str(alg.loss_)
	print "Iterations: " + str(alg.n_iter_)
	print "Score in test set: " + str(score)

	J_cv.append(1.0 - score)
	J_train.append(alg.loss_)
	m.append(data_size)


print J_cv
print J_train
print m


plt.plot(m,J_cv)
plt.plot(m,J_train)
plt.show()

'''

X_train, X_test, y_train, y_test = train_test_split(train_df[predictors], train_df['label'], test_size=0.4,random_state=0)

mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(y_train.apply(lambda x: [x]))
y_test = mlb.fit_transform(y_test.apply(lambda x: [x]))


alg.fit(X_train,y_train)

score = alg.score(X_test,y_test)


print "Neural network train"
print "Loss: " + str(alg.loss_)
print "Iterations: " + str(alg.n_iter_)
print "Score in test set: " + str(score)


#Load test data set
test_df = pd.read_csv('test.csv')

predictions = alg.predict(test_df)


new_predictions = map(get_el_tuple,mlb.inverse_transform(predictions))
df_dict = {'ImageId':range(1,len(test_df)+1), 'Label':new_predictions}

submit_df = pd.DataFrame(df_dict)
submit_df = submit_df.set_index("ImageId")

print submit_df.head()

submit_df.to_csv('predictions.csv')



 
