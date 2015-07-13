#Kaggle: Liberty Mutual First Shot
cd desktop\dataScience\kaggle\libertyMutual

import numpy as np
import pandas as pd
import sklearn
from sklearn import cross_validation
from sklearn.ensemble import RandomForestRegressor

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

#check for missing data
bad = train.isnull()
bad.apply(sum)

#metric 
def normalizedGini(actual,predicted):
	data = pd.DataFrame({'actual': actual, 'predicted': predicted})
	data = data.sort_index(by = "predicted", ascending = False) #how to deal with ties?
	actual = np.asarray(data['actual'])
	cumSum = np.cumsum(actual)
	sumTotal = len(actual)
	randomGuess = (np.array(np.arange(len(actual)))+1)/float(sumTotal)
	diff = cumSum/float(actual.sum()) - randomGuess
	return diff.sum()

#convert categorical variables to dummies 	
cat = ['T1_V4',	'T1_V5','T1_V6','T1_V7','T1_V8','T1_V9','T1_V11','T1_V12',
		'T1_V15','T1_V16','T1_V17','T2_V3','T2_V5','T2_V11','T2_V12','T2_V13']

for var in cat:
	dummiesTrain = pd.get_dummies(train[var], prefix = var)
	train = train.drop(var, axis = 1)
	train = train.join(dummiesTrain)
	dummiesTest = pd.get_dummies(test[var], prefix = var)
	test = test.drop(var, axis = 1)
	test = test.join(dummiesTest)

trainHazard = train['Hazard']
trainId = train["Id"]
train = train.drop("Hazard", axis = 1)	
train = train.drop("Id", axis = 1)
	
kFolds = sklearn.cross_validation.KFold(len(train), 10)
maxFeatures = [9,10,11,12,13, 14]
ntrees = [500]
gini = pd.DataFrame(index = ntrees, columns = maxFeatures)
gini = gini.fillna(0)

for i in range(len(maxFeatures)):
	print i
	for j in range(len(ntrees)):
		print j
		rf = RandomForestRegressor(n_jobs = 1, n_estimators = ntrees[j],	
			max_features = maxFeatures[i])
		temp = []
		for trainInd, valInd in kFolds:
			print "rock"
			rfFit = rf.fit(train.iloc[trainInd], trainHazard.iloc[trainInd])
			pred = rfFit.predict(train.iloc[valInd])
			temp.append(normalizedGini(trainHazard.iloc[valInd], pred))
		gini.iloc[j,i] = np.array(temp).mean()
				

testId = test["Id"]
rf = RandomForestRegressor(n_jobs = 1, n_estimators = 1000, max_features = 10)
rfFit = rf.fit(train, trainHazard)
pred = rfFit.predict(test.drop("Id", axis = 1))

predSubmit = pd.DataFrame({"Id": testId, "Hazard": pred})
predSubmit.to_csv("pred1.csv", index = False, index_label="Id")