import pandas as pd
import librosa
import numpy as np
import scipy.stats as sps
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_auc_score, recall_score
from xgboost import XGBClassifier

# Loading data
df = pd.read_csv('equal_version.csv')

# Preparing data for model use
x = df.drop(['filepath', 'classification'], axis=1)
y = df['classification']

# Converting imaginary values to float
for column in x.columns:
	if x[column].dtype == object:
		x[column] = x[column].apply(complex)

for column in x.columns:
	if x[column].dtype == complex:
		x[column] = x[column].apply(lambda x: x.real)
	
# Creating empty lists of features to keep
xgb_columns = []
voting_columns = []

# Iteratively selecting high performing columns for XGBClassifier
for a in range(len(x.columns)):
	try:	
		X = x[x.columns[a]].values.reshape(-1,1)
		x_train, x_test, y_train, y_test = train_test_split(X, y)
		model = XGBClassifier()
		model.fit(x_train, y_train)
		roc1 = roc_auc_score(y_test, model.predict(x_test))
		print(a, x.columns[a], 'XGBClassifier Fold 1')
		x_train, x_test, y_train, y_test = train_test_split(X, y)
		model = XGBClassifier()
		model.fit(x_train, y_train)
		roc2 = roc_auc_score(y_test, model.predict(x_test))
		print(a, x.columns[a], 'XGBClassifier Fold 2')
		x_train, x_test, y_train, y_test = train_test_split(X, y)
		model = XGBClassifier()
		model.fit(x_train, y_train)
		roc3 = roc_auc_score(y_test, model.predict(x_test))
		score = np.mean([roc1, roc2, roc3])
		print(a, x.columns[a], 'XGBClassifier Fold 3')
		if score > 0.6:
			xgb_columns.append(x.columns[a])
			voting_columns.append(x.columns[a])
			print(a, x.columns[a])
	except:
		pass

# Copy and past of above block but for AdaBoost
abc_columns = []

# Iteratively selecting high performing columns for AdaBoostClassifier
for a in range(len(x.columns)):
	try:	
		X = x[x.columns[a]].values.reshape(-1,1)
		x_train, x_test, y_train, y_test = train_test_split(X, y)
		model = AdaBoostClassifier()
		model.fit(x_train, y_train)
		roc1 = roc_auc_score(y_test, model.predict(x_test))
		print(a, x.columns[a], 'AdaBoostClassifier Fold 1')
		x_train, x_test, y_train, y_test = train_test_split(X, y)
		model = AdaBoostClassifier()
		model.fit(x_train, y_train)
		roc2 = roc_auc_score(y_test, model.predict(x_test))
		print(a, x.columns[a], 'AdaBoostClassifier Fold 2')
		x_train, x_test, y_train, y_test = train_test_split(X, y)
		model = AdaBoostClassifier()
		model.fit(x_train, y_train)
		roc3 = roc_auc_score(y_test, model.predict(x_test))
		score = np.mean([roc1, roc2, roc3])
		print(a, x.columns[a], 'AdaBoostClassifier Fold 3')
		if score > 0.6:
			abc_columns.append(x.columns[a])
			voting_columns.append(x.columns[a])
			print(a, x.columns[a])
	except:
		pass

# Fitting a final XGB model with the selected features
x_final = x[xgb_columns]
x_train, x_test, y_train, y_test = train_test_split(x_final, y)
model = XGBClassifier()
model.fit(x_train, y_train)
print('XG ROC ', (roc_auc_score(y_test, model.predict(x_test))))
print('XG Recall ', (recall_score(y_test, model.predict(x_test))))
print(len(xgb_columns), '\n')

# Pickling model for use in another document
from sklearn.externals import joblib
joblib.dump(model, 'xgboost___.pkl') 

# Same as above but for AdaBoost
x_final = x[abc_columns]
x_train, x_test, y_train, y_test = train_test_split(x_final, y)
model = AdaBoostClassifier()
model.fit(x_train, y_train)
print('Ada ROC ', (roc_auc_score(y_test, model.predict(x_test))))
print('Ada Recall ', (recall_score(y_test, model.predict(x_test))))
print(len(abc_columns), '\n')
joblib.dump(model, 'adaboost___.pkl') 

# Trying variations of VotingClassifier; manual cross validation
from sklearn.ensemble import VotingClassifier
xg = XGBClassifier()
ada = AdaBoostClassifier()

voter_columns = list(set(voting_columns))
x_voting = x[voter_columns]
x_train, x_test, y_train, y_test = train_test_split(x_voting, y)

voting_soft = VotingClassifier(estimators=[('xg', xg), ('ada', ada)], voting='soft')
voting_soft.fit(x_train, y_train)
print('Voting Soft ROC ', roc_auc_score(y_test, voting_soft.predict(x_test)))
print('Voting Soft Recall ', recall_score(y_test, voting_soft.predict(x_test)))
joblib.dump(model, 'softvoting.pkl') 

voting_weight_1 = VotingClassifier(estimators=[('xg', xg), ('ada', ada)], voting='soft', weights=[3,1])
voting_weight_1.fit(x_train, y_train)
print('Voting Weight on XGB ROC ', roc_auc_score(y_test, voting_weight_1.predict(x_test)))
print('Voting Weight on XGB Recall ', recall_score(y_test, voting_weight_1.predict(x_test)))
joblib.dump(model, 'softvoting-xgb-weighted.pkl') 

voting_weight_2 = VotingClassifier(estimators=[('xg', xg), ('ada', ada)], voting='soft', weights=[1,3])
voting_weight_2.fit(x_train, y_train)
print('Voting Weight on Ada ROC ', roc_auc_score(y_test, voting_weight_2.predict(x_test)))
print('Voting Weight on Ada Recall ', recall_score(y_test, voting_weight_2.predict(x_test)))
joblib.dump(model, 'softvoting-ada-weighted.pkl') 

print(xgb_columns)
print(abc_columns)





