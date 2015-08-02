from time import time
from itertools import combinations
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier

import scipy as sp, numpy as np, pandas as pd
import re

def prepare_data():
	"""
	Load train data as well as test data
	"""
	train_data = pd.read_csv('train.csv', parse_dates=[0])
	test_data = pd.read_csv('test.csv', parse_dates=[0])

	train_data['Hour'] = train_data['Dates'].map(lambda x: x.hour)
	train_data['Month'] = train_data['Dates'].map(lambda x: x.month)
	train_data['Day'] = train_data['Dates'].map(lambda x: x.day)
	train_data['Year'] = train_data['Dates'].map(lambda x: x.year)
	train_data['WeekDay'] = train_data['Dates'].map(lambda x: x.dayofweek)

	# reg_st = re.compile(r"\d*\w+\s\w{2}$")
	# def parse_st(s):
	# 	p=reg_st.search(s)
	# 	if p:
	# 		return p.group(0)
	# 	else:
	# 		print s
	# 		return ''
	# train_data['St'] = train_data['Address'].map(lambda x: reg_st.search(x).group(0))
	# train_data['St'] = train_data['Address'].map(parse_st)	

	# continuous features
	continuous = ['Year', 'Month', 'Day', 'WeekDay', 'Hour', 'X', 'Y']
	# categorical features
	discrete = ['PdDistrict']
	# extra feature?
	extra=[]
	target = ['Category']


	# Fill NAs (in X and Y)

	# lower left and upper left, boundary of sf
	ur_lat = 37.82986
	ll_lat = 37.69862

	ur_lon = -122.33663 
	ll_lon = -122.52469

	train_data = train_data[(train_data.X>ll_lon) & (train_data.X<ur_lon) & (train_data.Y<ur_lat) & (train_data.Y>ll_lat)]


	encoders = dict()


	for col in discrete:
		encoders[col] = preprocessing.LabelEncoder()
		train_data[col] = encoders[col].fit_transform(train_data[col])

	train_x = train_data[continuous+discrete].values
	train_y = train_data[target].values.ravel()


	####   Read Test Data

	test_data = pd.read_csv('test.csv', parse_dates=[1])

	test_data['Hour'] = test_data['Dates'].map(lambda x: x.hour)
	test_data['Month'] = test_data['Dates'].map(lambda x: x.month)
	test_data['Day'] = test_data['Dates'].map(lambda x: x.day)
	test_data['Year'] = test_data['Dates'].map(lambda x: x.year)
	test_data['WeekDay'] = test_data['Dates'].map(lambda x: x.dayofweek)

	for col in discrete:
		test_data[col] = encoders[col].transform(test_data[col])

	test_x = test_data[continuous+discrete].values
	return train_data, train_x, train_y, test_data, test_x




def submit(test_y, filename='submit.csv'):
	"""
	Given predicted results test_y, prepare to be submitted file
	"""
	ids = pd.Series(range(len(test_y)), name='Id')
	with open('sampleSubmission.csv') as f:
		Categories=f.readline().strip().split(',')[1:]
	result = pd.DataFrame(columns=Categories, index=ids)
	result['preds'] = test_y
	for s in Categories:
		result[s] = result['preds'].map(lambda x: 1 if x==s else 0)
	result=result.drop('preds', axis=1)
	result.to_csv(filename)
	return


if __name__=='__main__':
	train_data, train_x, train_y, test_data, test_x = prepare_data()
	####  Fit RandomForest Model
	params = [10, 3, 10]
	# ntree, maxfea, leafsie

	n,m,l=params


	rf = RandomForestClassifier(n_estimators=n, max_features=m)
	rf.fit(train_x, train_y)
	test_y = rf.predict(test_x)
	submit(test_y)
































