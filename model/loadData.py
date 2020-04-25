import pandas as pd
from sklearn.model_selection import train_test_split

def get_data():
	'''
	Simple function, that ready in the data, cleans it
	and returns it already split and train and test
	'''
	complete_data = pd.read_csv('../datasets/complete-set.csv')
	complete_data.dropna()
	texts = complete_data['content'].to_numpy()

	labels = complete_data['labels'].to_numpy()

	print('Data will be returned as: ')
	print('x_train, x_test, y_train, y_test')
	return train_test_split(texts,labels)
