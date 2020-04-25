import keras
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
import h5py
from keras.models import load_model
import numpy as np
from keras.models import model_from_json


class Predictor():

	def __init__(self):
		self._load()

		
	def _load(self):
		try:
			self.model = keras.models.load_model('weights/GloVe6b-100-LSTM-1.h5')
		except:
			# Normal loading did not work. 
			#print('regular load failed')
			try:
				#print('try to load the model from json')
				json_file = open('weights/model.json', 'r')
				loaded_model_json = json_file.read()
				json_file.close()
				loaded_model = model_from_json(loaded_model_json)

				loaded_model.load_weights("weights/model_weights.h5")
				self.model=loaded_model
			except:
				print('Loading backup weights')
				# Loading it with json did not work either. 
				# Only thing that is left is to load the weights with numpy
				# horrible.. but works
				
				json_file = open('../Experiments/GloVe/model.json', 'r')
				loaded_model_json = json_file.read()
				json_file.close()
				loaded_model = model_from_json(loaded_model_json)
				

				embedded = np.load('weights/backup/embedded_weights.npy')
				lstm_weights = np.load('weights/backup/LSTM_weights.npy', allow_pickle=True)
				dense_weights = np.load('weights/backup/dense_weights.npy',allow_pickle=True)
				out_weights = np.load('weights//backupout_weights.npy',allow_pickle=True)

				loaded_model.layers[0].set_weights(embedded)
				loaded_model.layers[1].set_weights(lstm_weights)
				loaded_model.layers[4].set_weights(dense_weights)
				loaded_model.layers[6].set_weights(out_weights)
				#print(loaded_model.summary())
				self.model = loaded_model

		#print(self.model.summary())
		
		
	def analyze(to_test):

		tokenized = self.tokenizer.texts_to_sequence([to_test])

		acc = self.model.predict(tokenized)
		return acc



#pre = Predictor()


