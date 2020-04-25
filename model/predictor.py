import keras
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
import h5py


class Predictor():

	def __init__(self):
		self._load()

		
	def _load(self):
		self.model = keras.models.load_model('GloVe6b-100-LSTM.h5')
		#self.model.load_weights('weights.h5')
		print(model.summary())
		'''
		self.embeddings_index = {}
		f = open('embedding/', encoding='utf8')
		#print('Loading Glove from:', GLOVE_DIR,'…', end='')
		for line in f:
		    values = line.split()
		    word = values[0]
		    self.embeddings_index[word] = np.asarray(values[1:], dtype='float32')
		f.close()
		'''
		self.tokenizer = Tokenizer(100000)

	def analyze(to_test):

		tokenized = self.tokenizer.texts_to_sequence([to_test])

		acc = self.model.predict(tokenized)
		return acc



pre = Predictor()


