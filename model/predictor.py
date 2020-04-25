from keras.models import load_model


class Predictor()

	def __init__(self):
		self._load()

		
	def analyze(to_test):

		acc = self.model.predict(to_test)
		return acc


	def _load():
		self.model = load_model('model.h5')
		self.model.load_weights('weights.h5')
