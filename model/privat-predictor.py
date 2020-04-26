import keras
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
import h5py
from keras.models import load_model
import numpy as np
from keras.models import model_from_json
import pickle
from keras.preprocessing.sequence import pad_sequences


class Predictor():

    def __init__(self):
        self._load()
        self._load_tokenizer()


    def _load(self):
        '''
        initalize the model with a predefined architecture
        and weights
        '''
        try:
            json_file = open('weights/model.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)

            loaded_model.load_weights("weights/model_weights.h5")
            self.model=loaded_model
            print(self.model.summary())
        except:
            print('Loading backup weights')
            self._load_backup()
                

    def _load_tokenizer(self):
        with open('tokenizer/tokenizer.pickle', 'rb') as handle:
            self.tokenizer = pickle.load(handle)



    def _load_backup(self):
        ''' 
        These are backu weights! We might update them once everything is over.
        But they do not perform to well
        '''

        #Create an instance of the model architecture
        json_file = open('weights/model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
                

        #load specific weights from np array
        embedded = np.load('weights/backup/embedded_weights.npy')
        lstm_weights = np.load('weights/backup/LSTM_weights.npy', allow_pickle=True)
        dense_weights = np.load('weights/backup/dense_weights.npy',allow_pickle=True)
        out_weights = np.load('weights//backupout_weights.npy',allow_pickle=True)

        #manually put the weights in their place
        loaded_model.layers[0].set_weights(embedded)
        loaded_model.layers[1].set_weights(lstm_weights)
        loaded_model.layers[4].set_weights(dense_weights)
        loaded_model.layers[6].set_weights(out_weights)
        
        self.model = loaded_model
        

    


    def update_weights(path_to_weights):
        '''
        Update the weights of an existing architecture
        If we have trained the current architecture and got better
        results, etc.
        input: path to the weights respectively from the folder
                of this instance
        return True if succeded or false if it failed
        '''
        try:
            self.model.load_weights(path_to_weights)
            return True
        except:
            return False




    def update_model(path_to_model, path_to_weights):
        '''
        Update the whole model with new weights.
        Can be used if we have an instance of the model running
        and want to deploy a new architecture
        input: path to the model.json and weights.h5 for the new arch
                They need to be respectively to the model place
        return True if secceded and False if failed
        '''

        try:
            json_file = open(path_to_model, 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)

            loaded_model.load_weights(path_to_weights)
            self.model=loaded_model
            return True
        except:
            return False


    


    def analyze(self,to_test):
        '''
        Use the model to analyze a text string and try to 
        find if it is fake or not
        input: text string
        return: prediction of the model
        '''
        tokenized = self.tokenizer.texts_to_sequences(texts=[to_test])
        model_input = pad_sequences(np.asarray(tokenized), maxlen=1000)

        acc = self.model.predict(model_input)
        return acc[0]

pre = Predictor()
