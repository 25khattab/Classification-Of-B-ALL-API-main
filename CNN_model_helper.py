import numpy as np
import tensorflow as tf

class CNNModel:
	def __init__(self):
		self.mean = 9.854626920021918
		self.std = 24.502705188655078
		self.model = tf.keras.models.load_model('models/CNN_model/')
		
	def predict(self,img):
		img = np.array((img-self.mean)/self.std)
		inputs = np.expand_dims(img,0)
		outputs = self.model.predict(inputs)
		if outputs[0][0]>0.5:
			return "ALL"
		else:
			return "Hem"
