import io
import numpy as np
from PIL import Image
from transformer_model_helper import TransformerModel 
from CNN_model_helper import CNNModel
from flask import Flask, jsonify, request

transformerModel = TransformerModel()
cnnModel = CNNModel()

def prepare_image(img):
    img = Image.open(io.BytesIO(img))
    img = img.resize((250,250))
    img = np.asarray(img)
    return img

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
	if 'file' not in request.files:
		return jsonify("Please try again. The Image doesn't exist")
		    
	file = request.files.get('file')
	model_name = request.form.get('model')
	 
	try:
		img_bytes = file.read()
		img = prepare_image(img_bytes)
	except:
		return jsonify("Please try again. error loading the image")
	
	if model_name == 'Transformer':
		return jsonify(transformerModel.predict(img))
	elif model_name == 'CNN':
		return jsonify(cnnModel.predict(img))
	else:
		return jsonify('The Model Not Found')
