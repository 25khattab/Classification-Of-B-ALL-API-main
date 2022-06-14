from transformers import ViTFeatureExtractor, ViTForImageClassification

class TransformerModel:
	def __init__(self):
		self.feature_extractor = ViTFeatureExtractor.from_pretrained("models/transformer_model/")
		self.model = ViTForImageClassification.from_pretrained("models/transformer_model/")
	
	def predict(self,img):
		inputs = self.feature_extractor(img, return_tensors="pt")
		outputs = self.model(**inputs)
		if outputs['logits'][0][0]>outputs['logits'][0][1]:
			return "ALL"
		else:
			return "Hem"
