from flask import Flask, redirect, url_for,request,jsonify
app = Flask(__name__)

import sframe, sklearn, keras, numpy as np,uuid,os
from sklearn import neighbors
from sklearn.externals import joblib
from keras.applications.vgg16 import VGG16,preprocess_input
from keras.preprocessing import image
from keras.models import Model

#Load DataSet
data = sframe.SFrame.read_csv('real_data.csv')

#Load KNN Model
knn_model = joblib.load('knn_model/knn_model.pkl')

#Load VGG Model
base_model = VGG16(weights="imagenet")
new_model = Model(input=base_model.input, output=base_model.get_layer('fc2').output)

#Get Similar from features(4096,) and count with default of 5
#Returns indexes(in data) and distance
def get_similar(features, count=5):
	tmp = knn_model.kneighbors(features, count, return_distance=True)
	(indexes,distances) = tmp[1][0],tmp[0][0]
	return (indexes,distances)

#Get Features from numpy image(1,224,224,3) 
#Returns (4096,) feature
def get_features(img):
	return (new_model.predict(img))[0]

#Get Image in numpy form of (1,224,224,3)
#Input valid image path
def prepare_image(img_path):
	img = image.load_img(img_path, target_size=(224,224))
	tmp = image.img_to_array(img)
	return np.expand_dims(tmp,axis=0)

#Get Real Data(asin,title and imUrl)
def get_data(indexes, similar=False):
	cnt = 0
	asin = []
	title = []
	imUrl = []
	for index in indexes:
		if similar==True and cnt==0:
			cnt = cnt+1
			continue
		else:
			asin.append(data[index]['asin'])
			title.append(data[index]['title'])
			imUrl.append(data[index]['imUrl'])
	return (asin,title,imUrl)

@app.route('/')
def index():
	result = {}
	message = {
		"status": 200,
		"message": "Default Message",
		"data": result
	}
	resp = jsonify(message)
	resp.status_code = 200
	return resp

@app.route('/asin/<asin>/<int:count>', methods=['GET'])
def byasin(asin,count):
	feat = data[data['asin'] == asin]['deep_features']
	(ind,dist) = get_similar(feat,count)
	(i,j,k) = get_data(ind)
	result = []
	x = 0
	while x<len(i):
		tmp = {"asin": i[x],"title": j[x],"imUrl": k[x],"dist": dist[x]}
		result.append(tmp)
		x = x+1
	message = {
		"status": 200,
		"message": "Successfully fetched by asin!",
		"data": result
	}
	resp = jsonify(message)
	resp.status_code = 200
	return resp

@app.route('/index/<int:index>/<int:count>', methods=['GET'])
def byindex(index,count):
	feat = data[index]['deep_features']
	(ind,dist) = get_similar(feat,count)
	(i,j,k) = get_data(ind)
	result = []
	x = 0
	while x<len(i):
		tmp = {"asin": i[x],"title": j[x],"imUrl": k[x],"dist": dist[x]}
		result.append(tmp)
		x = x+1
	message = {
		"status": 200,
		"message": "Successfully fetched by index!",
		"data": result
	}
	resp = jsonify(message)
	resp.status_code = 200
	return resp

@app.route('/image/<int:count>', methods=['GET', 'POST'])
def byimage(count):
	tmp = str(uuid.uuid4())
	os.makedirs("images/"+tmp+"/")
	name = "images/"+tmp+"/image.jpg"
	ima = request.files['image']
	ima.save(name)
	img_data = prepare_image(name)
	feat = get_features(img_data)
	#TODO:: Add code to remove file

	(ind,dist) = get_similar(feat,count)
	(i,j,k) = get_data(ind)
	result = []
	x = 0
	while x<len(i):
		tmp = {"asin": i[x],"title": j[x],"imUrl": k[x],"dist": dist[x]}
		result.append(tmp)
		x = x+1
	message = {
		"status": 200,
		"message": "Successfully fetched by image!",
		"data": result
	}
	resp = jsonify(message)
	resp.status_code = 200
	return resp

@app.errorhandler(404)
def not_found(error=None):
	message = {
		"status": 404,
		"message": "Path not exists",
		"data": {}
	}
	resp = jsonify(message)
	resp.status_code = 404
	return resp;


if __name__ == '__main__':
	app.run(host='0.0.0.0', port='80')
