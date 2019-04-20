from flask import Flask, redirect, url_for,request,jsonify,render_template,send_from_directory
app = Flask(__name__, static_url_path='')

import sframe, sklearn, keras, numpy as np,uuid,os,random
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
	count = 8
	index = random.randint(1,9900)
	feat = data[index]['deep_features']
	query_img = data[index]['imUrl']
	query_pid = data[index]['asin']

	(ind,dist) = get_similar(feat,count)
	(i,j,k) = get_data(ind, False)

	return render_template('index.html', data = zip(i,k,dist),
		query_imUrl = query_img, query_asin = query_pid)

@app.route('/<int:count>', methods=['GET'])
def indexByCount(count):
	index = random.randint(1,9900)
	feat = data[index]['deep_features']
	query_img = data[index]['imUrl']
	query_pid = data[index]['asin']

	(ind,dist) = get_similar(feat,count)
	(i,j,k) = get_data(ind, False)

	return render_template('index.html', data = zip(i,k,dist),
		query_imUrl = query_img, query_asin = query_pid)

@app.route('/<int:count>/', methods=['GET'])
def indexByCountt(count):
	index = random.randint(1,9900)
	feat = data[index]['deep_features']
	query_img = data[index]['imUrl']
	query_pid = data[index]['asin']

	(ind,dist) = get_similar(feat,count)
	(i,j,k) = get_data(ind, False)

	return render_template('index.html', data = zip(i,k,dist),
		query_imUrl = query_img, query_asin = query_pid)

@app.route('/asin/<asin>', methods=['GET'])
def asin(asin):
	count = 8
	feat = data[data['asin'] == asin]['deep_features']
	query_img = data[data['asin'] == asin]['imUrl'][0]
	query_pid = data[data['asin'] == asin]['asin'][0]

	(ind,dist) = get_similar(feat,count)
	(i,j,k) = get_data(ind, False)

	return render_template('index.html', data = zip(i,k,dist),
		query_imUrl = query_img, query_asin = query_pid)

@app.route('/asin/<asin>/', methods=['GET'])
def asinn(asin):
	count = 8
	feat = data[data['asin'] == asin]['deep_features']
	query_img = data[data['asin'] == asin]['imUrl'][0]
	query_pid = data[data['asin'] == asin]['asin'][0]

	(ind,dist) = get_similar(feat,count)
	(i,j,k) = get_data(ind, False)

	return render_template('index.html', data = zip(i,k,dist),
		query_imUrl = query_img, query_asin = query_pid)

@app.route('/asin/<asin>/<int:count>', methods=['GET'])
def asinByCount(asin,count):
	feat = data[data['asin'] == asin]['deep_features']
	query_img = data[data['asin'] == asin]['imUrl'][0]
	query_pid = data[data['asin'] == asin]['asin'][0]

	(ind,dist) = get_similar(feat,count)
	(i,j,k) = get_data(ind, False)

	return render_template('index.html', data = zip(i,k,dist),
		query_imUrl = query_img, query_asin = query_pid)

@app.route('/asin/<asin>/<int:count>/', methods=['GET'])
def asinByCountt(asin,count):
	feat = data[data['asin'] == asin]['deep_features']
	query_img = data[data['asin'] == asin]['imUrl'][0]
	query_pid = data[data['asin'] == asin]['asin'][0]

	(ind,dist) = get_similar(feat,count)
	(i,j,k) = get_data(ind, False)

	return render_template('index.html', data = zip(i,k,dist),
		query_imUrl = query_img, query_asin = query_pid)

@app.route('/image', methods=['GET', 'POST'])
def byImage():
	tmp = str(uuid.uuid4())
	os.makedirs("images/"+tmp+"/")
	name = "images/"+tmp+"/image.jpg"
	image = request.files['image']
	image.save(name)
	img_data = prepare_image(name)
	feat = get_features(img_data)
	#TODO:: Add code to remove file
	count = 8
	(ind,dist) = get_similar(feat,count)
	(i,j,k) = get_data(ind, False)
	return render_template('imagesearch.html', data = zip(i,k,dist),
		query_imUrl = name)

@app.route('/asin.html')
def asinhtml():
	return render_template('asin.html')

@app.route('/image.html')
def imagehtml():
	return render_template('image.html')

@app.route('/details.html')
def detailshtml():
	return render_template('details.html')

@app.route('/images/<uid>/image.jpg', methods=['GET'])
def getImage(uid):
	return send_from_directory('images', uid+'/image.jpg')

@app.errorhandler(404)
def not_found(error=None):
	return render_template('404.html')

if __name__ == '__main__':
	app.run(host='0.0.0.0', port='80')