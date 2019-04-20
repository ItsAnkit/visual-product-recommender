import sframe,sklearn,numpy as np
from sklearn import neighbors
from sklearn.externals import joblib

#Load Data
data = sframe.SFrame.read_csv('real_data.csv')

#knn model
knn_model = sklearn.neighbors.NearestNeighbors()
x = data['deep_features']
y = data['asin']
knn_model.fit(x,y)

#dump knn_model to location knn_model/knn_model.pkl
joblib.dump(knn_model, 'knn_model/knn_model.pkl')