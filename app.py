from flask import Flask, request, redirect, url_for, flash, jsonify
from asl_feat_preprocessing_final import drop_cols_df, remove_org_cols_df,get_rel_score_df
from asl_feat_extracxn_final import feat_ext, list_to_arr, create_Y_matrix
from convert_to_csv import convert_to_csv

import pandas as pd
import numpy as np
import pickle as p
import json

app = Flask(__name__)

@app.route("/", methods=['GET','POST'])

def predict():
	data = request.get_json(force=True)

	with open('keypoints.json', 'w') as f:
		json.dump(data, f)
	#
	convert_to_csv('keypoints.json', 'action.csv')
	# if(data != None)
	# 	df = []
	# 	for frame in data:
	# 		keypoints = frame['keypoints']
	df = pd.read_csv('action.csv')

	df = drop_cols_df(df)
	df = get_rel_score_df(df)
	df = remove_org_cols_df(df)
	df = df.iloc[:,8:24]



	# feature_extraction
	feat = feat_ext(df)
	feat = list_to_arr(feat)
	feat=np.reshape(feat,(1,feat.shape[0]))

	# predict

	filename1='models/model0.pkl'
	filename2='models/model1.pkl'
	filename3='models/model2.pkl'
	filename4='models/model3.pkl'
	loaded_model1 = p.load(open(filename1, 'rb'))
	loaded_model2 = p.load(open(filename2, 'rb'))
	loaded_model3 = p.load(open(filename3, 'rb'))
	loaded_model4 = p.load(open(filename4, 'rb'))
	models=[loaded_model1,loaded_model2,loaded_model3,loaded_model4]
	predicts=[]
	for m in models:
		label=m.predict(feat)
		if label==0.0:
			predicts.append("buy")
		elif label==1.0:
			predicts.append("communicate")
		elif label==2.0:
			predicts.append("fun")
		elif label==3.0:
			predicts.append("hope")
		elif label==4.0:
			predicts.append("mother")
		else:
			predicts.append("really")

	model_list=['1','2','3','4']
	result = dict(zip(model_list, predicts))
	# res = {loaded_model1:predicts[0], loaded_model2:predicts[1], loaded_model3:predicts[2],loaded_model4:predicts[3]}
	return(str(json.dumps(result)))

	# print(loaded_model2.predict(feat))
	# print(loaded_model3.predict(feat))
	# print(loaded_model4.predict(feat))


	# print('hello')
	# return str(len(feat))



# def hello():
# 	return "hello "

if __name__ == "__main__":
	app.run(debug=True, host='0.0.0.0')
