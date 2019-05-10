import numpy as np
import pandas as pd
# from sklearn import preprocessing

from flask import request, Blueprint,jsonify


advance_encodage = Blueprint('advance_encodage', __name__)
# _________________________________functions_____________________________________

#Encoage automatique
def encod_aut(df):
   listcol=list(df.columns.values)

   for cool in listcol:
   
      if (df[cool]).dtype == dtype('O'):
        df[cool] = df[cool].astype("category")
        df[cool] = df[cool].cat.codes
		
   df = df.replace(-1, NaN, regex=True)
   return df   
   
#encodage manuel
def Model_col(dataset, X):
   labels = dataset[X].astype('category').cat.categories.tolist()
   return labels 
   
def encode_catg(dataset,X,z,y):
   dataset[X] = dataset[X].replace(z, y, regex=True)
   return dataset
   
   
# _______________________________ end_functions__________________________________

@advance_encodage.route('/twitter', methods=['POST','GET'])
def receive_data():
    if request.method == 'POST':


       



















