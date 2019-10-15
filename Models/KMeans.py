__author__ = "EA"

import pandas as  pd
from sklearn.cluster import KMeans

import json


def Training(df,n_clusters):
    clf = KMeans(n_clusters=n_clusters)
    clf.fit(df)
    return clf  

#**************************Make prediction
#__________________________________________|
def Prediction(df, clf):

    clusters=clf.predict(df)
    df['cluster'] = clusters
    return df

def Fit_Predict(df, n_clusters):
    clf = KMeans(n_clusters=n_clusters)
    labels = clf.fit_predict(df)
    
    return labels 
