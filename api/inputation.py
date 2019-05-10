import numpy as np
import pandas as pd
from missingpy import KNNImputer
from missingpy import MissForest

from flask import request, Blueprint,jsonify

# _____________requirements:_____________
# missingpyVersion: 0.2.0

advance_inputation = Blueprint('advance_inputation', __name__)

# _________________________________functions_____________________________________
#inputation miss_forest
def input_missForest(df):
   X = np.asarray(df)
   imputer = MissForest()
   W=imputer.fit_transform(X)
   df1 = pd.DataFrame(W)
   df1.columns = df.columns
   return(df1)

#inputation kNN
def knn_inp(df, k):
   X = np.asarray(df)
   imputer = KNNImputer(n_neighbors=k, weights="uniform")
   W=imputer.fit_transform(X)
   newdf = pd.DataFrame(W)
   newdf.columns = df.columns
   return(newdf)
   

#inputation most_frequent/median/mean	  
def input_most_frequent(df):
   X = np.asarray(s)
   from sklearn.impute import SimpleImputer
   imp_median = SimpleImputer( strategy='most_frequent') #for median imputation replace 'mean' with 'median'
   print(imp_median.fit(X))
   X = imp_median.transform(X)
   df1 = pd.DataFrame(X)
   df1.columns = df.columns
   return(df1)
    


def input_median(df):
   X = np.asarray(s)
   from sklearn.impute import SimpleImputer
   imp_median = SimpleImputer( strategy='median') #for median imputation replace 'mean' with 'median'
   print(imp_median.fit(X))
   X = imp_median.transform(X)
   df1 = pd.DataFrame(X)
   df1.columns = df.columns
   return(df1)



def input_mean(df):
   X = np.asarray(s)
   from sklearn.impute import SimpleImputer
   imp_median = SimpleImputer( strategy='mean') #for median imputation replace 'mean' with 'median'
   print(imp_median.fit(X))
   X = imp_median.transform(X)
   df1 = pd.DataFrame(X)
   df1.columns = df.columns
   return(df1)

# _______________________________ end_functions__________________________________

@advance_inputation.route('/twitter', methods=['POST','GET'])
def receive_data():
    if request.method == 'POST':
        json_data = request.get_json()


        since = json_data["since"] # yyyy-mm-dd
        until = json_data["until"] # yyyy-mm-dd
        QuerySearch = json_data["QuerySearch"] # str, example: 'royal air maroc wifi'
        MaxTweets = json_data["MaxTweets"] # int, example: 10
        Lang = "fr"

        criteria = TweetCriteria()
        criteria = initialize_criteria(criteria,since,until,QuerySearch,MaxTweets,Lang)
        tweets = TweetManager.getTweets (criteria)

        i=0

        
        tweets_list = []
        counts ={"pos":0,"neg":0,"neu":0}
        result = {}

        for tweet in tweets:
            i=i+1
            score=normalize(tweet.text)
            sentiment=''

            if score >= 0.05 : 
                sentiment = "pos"
                counts["pos"] = counts["pos"]+1
            if (score > -0.05 and score < 0.05 ) :
                sentiment = "neu"
                counts["neu"] = counts["neu"]+1
            if score <= -0.05 : 
                sentiment = "neg"
                counts["neg"] = counts["neg"]+1
            
            tweets_list.append({
            'user':tweet.username,
            'text':tweet.text,
            'date':tweet.date.strftime("%Y-%m-%d"),
            # 'location':tweet.geo,
            'sentiment':sentiment
            })
            print(i)
        
        result ["tweets"]=tweets_list
        result ["stats"]=counts

        return jsonify(result)
