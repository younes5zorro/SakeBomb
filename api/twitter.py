from GetOldTweets3.manager.TweetCriteria import TweetCriteria
from GetOldTweets3.manager.TweetManager import TweetManager
import re
from googletrans import Translator
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from flask import request, Blueprint,jsonify

# _____________requirements:_____________
# GetOldTweets3==0.0.9
# googletrans==2.4.0
# vaderSentiment==3.2.1
#_________________________________________

advance_tweet = Blueprint('advance_tweet', __name__)
analyzer = SentimentIntensityAnalyzer()
# _________________________________functions_____________________________________
def special_removal(text):
    return [re.sub(r"[^a-zA-Z0-9]+", ' ', k) for k in text.split("\n")][0]

def remove_links(text):
    result = re.sub(r'http\S+', '', text)
    return result

def translate_teweet(text):
    translator = Translator()
    translation = translator.translate(text, dest='en')
    return translation.text
    

def normalize(text,lang):
    if lang != "en":
        text= translate_teweet(text)
    text=remove_links(text)
    text=special_removal(text)
    vs = analyzer.polarity_scores(text)
    return vs["compound"]


def initialize_criteria(criteria,since,until,QuerySearch,MaxTweets,Lang):
    criteria.setSince(str(since)) #"yyyy-mm-dd"
    criteria.setUntil (str(until)) #"yyyy-mm-dd"
    criteria.setQuerySearch (QuerySearch) #"royal air maroc wifi"
    criteria.setMaxTweets(MaxTweets)
    criteria.setLang(Lang) ##"fr"
    return criteria

# _______________________________ end_functions__________________________________

@advance_tweet.route('/v1/twitter', methods=['POST','GET'])
def receive_data():
    if request.method == 'POST':
        json_data = request.get_json()


        since = json_data["since"] # yyyy-mm-dd
        until = json_data["until"] # yyyy-mm-dd
        QuerySearch = json_data["QuerySearch"] # str, example: 'royal air maroc wifi'
        MaxTweets = json_data["MaxTweets"] # int, example: 10
        Lang = json_data["lang"] # int, example: 10

        criteria = TweetCriteria()
        criteria = initialize_criteria(criteria,since,until,QuerySearch,MaxTweets,Lang)
        tweets = TweetManager.getTweets (criteria)

        i=0

        
        tweets_list = []
        counts ={"Positif":0,"Neutre":0,"Negatif":0}
        result = {}

        for tweet in tweets:
            i=i+1
            score=normalize(tweet.text,Lang)
            sentiment=''

            if score >= 0.05 : 
                sentiment = "Positif"
                counts["Positif"] = counts["Positif"]+1
            if (score > -0.05 and score < 0.05 ) :
                sentiment = "Neutre"
                counts["Neutre"] = counts["Neutre"]+1
            if score <= -0.05 : 
                sentiment = "Negatif"
                counts["Negatif"] = counts["Negatif"]+1
            
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
