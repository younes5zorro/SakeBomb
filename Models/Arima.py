import pandas as pd

from pyramid.arima import auto_arima
from sklearn.metrics import mean_squared_error,r2_score
from io import BytesIO
import requests
import math 
# file ="https://raw.githubusercontent.com/Pierian-Data/AutoArima-Time-Series-Blog/master/Electric_Production.csv"

# data = pd.read_csv(file,index_col=0)

def get_dataFram(json_data):

        df = {}
        if  json_data['type'] =="excel":
                df = get_df_from_excel(json_data['link'],json_data['sheet'])
        elif  json_data['type'] =="csv":
                df = get_df_from_csv(json_data['link'])

        # input_field  = json_data["input"]
        # input_field.append(json_data["output"])

        # df = df[input_field]

        train_labels=df.iloc[:,-1]

        return df,train_labels

def get_df_from_excel(link,sheet):

        file = requests.get(link).content
        
        df =pd.read_excel(BytesIO(file),sheet_name =sheet,index_col=0)

        return df

def get_df_from_csv(link):

        file = requests.get(link).content
        
        df =pd.read_csv(BytesIO(file),index_col=0)

        return df

def splitData(df, test_size):

        #     df.index = df.iloc[:,0]    
    size = int(len(df) * test_size)
    print(size)
    train, test = df[:size], df[size:]
    
    return df,train, test

def xtestSetPrediction(X_test,clf):

    predict_test = clf.predict(n_periods=len(X_test))
    predict_test = pd.DataFrame(predict_test,index = X_test.index,columns=['Prediction'])

    return predict_test

def autotuning(data,train, seasonal):

    clf = auto_arima(data, start_p=1, start_q=1,
                           max_p=3, max_q=3,
                           max_d=0, start_d=2, 
                           start_P=0, seasonal=seasonal,
                           m=12,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)

    clf.fit(train)
    return clf
    
def scoring(y_test,predict_test,clf):   

    data={}

    data["p"],data["d"],data["q"] = clf.order    
    data["Status"]  ="Train"

    data["test_data"] = y_test.iloc[:,0].tolist()
    data["predict_test"] = predict_test.iloc[:,0].tolist()
    data["index_data"] = y_test.index.astype(str).tolist()
    
    data["Accuracy Trainning"] = round(r2_score(y_test,predict_test),2)**2

    if data["Accuracy Trainning"] >= 0.8:
            data["Etat Trainning"] = "Excellent"
    elif data["Accuracy Trainning"] >= 0.6:
            data["Etat Trainning"] = "Moyen"
    else :
            data["Etat Trainning"] = "Mauvais"  

    return data

