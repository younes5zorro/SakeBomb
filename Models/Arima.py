import pandas as pd
import numpy as np

from pyramid.arima import auto_arima

file ="https://raw.githubusercontent.com/Pierian-Data/AutoArima-Time-Series-Blog/master/Electric_Production.csv"

data = pd.read_csv(file,index_col=0)

data.index = pd.to_datetime(data.index)

data['IPG2211A2N'].plot(figsize=(12,5))

stepwise_model = auto_arima(data, start_p=1, start_q=1,
                           max_p=3, max_q=3, 
                           start_P=0, seasonal=False,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)
                           
​
print("order => ",stepwise_model.order)

​test_size = 0.8

size = int(len(data) * test_size)
train, test = data[0:size], data[size:]

train.columns = ['training']
test.columns = ['test']


print("test => ",len(test))

stepwise_model.fit(train)

future_forecast = stepwise_model.predict(n_periods=len(test))

future_forecast = pd.DataFrame(future_forecast,index = test.index,columns=['Prediction'])

pd.concat([test,future_forecast],axis=1).plot(figsize=(12,5))

pd.concat([train,test,future_forecast],axis=1).plot(figsize=(12,5))

import math
from sklearn.metrics import mean_squared_error,r2_score

rms = math.sqrt(mean_squared_error(test, future_forecast))
r2 = r2_score(test, future_forecast)

print(rms)
r2
