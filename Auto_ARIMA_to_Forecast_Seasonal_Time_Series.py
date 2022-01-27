#!/usr/bin/env python
# coding: utf-8

# ## Forecasting the Electric Production Using Time series ARIMA Method

# ### Author : Saurabh Nigam

# # Import Libraries

# In[ ]:


import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose


# ## Read the data

# In[8]:


data=pd.read_csv(r'C:\Users\Dhruv Singh\Desktop\ptyhon ass\Electric_Production.csv',index_col=0)
data.head()


# #Convert Date column as index

# In[9]:


data.index = pd.to_datetime(data.index)


# #Change the name of the variable to Energy Production for ease of understanding

# In[10]:


data.columns = ['Energy Production']


# #Plot the data

# In[11]:


data['Energy Production'].plot(figsize=(12,5))


# #Decompose the data into components

# In[12]:


# ETS Decomposition
result = seasonal_decompose(data['Energy Production'], 
                            model ='multiplicative')
  
# ETS plot 
result.plot()


# #Stationarity Test

# In[13]:


from statsmodels.tsa.stattools import adfuller


# In[14]:


adfuller(data['Energy Production'])


# #The ‘auto_arima’ function from the ‘pmdarima’ library helps us to identify the most optimal parameters for an ARIMA model and returns a fitted ARIMA model.

# In[15]:


get_ipython().system('pip install pmdarima')


# In[37]:


from statsmodels.compat.pandas import Appender


# In[38]:


import pmdarima as pm


# In[39]:


from pmdarima import auto_arima


# In[19]:


stepwise_fit = auto_arima(data['Energy Production'], trace=True,
suppress_warnings=True)


# In[40]:


stepwise_fit.summary()


# #We’ll train from the years 1985–2016 and test our forecast on the years after that and compare it to the real data:

# In[41]:


train = data.loc['1985-01-01':'2016-12-01']
test = data.loc['2017-01-01':]


# In[42]:


print(train.shape,test.shape)


# # Fit ARIMA model to the train dataset

# In[43]:


from statsmodels.tsa.statespace.sarimax import SARIMAX


# In[24]:


model = SARIMAX(train['Energy Production'], 
                order = (2, 1, 4))


# In[44]:


result = model.fit()
result.summary()


# #Predictions of ARIMA model against the test set

# In[45]:


start = len(train)
end = len(train) + len(test) - 1


# In[46]:


predictions = result.predict(start, end,
                             typ = 'levels').rename("Predictions")
  


# In[47]:


# plot predictions and actual values
predictions.plot(legend = True)
test['Energy Production'].plot(legend = True)


# #Evaluate the model using MSE and RMSE

# In[48]:


from sklearn.metrics import mean_squared_error
from statsmodels.tools.eval_measures import rmse


# In[49]:


rmse(test["Energy Production"], predictions)


# In[50]:


mean_squared_error(test["Energy Production"], predictions)


# #Forecast using ARIMA Model

# In[51]:


model = model = SARIMAX(data['Energy Production'], 
                        order = (2, 1, 4))
result = model.fit()


# #Forecast for the next 3 years

# In[52]:


forecast = result.predict(start = len(data), 
                          end = (len(data)-1) + 3 * 12, 
                          typ = 'levels').rename('Forecast')


# ## Plot the forecast values

# In[53]:


data['Energy Production'].plot(figsize = (12, 5), legend = True)
forecast.plot(legend = True)


# In[ ]:





# In[ ]:




