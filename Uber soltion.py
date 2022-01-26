#!/usr/bin/env python
# coding: utf-8

# ## The aim of analysis is to identify the root cause of the problem (i.e. cancellation and non-availability of cars) and recommend ways to improve the situation. As a result of your analysis, you should be able to present to the client the root cause(s) and possible hypotheses of the problem(s) and recommend ways to improve them
# 
# 

# # Author: Saurabh Nigam

# In[42]:


import pandas as pd #Importing the Library


# In[43]:


df=pd.read_csv(r"C:\Users\Dhruv Singh\Desktop\ptyhon ass\Uber Request Data.csv")#Importing the Dataset


# In[44]:


df.head()# Reading the data


# ## Analyzing the Data set

# In[45]:


len(df["Request id"].unique()) 


# In[46]:


df.shape


# In[47]:


df.isnull().sum()


# In[48]:


df.isnull().sum()/df.shape[0]*100


# In[49]:


df.info()


# In[50]:


df["Request timestamp"].value_counts()


# In[51]:


df["Request timestamp"]=df["Request timestamp"].astype(str)


# In[52]:


df["Request timestamp"]=df["Request timestamp"].replace("/","-")


# In[53]:


df["Request timestamp"]=pd.to_datetime(df["Request timestamp"],dayfirst=True)


# In[54]:


df.info()


# ## Cleaning the Data Set

# In[55]:


df["Drop timestamp"]=pd.to_datetime(df["Drop timestamp"],dayfirst=True)


# In[56]:


df.info()


# In[57]:


df["Drop timestamp"]


# In[58]:


req_hour=df["Request timestamp"].dt.hour


# In[59]:


len(req_hour)


# In[60]:


df["req_hour"]=req_hour


# In[61]:


req_day=df["Request timestamp"].dt.day


# In[62]:


df["req_day"]=req_day


# In[63]:


import  seaborn as sns


# In[64]:


import matplotlib.pyplot as plt


# ### Visualizing the Data set

# In[65]:


sns.countplot(x="req_hour",data=df,hue="Status")
plt.show()


# In[66]:


sns.factorplot(x="req_hour",data=df,row="req_day",hue="Status",kind="count")
plt.show()


# In[67]:


sns.factorplot(x="req_hour",data=df,row="req_day",hue="Pickup point",kind="count")
plt.show()


# In[68]:


sns.factorplot(x="req_hour",data=df,hue="Pickup point",kind="count")
plt.show()


# In[69]:


df


# In[70]:


df["Time_Slot"]=0


# In[71]:


df


# In[72]:


j=0
for i in df["req_hour"]:
    if df.iloc[j,6]<5:
        df.iloc[j,8]="Pre_Morning"
    elif 5<=df.iloc[j,6]<10:
        df.iloc[j,8]="Morning_Rush"
        
    elif 10<=df.iloc[j,6]<17:
        df.iloc[j,8]="Day_Time"
        
    elif 17<=df.iloc[j,6]<22:
        df.iloc[j,8]="Evening_Rush"
    else:
        df.iloc[j,8]="Late_Night"
    j=j+1


# In[73]:


df


# In[74]:


df["Time_Slot"].value_counts()


# In[75]:


plt.figure(figsize=(10,6))
sns.countplot(x="Time_Slot",hue="Status",data=df)
plt.show()


# In[76]:


df_morning_rush=df[df['Time_Slot']=='Morning_Rush']


# In[77]:


sns.countplot(x="Pickup point",hue="Status",data=df_morning_rush)


# # Severity of problem by location and their count (cancellation of cab as per the pickup location at morning rush hours)

# In[78]:


df_airport_cancelled=df_morning_rush.loc[(df_morning_rush["Pickup point"]=="Airport") & (df_morning_rush["Status"]=="Cancelled")]


# In[79]:


df_airport_cancelled.shape[0]


# In[80]:


df_city_cancelled=df_morning_rush.loc[(df_morning_rush["Pickup point"]=="City") & (df_morning_rush["Status"]=="Cancelled")]


# In[81]:


df_city_cancelled.shape[0]


# # Supply and demand

# In[82]:


df_morning_rush


# In[83]:


df_morning_rush.loc[(df_morning_rush["Pickup point"]=="City")].shape[0]


# In[84]:


df_morning_rush.loc[(df_morning_rush["Pickup point"]=="City") & (df_morning_rush["Status"]=="Trip Completed")].shape[0]


# In[85]:


df_morning_rush.loc[(df_morning_rush["Pickup point"]=="Airport")].shape[0]


# In[86]:


df_morning_rush.loc[(df_morning_rush["Pickup point"]=="Airport") & (df_morning_rush["Status"]=="Trip Completed")].shape[0]


# # Supply and Demand for evening rush

# In[87]:


df_evening_rush=df[df['Time_Slot']=='Evening_Rush']


# In[88]:


df_city_cancelled=df_evening_rush.loc[(df_evening_rush["Pickup point"]=="City") & (df_evening_rush["Status"]=="Cancelled")]


# In[89]:


sns.countplot(x="Pickup point",hue="Status",data=df_evening_rush)


# In[90]:


df_city_cancelled.shape[0]


# In[91]:


df_evening_rush["Status"].value_counts()


# In[92]:


df_evening_rush.loc[(df_evening_rush["Pickup point"]=="City")].shape[0]


# In[93]:


df_evening_rush.loc[(df_evening_rush["Pickup point"]=="City") & (df_evening_rush["Status"]=="Trip Completed")].shape[0]


# In[94]:


df_evening_rush.loc[(df_evening_rush["Pickup point"]=="Airport")].shape[0]


# In[95]:


df_evening_rush.loc[(df_evening_rush["Pickup point"]=="Airport") & (df_evening_rush["Status"]=="Trip Completed")].shape[0]


# # Severity problem at each location by looking at cancellation of cabs in each of the pickup location

# In[96]:


df_evening_rush.loc[(df_evening_rush["Pickup point"]=="Airport") & (df_evening_rush["Status"]=="Cancelled")].shape[0]


# In[97]:


df_evening_rush.loc[(df_evening_rush["Pickup point"]=="City") & (df_evening_rush["Status"]=="Cancelled")].shape[0]


# # Severity of problem by location in morning rush

# In[ ]:




