#!/usr/bin/env python
# coding: utf-8

# ### This project is meant to explore, analyse and visualize aviation accidents and related factors such as reasons, survival rates, fatalities, locations etc.

# ### Author : Saurabh Nigam

# ### Import the standard libraries

# In[29]:


import pandas as pd #data processing and I/O operations
import numpy as np #linear algebra
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import date, datetime, timedelta #pip install datatime


# ### Import my datatset

# In[30]:


data = pd.read_csv(r'C:\Users\Dhruv Singh\Desktop\ptyhon ass\Dataset1.csv')


# ### Analizing Data

# In[31]:


data.shape


# In[32]:


data.head(10)


# In[33]:


data.tail(5)


# ### Check for missing values

# In[34]:


data.isnull().sum()


# In[35]:


data['Time'] = data['Time'].replace(np.nan, '00:00')


# In[36]:


data.isnull().sum()


# In[37]:


data['Time'].value_counts()


# ### Standardizing Dataset

# In[38]:


data['Time'] = data['Time'].str.replace('c: ','')
data['Time'] = data['Time'].str.replace('c:','')
data['Time'] = data['Time'].str.replace('c','')
data['Time'] = data['Time'].str.replace('12\'20','12:20')
data['Time'] = data['Time'].str.replace('18.40','18:40')
data['Time'] = data['Time'].str.replace('0943','09:43')
data['Time'] = data['Time'].str.replace('22\'08','22:08')
data['Time'] = data['Time'].str.replace('114:20','00:00')


# In[39]:


data['Time'] = data['Date'] + ' ' +data['Time']

def todate(x):
    return datetime.strptime(x, '%m/%d/%Y %H:%M')

data['Time'] = data['Time'].apply(todate)


# In[40]:


print('Data ranges from ' + str(data.Time.min()) + ' to ' + str(data.Time.max()))


# In[41]:


data.Operator = data.Operator.str.upper()


# In[42]:


data.head()


# ### Exploratory Data Analysis

# ### Total Accidents by year

# In[43]:


Temp = data.groupby(data.Time.dt.year)[['Date']].count()
Temp.head()


# In[44]:


Temp = Temp.rename(columns={'Date':'Count'})


# In[45]:


Temp.head(1)


# In[46]:


plt.figure(figsize=(12,6))
plt.style.use('bmh')
plt.plot(Temp.index, 'Count', data=Temp, color='blue', marker='.', linewidth=1)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Count of accidents by year', fontsize=15)
plt.show()


# In[47]:


import matplotlib.pylab as pl
import matplotlib.gridspec as gridspec


gs = gridspec.GridSpec(2,2)
pl.figure(figsize=(15,10))
plt.style.use('seaborn-muted')
ax = pl.subplot(gs[0,:])
sns.barplot(data.groupby(data.Time.dt.month)[['Date']].count().index, 'Date',
            data = data.groupby(data.Time.dt.month)[['Date']].count(), color='lightskyblue', linewidth=2)
plt.xticks(data.groupby(data.Time.dt.month)[['Date']].count().index, 
           ['Jan', 'Feb', 'March', 'April', 'May', 'June', 'July', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.xlabel('Month', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Count of accidents by month', fontsize=14)

ax = pl.subplot(gs[1,0])
sns.barplot(data.groupby(data.Time.dt.weekday)[['Date']].count().index, 'Date',
            data = data.groupby(data.Time.dt.weekday)[['Date']].count(), color='lightskyblue', linewidth=2)
plt.xticks(data.groupby(data.Time.dt.weekday)[['Date']].count().index, 
           ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
plt.xlabel('Weekday', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Count of accidents by Weekday', fontsize=14)

ax = pl.subplot(gs[1,1])
sns.barplot(data[data.Time.dt.hour != 0].groupby(data.Time.dt.hour )[['Date']].count().index, 'Date',
            data = data[data.Time.dt.hour != 0].groupby(data.Time.dt.hour)[['Date']].count(), color='lightskyblue', linewidth=2)
plt.xlabel('Hour', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Count of accidents by Hour', fontsize=14)
plt.tight_layout()
plt.show()


# ### Passenger vs military flights

# In[48]:


Temp = data.copy()
Temp['isMilitary'] = Temp.Operator.str.contains('MILITARY')
Temp = Temp.groupby('isMilitary')[['isMilitary']].count()
Temp.index = ['Passenger', 'Military']
Temp


# In[49]:


Temp2 = data.copy()
Temp2['Military'] = Temp2.Operator.str.contains('MILITARY')
Temp2['Passenger'] = Temp2.Military == False
Temp2 = Temp2.loc[:,['Time', 'Military', 'Passenger']]
Temp2


# In[50]:


Temp2 = Temp2.groupby(Temp2.Time.dt.year)[['Military', 'Passenger']].aggregate(np.count_nonzero)


# In[51]:


Temp2


# In[52]:


colors = ['yellowgreen', 'lightskyblue']
plt.figure(figsize=(15,6))
plt.subplot(1,2,1)
patches, texts = plt.pie(Temp.isMilitary, colors=colors, labels=Temp.isMilitary, startangle=90)
plt.legend(patches, Temp.index, fontsize=12)
plt.axis('equal')
plt.title('Total number of accidents by flight type', fontsize=15)

plt.subplot(1,2,2)
plt.plot(Temp2.index, 'Military', data=Temp2, color='lightskyblue', marker='.', linewidth=1)
plt.plot(Temp2.index, 'Passenger', data=Temp2, color='yellowgreen', marker='.', linewidth=1)
plt.legend(fontsize=12)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Count of accidents by flight type', fontsize=15)
plt.tight_layout()
plt.show()


# ### Total Number of fatalities

# In[53]:


Fatalities = data.groupby(data.Time.dt.year).sum()
Fatalities['Proportion'] = Fatalities['Fatalities'] / Fatalities['Aboard']

plt.figure(figsize=(15,6))
plt.subplot(1,2,1)
plt.fill_between(Fatalities.index, 'Aboard', data=Fatalities, color='skyblue', alpha=0.2)
plt.plot(Fatalities.index, 'Aboard', data=Fatalities, marker='.', color='Slateblue', alpha=0.6, linewidth=1)

plt.fill_between(Fatalities.index, 'Fatalities', data=Fatalities, color='olive', alpha=0.2)
plt.plot(Fatalities.index, 'Fatalities', data=Fatalities, marker='.', color='olive', alpha=0.6, linewidth=1)

plt.legend(fontsize=12)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Number of People', fontsize=12)
plt.title('Total number of Fatalities by Year')



plt.subplot(1,2,2)
plt.plot(Fatalities.index, 'Proportion', data=Fatalities, marker='.', color='red', linewidth=2)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Fatalities ratio', fontsize=12)
plt.title('Fatalities ratio by year', fontsize=15)
plt.show()


# ### Dataset 2 import

# In[54]:


Totals = pd.read_csv('Dataset2.csv')
Totals.head()


# In[ ]:


Totals = Totals.drop(['Country Name', 'Country Code', 'Indicator Code', 'Indicator Name'], axis=1)


# In[ ]:


Totals = Totals.replace(np.nan, 0)


# In[ ]:


Totals = pd.DataFrame(Totals.sum())


# In[ ]:


Totals.tail()


# In[ ]:


Totals = Totals.drop(Totals.index[0:10])
Totals = Totals['1970':'2008']
Totals.columns = ['Sum']
Totals.index.name = 'Year'


# In[ ]:


Totals.head()


# In[ ]:


Fatalities = Fatalities.reset_index()


# In[ ]:


Fatalities.head()


# In[ ]:


Fatalities.Time = Fatalities.Time.apply(str)
Fatalities.index = Fatalities['Time']
del Fatalities['Time']
Fatalities = Fatalities['1970':'2008']
Fatalities = Fatalities[['Fatalities']]
Totals = pd.concat([Totals,Fatalities], axis=1)
Totals['Ratio'] = Totals['Fatalities'] / Totals['Sum'] * 100


# In[ ]:


Totals.head()


# In[ ]:


gs = gridspec.GridSpec(2,2)
pl.figure(figsize=(15,10))

ax= pl.subplot(gs[0,0])
plt.plot(Totals.index, 'Sum', data=Totals, marker='.', color='green', linewidth=1)
plt.xlabel('Year')
plt.ylabel('Number of passengers')
plt.title('Total number of passengers by Year', fontsize=15)
plt.xticks(rotation=90)

x= pl.subplot(gs[0,1])
plt.plot(Fatalities.index, 'Fatalities', data=Totals, marker='.', color='red', linewidth=1)
plt.xlabel('Year')
plt.ylabel('Number of Deaths')
plt.title('Total number of Deaths by Year', fontsize=15)
plt.xticks(rotation=90)

x= pl.subplot(gs[1,:])
plt.plot(Totals.index, 'Ratio', data=Totals, marker='.', color='orange', linewidth=1)
plt.xlabel('Year')
plt.ylabel('Ratio')
plt.title('Fatalities/Total number of passengers ratio by Year', fontsize=15)
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# ### Plot ratio and number of deaths in one plot

# In[ ]:


fig = plt.figure(figsize=(12,6))
ax1 = fig.subplots()
ax1.plot(Totals.index, 'Ratio', data=Totals, color='orange', marker='.', linewidth=1)
ax1.set_xlabel('Year', fontsize=12)
for label in ax1.xaxis.get_ticklabels():
    label.set_rotation(45)
ax1.set_ylabel('Ratio', color='orange', fontsize=12)
ax1.tick_params('y', colors='orange')
ax2 = ax1.twinx()
ax2.plot(Fatalities.index, 'Fatalities', data=Fatalities, color='green', marker='.', linewidth=1)
ax2.set_ylabel('Number of Fatalities', color='green', fontsize=12)
ax2.tick_params('y', colors='g')
plt.title('Fatalities VS Ratio by year', fontsize=15)
plt.tight_layout()
plt.show()


# ### Operator Analysis

# In[ ]:


data.Operator = data.Operator.str.upper()
data.Operator = data.Operator.replace("A B AEROTRANSPORT", 'AB AEROTRANSPORT')

Total_by_Op = data.groupby('Operator')[['Operator']].count()
Total_by_Op = Total_by_Op.rename(columns={'Operator':'Count'})
Total_by_Op = Total_by_Op.sort_values(by='Count', ascending=False).head(15)


# In[ ]:


Total_by_Op


# In[ ]:


plt.figure(figsize=(12,6))
sns.barplot(y=Total_by_Op.index, x='Count', data=Total_by_Op, palette='gist_heat', orient='h')
plt.xlabel('Count', fontsize=12)
plt.ylabel('Operator', fontsize=12)
plt.title("Total Count of the Operator", fontsize=15)
plt.show()


# In[ ]:


Prop_by_Op = data.groupby('Operator')[['Fatalities']].sum()
Prop_by_Op = Prop_by_Op.rename(columns={'Operator':'Fatalities'})
Prop_by_Op = Prop_by_Op.sort_values(by='Fatalities', ascending=False)
Prop_by_OpTop = Prop_by_Op.head(15)


# In[ ]:


plt.figure(figsize=(12,6))
sns.barplot(y=Prop_by_OpTop.index, x='Fatalities', data=Prop_by_OpTop, palette='gist_heat', orient='h')
plt.xlabel('Fatalities', fontsize=12)
plt.ylabel('Operator', fontsize=12)
plt.title("Total Fatalities of the Operator", fontsize=15)
plt.show()


# In[ ]:


Prop_by_Op[Prop_by_Op['Fatalities'] == Prop_by_Op.Fatalities.min()].index.tolist()


# In[ ]:


Aeroflot = data[data.Operator == 'AEROFLOT']
Count_by_year = Aeroflot.groupby(data.Time.dt.year)[['Date']].count()
Count_by_year = Count_by_year.rename(columns={'Date':'Count'})

plt.figure(figsize=(12,6))
plt.plot(Count_by_year.index, 'Count', data=Count_by_year, marker='.', color='red', linewidth=1)
plt.xlabel('Year', fontsize=11)
plt.ylabel('Count', fontsize=11)
plt.title('Count of accidents by year (Aeroflot)', fontsize=16)
plt.show()


# In[ ]:




