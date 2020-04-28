import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
df=pd.read_csv('time_series_2019-ncov-Confirmed.csv')

#print(data.iloc[:,2:4])

df=df.drop(columns=['Lat','Long']) #Dropping Lat and Long
#print(data)
count = df.iloc[:,2].value_counts()

for i in range(2,df.shape[1]):
	count = df.iloc[:,2].value_counts()
	if count[0]/487>0.7:
		df=df.drop(columns=[df.columns.values[2]])
	else:
		break
#print(count[0]/487)
#print(df)
country=df[['Country/Region','3/22/20']]
#sns.barplot(x='Country/Region',y='3/22/20',data=df)
unique={}
#print(country.iloc[:,0])
for i in range(0,df.shape[0]):
	if country.iloc[i,0] in unique:
   		unique[country.iloc[i,0]]=unique[country.iloc[i,0]]+country.iloc[i,1]
	else:
    		unique[country.iloc[i,0]]=country.iloc[i,1]
unique2={}
unique2['Others']=0
sumval=0
for i in unique: 
        sumval = sumval + unique[i]
for i in unique:
	if (unique[i]/sumval)<=0.01:
		unique2['Others']=unique2['Others']+unique[i]
	else:
		unique2[i]=unique[i]
		
#print(unique)			 
data_items = unique2.items()
data_list = list(data_items)
country = pd.DataFrame(data_list)
country.set_index([0], inplace=True)
#print(country)	
plot=country.plot.pie(y=1,legend=None,figsize=(13,13))
#plot.plot(legend=None)
plot.legend(loc='upper right')
plt.title('Country Wise cases distribution on 22nd March')
plt.show()
#print(df.sum(axis=0))
tp=df.sum(axis=0)
tp=tp.iloc[1:tp.shape[0]]
#tp.set_index([0], inplace=True)
#print(tp)
tp.plot.bar(legend=None,figsize=(10,10))
plt.title('Date wise Total Covid cases in the world')
plt.show()

df2=pd.read_csv('c2c.csv')
tracer={}
continents={}
#print(df2.iloc[3,0])
for i in range(0,df2.shape[0]):
	tracer[df2.iloc[i,1]]=df2.iloc[i,0]
continents['Africa']=0
continents['Asia']=0
continents['Europe']=0
continents['North America']=0
continents['Oceania']=0
continents['South America']=0
for i in unique:
	if i in tracer:
		continents[tracer[i]]=continents[tracer[i]]+unique[i]
	
data_items = continents.items()
data_list = list(data_items)
continents = pd.DataFrame(data_list)
#print(continents)	
continents.set_index([0], inplace=True)	
#print(type(continents))	
plot=continents.plot.pie(y=1,legend=None,figsize=(10,10))
plot.legend(loc='upper left')
plt.title('Continent Wise cases distribution on 22nd March')
plt.show()
	
	
		
