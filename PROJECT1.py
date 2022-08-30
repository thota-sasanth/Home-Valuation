#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np


# In[4]:


import pandas as pd
pd.set_option('display.max_columns',100)


# In[5]:


from matplotlib import pyplot as plt
import seaborn as sns


# In[6]:


df = pd.read_csv('real_estate_data.csv')


# In[7]:


df.head()


# In[492]:


df.shape


# In[493]:


df.dtypes


# In[494]:


df.tail()


# In[495]:


type(df.dtypes)


# In[496]:


df.dtypes[df.dtypes == 'object']


# In[497]:


df.hist()
plt.show()


# In[498]:


df.year_built.hist()
plt.show()


# In[499]:


df.describe()


# In[500]:


df.describe(include =['object'])


# In[501]:


sns.countplot(y='exterior_walls',data = df)


# In[502]:


for feature in df.dtypes[df.dtypes == 'object'].index:
    sns.countplot(y=feature,data=df)
    plt.show()


# In[503]:


sns.boxplot(y='property_type',x ='tx_price',data = df)


# In[504]:


sns.boxplot(y= 'property_type',x = 'sqft',data =df)


# In[505]:


df.groupby('property_type').mean()


# In[506]:


df.groupby('property_type').agg(['mean','std'])


# In[507]:


correlations =df.corr()


# In[508]:


correlations


# In[509]:


plt.figure(figsize = (9,9))
sns.heatmap(correlations,cmap = 'RdBu_r')


# In[510]:


sns.set_style('white')
plt.figure(figsize=(9,9))
sns.heatmap(correlations,cmap = 'RdBu_r')
plt.show()


# In[511]:


plt.figure(figsize = (10,9))
sns.heatmap(correlations *100,cmap ='RdBu_r',annot =True,fmt ='.0f')
plt.show()


# In[512]:


mask =np.zeros_like(correlations)
mask[np.triu_indices_from(mask)]=1
sns.heatmap(mask)
plt.show()


# In[513]:


plt.figure(figsize = (10,9))
sns.heatmap(correlations *100,cmap ='RdBu_r',annot =True,fmt ='.0f',mask =mask,cbar =False)


# In[514]:


y


# In[515]:


df.basement.unique()


# In[516]:


df['basement']=df.basement.fillna(0)


# In[517]:


df.basement.unique()


# In[518]:


sns.countplot(y ='roof',data =df)


# In[519]:


df.roof.replace('composition','Composition',inplace =True)
df.roof.replace('asphalt','Asphalt',inplace=True)
df.roof.replace(['shake-shingle','asphalt,shake-shingle'],'Shake Shingle',inplace =True)


# In[520]:


sns.countplot(y='roof',data=df)


# In[521]:


df.exterior_walls.replace('Rock, Stone','Masonry',inplace =True)
df.exterior_walls.replace(['Concrete','Block'],'Concrete Block',inplace =True)


# In[522]:


sns.countplot(y='exterior_walls',data =df)


# In[523]:


sns.boxplot(df.tx_price)
plt.xlim(0,1000000)
plt.show()


# In[524]:


sns.violinplot(df.tx_price)
plt.show()
sns.violinplot(df.beds)
plt.show()
sns.violinplot(df.sqft)
plt.show()
sns.violinplot(df.lot_size)
plt.show()


# In[525]:


df.lot_size.sort_values(ascending=False).head()


# In[526]:


df = df[df.lot_size <= 500000]
print(len(df))


# In[527]:


df.select_dtypes(include =['object']).isnull().sum()


# In[528]:


df['exterior_walls'] = df['exterior_walls'].fillna('Missing')


# In[529]:


df.select_dtypes(include =['object']).isnull().sum()


# In[530]:


for column in df.select_dtypes(include =['object']):
    df[column]=df[column].fillna('Missing')


# In[531]:


df.select_dtypes(include =['object']).isnull().sum()


# In[532]:


df.select_dtypes(exclude =['object']).isnull().sum()


# In[533]:


df.to_csv('cleaned_df.csv',index =None)


# In[534]:


df['two_and_two'] = ((df.beds ==2) & (df.baths ==2)).astype(int)


# In[535]:


df.head()


# In[536]:


df.two_and_two.mean()


# In[537]:


df.tail()


# In[538]:


df


# In[539]:


df['during_recession'] = ((df.tx_year >=2010) & (df.tx_year <= 2013)).astype(int)


# In[540]:


print(df.during_recession.mean())


# In[541]:


df['property_age'] = df.tx_year - df.year_built


# In[542]:


df.head()


# In[543]:


df.property_age.min()


# In[544]:


sum(df.property_age<0)


# In[545]:


df = df[df.property_age >=0]


# In[546]:


len(df)


# In[547]:


df['school_score'] = df.num_schools*df.median_school


# In[548]:


df.school_score.median()


# In[549]:


df.to_csv('cleaned_df.csv',index =None)


# In[550]:


sns.countplot(y ='exterior_walls',data =df)


# In[551]:


df.exterior_walls.replace(['Wood Siding','Wood Shingle'],'Wood',inplace =True)


# In[552]:


sns.countplot(y ='exterior_walls',data =df)


# In[553]:


other_exterior_walls = ['Concrete Block','Stucco','Masonry','Other','Asbestos shingle']
df.exterior_walls.replace(other_exterior_walls,'Other',inplace =True)


# In[554]:


sns.countplot(y='exterior_walls',data =df)


# In[555]:


sns.countplot(y='roof',data =df)


# In[556]:


df.roof.replace(['Composition','Wood Shake/ Shingles'],'Composition Shingle',inplace = True )


# In[557]:


sns.countplot(y= 'roof',data =df)


# In[558]:


df.roof.replace(['Gravel/Rock','Roll Composition','Slate','Metal','Asbestos','Built-up'],'Other',inplace =True)


# In[559]:


sns.countplot(y ='roof',data =df)


# In[560]:


df =pd.get_dummies(df,columns =['exterior_walls','roof','property_type'])


# In[561]:


df.head()


# In[562]:


df =df.drop(['tx_year','year_built'],axis =1)


# In[563]:


df.to_csv('analytical_base_table.csv',index ='None')


# In[564]:


df.head()


# In[565]:


df.shape


# In[566]:


from sklearn.linear_model import Lasso


# In[567]:


from sklearn.linear_model import Ridge


# In[568]:


from sklearn.linear_model import ElasticNet


# In[569]:


from sklearn.ensemble import RandomForestRegressor


# In[570]:


from sklearn.ensemble import GradientBoostingRegressor


# In[ ]:





# In[571]:


df.shape


# In[572]:


from sklearn.model_selection import train_test_split


# In[573]:


y =df.tx_price
x= df.drop('tx_price',axis =1)


# In[574]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size =0.2,random_state =1234)


# In[575]:


print(len(x_train),len(x_test),len(y_train),len(y_test))


# In[576]:


x_train.describe()


# In[577]:


x_train_new = (x_train - x_train.mean())/ x_train.std()


# In[578]:


x_train_new.describe()


# In[579]:


x_test_new = (x_test - x_train.mean())/x_train.std()


# In[580]:


x_test_new.describe()


# In[581]:


from sklearn.pipeline import make_pipeline


# In[582]:


from sklearn.preprocessing import StandardScaler


# In[583]:


make_pipeline(StandardScaler(),Lasso(random_state =123))


# In[584]:


type(make_pipeline(StandardScaler(),Lasso(random_state =123)))


# In[585]:


pipelines = {
    'lasso' : make_pipeline(StandardScaler(),Lasso(random_state =123)),
    'ridge' : make_pipeline(StandardScaler(),Ridge(random_state =123)),
    'enet'  : make_pipeline(StandardScaler(),ElasticNet(random_state =123))
}


# In[586]:


pipelines['rf'] = make_pipeline(StandardScaler(),RandomForestRegressor(random_state =123))
pipelines['gb'] = make_pipeline(StandardScaler(),GradientBoostingRegressor(random_state =123))


# In[587]:


len(pipelines)


# In[588]:


for key,value in pipelines.items():
    print(key,type(value))


# In[589]:


pipelines['lasso'].get_params()


# In[590]:


lasso_hyperparams = {
    'lasso__alpha': [0.001,0.005,0.01,0.05,0.1,0.5,1,5,10]
}

ridge_hyperparams = {
    'ridge__alpha': [0.001,0.005,0.1,0.5,0.01,0.05,1,5,10]
}


# In[591]:


enet_hyperparams = {
    'elasticnet__alpha' :[0.001,0.005,0.01,0.05,0.1,0.5,1,5,10],
    'elasticnet__l1_ratio' : [0.1,0.3,0.5,0.7,0.9]
}


# In[592]:


rf_hyperparams = {
    'randomforestregressor__n_estimators' : [100,200],
    'randomforestregressor__max_features' : ['auto','sqrt',0.33]
}


# In[600]:


gb_hyperparams ={
    'gradientboostingregressor__n_estimators': [100,200],
    'gradientboostingregressor__learning_rate' : [0.05,0.1,0.2],
    'gradientboostingregressor__max_depth' : [1,3,5]
}


# In[601]:


hyperparams = {
    'rf' : rf_hyperparams,
    'gb' : gb_hyperparams,
    'lasso' : lasso_hyperparams,
    'ridge' : ridge_hyperparams,
    'enet' : enet_hyperparams
}


# In[602]:


from sklearn.model_selection import GridSearchCV


# In[603]:


model = GridSearchCV(pipelines['lasso'],hyperparams['lasso'],cv=10,n_jobs =-1)


# In[604]:


type(model)


# In[605]:


model.fit(x_train,y_train)


# In[606]:


fitted_models={}
for name,pipeline in pipelines.items():
    model = GridSearchCV(pipeline,hyperparams[name],cv =10,n_jobs =-1)
    model.fit(x_train,y_train)
    fitted_models[name]  = model
    print(name,'has been fitted.')


# In[607]:


for key,value in fitted_models.items():
    print(key,type(value))


# In[608]:


for name,model in fitted_models.items():
    print(name,model.best_score_)


# In[609]:


from sklearn.metrics import r2_score


# In[610]:


from sklearn.metrics import mean_absolute_error


# In[611]:


fitted_models['rf']


# In[612]:


pred = fitted_models['rf'].predict(x_test)


# In[613]:


pred


# In[614]:


print('R^2:',r2_score(y_test,pred))
print('MAE:',mean_absolute_error(y_test,pred))


# In[616]:


for name,model in fitted_models.items():
    pred = model.predict(x_test)
    print(name)
    print('--------')
    print('R^2:',r2_score(y_test,pred))
    print('MAE:',mean_absolute_error(y_test,pred))
    print()


# In[617]:


rf_pred  = fitted_models['rf'].predict(x_test)
plt.scatter(rf_pred,y_test)
plt.xlabel('predicted')
plt.ylabel('actual')
plt.show()


# In[618]:


type(fitted_models['rf'])


# In[619]:


type(fitted_models['rf'].best_estimator_)


# In[620]:


fitted_models['rf'].best_estimator_


# In[621]:


import pickle


# In[622]:


with open ('final_model.pkl','wb') as f:
    pickle.dump(fitted_models['rf'].best_estimator_,f)


# In[ ]:




